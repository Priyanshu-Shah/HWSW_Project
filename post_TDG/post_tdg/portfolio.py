"""Budget-driven solver portfolio: PEFT, the GNN-seeded GA / SA (`fastheur`) and CP-SAT (`solvers.exact`).

Ported from `combo/partition.py` of the integrated package v2 (`../HWSW_integrated_v2/`, see its `PARTITION.md`
and `docs/person3_results.md`). The rules in `RULES` come from those measurements (all synthetic graphs). Every
candidate is scored with `sim.evaluate`; the best feasible one wins (else the least violating), and a proven
optimum wins a tie. Decision D8 in `decisions.md`.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field

import numpy as np

from estimate.annotate import Problem
from sim import evaluate
from solvers.heuristics import Plan

# Ratios below are makespan / optimum (or best known); Thor run 2, final model, 3 seeds, unless noted.
RULES = {
    # Below this budget: SA from HEFT. The GNN's 5-15 ms prior can cost more than it gives at 10 ms on slower
    # machines (Thakar's laptop, p <= 0.012), so the safe choice is kept.
    "min_budget_for_gnn_s": 0.05,
    # GA seeded with the GNN population helps up to 200 tasks (25-40: 1.043 -> 1.023 at 1 s; 100: 1.321 ->
    # 1.052 at 1 s; 200: 1.076 -> 1.048 at 1 s); 500 mixed, 1000 inconclusive.
    "gnn_max_tasks": 200,
    # CP-SAT proves 10-40-task graphs optimal in well under a second.
    "exact_max_tasks_small": 40,
    "exact_min_budget_small_s": 2.0,
    # 41-100 tasks with >= 10 s: SA 1 s + GA/GNN 1 s, then CP-SAT warm-started (Soham round 1: mean 1.040).
    "exact_max_tasks_large": 100,
    "exact_min_budget_large_s": 10.0,
}


@dataclass
class Candidate:
    name: str
    plan: Plan
    makespan_ms: float
    feasible: bool
    status: str = "heuristic"  # "optimal" / "feasible" (from CP-SAT) / "heuristic"


@dataclass
class Outcome:
    best: Candidate | None
    candidates: list[Candidate] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    runtime_s: float = 0.0
    bound_ms: float | None = None  # CP-SAT bound, when the exact phase ran
    gap: float | None = None
    exact_ran: bool = False


_WARM: dict = {}


def warmup(problem: Problem | None = None, gnn=None) -> None:
    """One-time costs kept out of the budget: numba compilation, OR-Tools import, the first torch pass."""
    if not _WARM.get("numba"):
        from fastheur.fast import warmup as compile_kernels

        compile_kernels()
        import solvers.exact  # noqa: F401  (OR-Tools import, 1-4 s the first time)

        _WARM["numba"] = True
    if gnn is not None and problem is not None and not _WARM.get("torch"):
        with contextlib.suppress(Exception):
            gnn.decode(problem, gnn.featurize(problem))
        _WARM["torch"] = True


def solve(problem: Problem, graph: dict, platform: dict, gnn, budget_s: float, seed: int = 0) -> Outcome:
    """Best plan within about `budget_s` seconds (after `warmup`). `graph` / `platform` are the IR JSON dicts."""
    from fastheur import bridge
    from fastheur.algos import METHODS
    from fastheur.fast import Fast
    from fastheur.instance import from_graph_platform

    warmup(problem, gnn)
    t0 = time.perf_counter()
    out = Outcome(None)

    def left() -> float:
        return budget_s - (time.perf_counter() - t0)

    def add(name: str, plan: Plan, status: str = "heuristic") -> None:
        ev = evaluate(problem, plan.mapping, priority=plan.priority, insertion=plan.insertion)
        c = Candidate(name, plan, ev.sim.makespan, bool(ev.feasible), status)
        out.candidates.append(c)
        if _better(out.best, c):
            out.best = c

    try:
        inst = from_graph_platform(graph, platform)
        fast = Fast(inst)
    except Exception as failure:  # noqa: BLE001 - a case the compiled model cannot represent
        from solvers.heuristics import plan as handover_plan

        out.steps.append(f"compiled heuristics unavailable ({type(failure).__name__}); handover heuristics only")
        for name in ("heft", "peft", "magellan"):
            if name != "magellan" or left() >= 0.5:
                add(name, handover_plan(problem, name))
        out.runtime_s = time.perf_counter() - t0
        return out

    def from_fast(name: str, found) -> None:
        if found.plan is None or found.makespan == float("inf"):
            return
        add(name, Plan(bridge.from_opt(inst, problem, found.plan.opt), bridge.handover_priority(inst, found.plan.order),
                       insertion=found.plan.insertion))

    n, R = inst.n, RULES
    exact_small = n <= R["exact_max_tasks_small"] and budget_s >= R["exact_min_budget_small_s"]
    exact_large = not exact_small and n <= R["exact_max_tasks_large"] and budget_s >= R["exact_min_budget_large_s"]
    use_gnn = gnn is not None and n <= R["gnn_max_tasks"] and budget_s >= R["min_budget_for_gnn_s"]

    # 0. Safety net: compiled PEFT (ms), also an extra start for the searches.
    peft = METHODS["peft"](inst, fast, seed, 3600.0)
    from_fast("peft", peft)
    extra = []
    if peft.plan is not None and peft.makespan < float("inf"):
        order = bridge.handover_priority(inst, peft.plan.order)
        keys = bridge.keys_from_task_priority(inst, bridge.task_order(order)) if order else np.linspace(0, 1, n)
        extra = [(peft.plan.opt.copy(), keys)]
    data = gnn.featurize(problem) if use_gnn else None

    def gnn_population(share_s: float):
        def population(fast_):
            sampler = bridge.GnnSampler(gnn, problem, data, inst)
            cap = min(0.3 * share_s, 0.25)
            return [(c, k) for c, k, _, _ in sampler.population(fast_, m=256, keep=8, seed=seed, time_cap_s=cap)] + extra

        return population

    # 1. Heuristic phase.
    if 0 < left() < R["min_budget_for_gnn_s"]:
        out.steps.append(f"SA from HEFT, {max(left(), 0) * 1e3:.0f} ms (budget too small for the GNN)")
        from_fast("sa", METHODS["sa"](inst, fast, seed, max(left(), 0.001)))
    elif exact_large:
        out.steps.append("SA from HEFT, 1 s (CP-SAT warm start)")
        from_fast("sa", METHODS["sa"](inst, fast, seed, 1.0))
        if use_gnn:
            out.steps.append("GA from the GNN population, 1 s (CP-SAT warm start)")
            with _extra_starts(gnn_population(1.0)):
                from_fast("ga+gnn", METHODS["ga"](inst, fast, seed, 1.0))
    elif left() > 0:
        share = min(max(0.2 * left(), 0.1), 1.0) if exact_small else left()
        if use_gnn:
            out.steps.append(f"GA from the GNN population, {share * 1e3:.0f} ms")
            with _extra_starts(gnn_population(share)):
                from_fast("ga+gnn", METHODS["ga"](inst, fast, seed, share))
        else:
            why = "no model" if gnn is None else f"{n} tasks > {R['gnn_max_tasks']}"
            out.steps.append(f"GA from HEFT + PEFT, {share * 1e3:.0f} ms (GNN skipped: {why})")
            with _extra_starts((lambda fast_: extra) if extra else None):
                from_fast("ga", METHODS["ga"](inst, fast, seed, share))

    # 2. Exact phase: CP-SAT warm-started with every candidate.
    if (exact_small or exact_large) and left() > 0.5:
        from solvers.exact import solve_exact

        limit = max(left() - 0.2, 0.3)
        out.steps.append(f"CP-SAT, {limit:.1f} s, warm starts {[c.name for c in out.candidates]}")
        r = solve_exact(problem, time_limit_s=limit, warm_starts=[c.plan for c in out.candidates],
                        workers=min(8, os.cpu_count() or 2), seed=seed)
        out.exact_ran, out.bound_ms, out.gap = True, r.bound_ms, r.gap
        if r.plan is not None:
            add("exact", r.plan, "optimal" if r.status == "optimal" else "feasible")
    out.runtime_s = time.perf_counter() - t0
    return out


def _better(current: Candidate | None, new: Candidate) -> bool:
    if current is None:
        return True
    if (not new.feasible, new.makespan_ms) < (not current.feasible, current.makespan_ms):
        return True
    tie = new.feasible == current.feasible and new.makespan_ms <= current.makespan_ms * (1 + 1e-9)
    return tie and new.status == "optimal" != current.status


@contextlib.contextmanager
def _extra_starts(population):
    """Add extra start solutions to the fast SA / tabu (`_warm_start`) and GA (`_seeds`) for one call."""
    from fastheur.algos import evo, local

    warm0, seeds0 = local._warm_start, evo._seeds

    def warm(inst, fast, tracker):
        opt, makespan, excess = warm0(inst, fast, tracker)
        starts = population(fast)
        for c, _ in starts:
            tracker.offer(*fast.evaluate(c), c)
        if not starts:
            return opt, makespan, excess
        first = starts[0][0]
        return (first.copy(), *fast.evaluate(first))

    def seeds(inst, fast, tracker):
        out = seeds0(inst, fast, tracker)
        for c, k in population(fast):
            tracker.offer(*fast.evaluate(c), c)
            out.append((c.copy(), k.copy()))
        return out

    if population is not None:
        local._warm_start, evo._seeds = warm, seeds
    try:
        yield
    finally:
        local._warm_start, evo._seeds = warm0, seeds0


__all__ = ["RULES", "Candidate", "Outcome", "solve", "warmup"]
