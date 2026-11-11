"""One entry point for the exact component: certified schedules, bounds and infeasibility proofs.

`solve_exact(problem, time_limit_s, warm_starts)` picks the method by structure and size:
  1. mapping relaxation (`relaxation.py`): a valid lower bound; if it is infeasible, no mapping can meet the
     deadline / capacity (an infeasibility proof with exact durations);
  2. incumbents: the caller's warm starts (heuristic or ML plans) and HEFT, scored by the simulator;
  3. a chain with at most one capacity dimension: the exact DP (`chain_dp.py`), optimal immediately;
  4. CP-SAT with bus contention, hinted with the best incumbent's mapping (mapping only: start hints triggered a
     false optimum in OR-Tools 9.15). Up to `small` tasks it gets the whole budget and normally proves optimality;
  5. larger or still unproven: relax-and-schedule (`hybrid.py`) for the rest of the budget.
Every schedule is re-scored by the simulator (exact durations, contention); the result is a `Plan` that replays it.
Bounds from CP-SAT get a margin of 1 µs per task (it schedules whole µs). A bound above a known schedule is a
contradiction: it is dropped and reported in `certificate` instead of being trusted.

`certify(problem, plan)` scores any plan and states its gap to the relaxation bound.
`run(problem, ...)` wraps `solve_exact` into the pipeline's `Result` contract (strong lower bound in diagnostics).
Only latency-only objectives get bounds and the portfolio; other objectives go to CP-SAT alone.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from estimate.annotate import Problem
from schemas.result import Result, SolverInfo
from sim import evaluate
from sim.evaluate import Evaluation
from solvers.common import build_result
from solvers.cp import solve_cp
from solvers.exact.chain_dp import chain_dp
from solvers.exact.hybrid import relax_and_schedule
from solvers.exact.relaxation import relaxation_bound
from solvers.heuristics.list_sched import Plan, heft

REL_TOL = 1e-3  # 0.1%: CP-SAT schedules whole µs, finer optimality claims are not meaningful


@dataclass
class ExactResult:
    status: str  # optimal | feasible | infeasible | unknown (no feasible schedule found, infeasibility not proven)
    plan: Plan | None  # replays the schedule: evaluate(problem, plan.mapping, priority=plan.priority, insertion=...)
    makespan_ms: float | None  # simulator makespan of `plan`
    bound_ms: float | None  # valid lower bound on the optimal makespan
    method: str | None  # which component produced `plan`
    certificate: list[str] = field(default_factory=list)  # how the status / bound were established
    runtime_s: float = 0.0
    evaluation: Evaluation | None = None

    @property
    def gap(self) -> float | None:
        if self.makespan_ms is None or self.bound_ms is None or self.makespan_ms <= 0:
            return None
        return max(0.0, (self.makespan_ms - self.bound_ms) / self.makespan_ms)


def _cp_bound_ms(problem: Problem, bound: float | None) -> float | None:
    if bound is None or not math.isfinite(bound):
        return None
    return bound * problem.norms["latency"] - 1e-3 * len(problem.order)


def certify(problem: Problem, plan: Plan, time_limit_s: float = 10.0) -> ExactResult:
    """Score a plan (from any solver) with the simulator and bound its gap with the mapping relaxation."""
    t0 = time.perf_counter()
    ev = evaluate(problem, plan.mapping, priority=plan.priority, insertion=plan.insertion)
    rb = relaxation_bound(problem, time_limit_s=time_limit_s)
    if rb.status == "infeasible":
        return ExactResult("infeasible", plan, ev.sim.makespan, None, "given", rb.notes, time.perf_counter() - t0, ev)
    bound = rb.bound_ms if rb.bound_ms is not None and math.isfinite(rb.bound_ms) else None
    cert = [f"mapping relaxation bound {bound:.6g} ms ({rb.status})"] if bound is not None else []
    out = ExactResult("feasible" if ev.feasible else "unknown", plan, ev.sim.makespan,
                      min(bound, ev.sim.makespan) if bound is not None else None, "given", cert,
                      time.perf_counter() - t0, ev)
    if ev.feasible and out.gap is not None and out.gap <= REL_TOL:
        out.status = "optimal"
    return out


def solve_exact(
    problem: Problem,
    time_limit_s: float = 60.0,
    warm_starts: Sequence[Plan] = (),
    workers: int = 8,
    seed: int = 0,
    small: int = 40,
) -> ExactResult:
    t0 = time.perf_counter()

    def left() -> float:
        return max(0.0, time_limit_s - (time.perf_counter() - t0))

    cands: list[tuple[float, Plan, str, Evaluation]] = []

    def add(plan: Plan | None, method: str) -> None:
        if plan is None:
            return
        ev = evaluate(problem, plan.mapping, priority=plan.priority, insertion=plan.insertion)
        if ev.feasible:
            cands.append((ev.sim.makespan, plan, method, ev))

    if any(not problem.options[t] for t in problem.order):
        return ExactResult("infeasible", None, None, None, None, ["some task has no legal option"],
                           time.perf_counter() - t0)
    for w in warm_starts:
        add(w, "warm start")
    try:
        add(heft(problem), "HEFT")
    except ValueError:
        pass
    best_hint = min(cands, key=lambda c: c[0])[1].mapping if cands else None

    if set(problem.weights) != {"latency"}:  # weighted objective: CP-SAT alone, no makespan certificate
        out = solve_cp(problem, time_limit_s=max(1.0, left()), workers=workers, seed=seed, hint=best_hint,
                       contention=True)
        if out.mapping is not None:
            add(Plan(out.mapping, out.priority(problem)), "CP-SAT")
        return _finish(problem, cands, None, out.status == "infeasible", out.status == "optimal",
                       [f"CP-SAT {out.status} on the weighted objective {problem.weights}"], t0)

    cert: list[str] = []
    rb = relaxation_bound(problem, time_limit_s=max(1.0, min(20.0, 0.15 * time_limit_s)), seed=seed)
    if rb.status == "infeasible":
        return ExactResult("infeasible", None, None, None, None,
                           ["mapping relaxation infeasible: no mapping meets the deadline / capacity "
                            "(exact durations, valid for any schedule)"], time.perf_counter() - t0)
    bound = rb.bound_ms if rb.bound_ms is not None and math.isfinite(rb.bound_ms) else 0.0
    cert.append(f"mapping relaxation: bound {bound:.6g} ms ({rb.status}, {rb.runtime_s:.1f} s)")

    try:  # chains with <= 1 capacity dimension: exact DP
        m, val = chain_dp(problem)
        add(Plan(m), "chain DP")
        return _finish(problem, cands, val, False, True, cert + ["chain DP: exact for chains"], t0)
    except ValueError:
        pass

    n = len(problem.order)
    out = solve_cp(problem, time_limit_s=max(1.0, left() if n <= small else 0.4 * left()), workers=workers,
                   seed=seed, hint=best_hint, contention=True)
    if out.mapping is not None:
        add(Plan(out.mapping, out.priority(problem)), "CP-SAT")
    proven = out.status == "optimal"
    if out.status in ("optimal", "feasible"):
        cb = _cp_bound_ms(problem, out.makespan / problem.norms["latency"] if proven else out.bound)
        if cb is not None:
            bound = max(bound, cb)
        cert.append(f"CP-SAT {out.status} ({out.runtime_s:.1f} s)" + (", µs-rounded model" if proven else ""))
    elif out.status == "infeasible":
        if problem.deadline_ms is None:
            return _finish(problem, cands, None, True, False, cert + ["CP-SAT proved infeasibility"], t0)
        # CP-SAT rounds durations to whole µs: re-check with n µs of deadline slack against the exact simulator
        slack = dataclasses.replace(problem, deadline_ms=problem.deadline_ms + 1e-3 * n, _norms=None)
        s = solve_cp(slack, time_limit_s=max(1.0, min(10.0, left())), workers=workers, seed=seed, contention=True)
        if s.mapping is not None:
            add(Plan(s.mapping, s.priority(slack)), "CP-SAT (µs slack)")
        if not cands:
            return _finish(problem, cands, None, s.status == "infeasible", False,
                           cert + ["CP-SAT infeasible on the µs-rounded model; also with n µs deadline slack"
                                   if s.status == "infeasible" else "CP-SAT infeasible; slack check inconclusive"],
                           t0)
        cert.append("CP-SAT infeasible only through µs rounding; schedule found with slack meets the exact deadline")

    if not proven and left() > 2.0:
        h = relax_and_schedule(problem, time_limit_s=left(), workers=workers, seed=seed)
        if h.mapping is not None:
            add(Plan(h.mapping, h.priority), "relax-and-schedule")
        if h.bound is not None and math.isfinite(h.bound):
            bound = max(bound, h.bound)
        cert.append(f"relax-and-schedule {h.status} ({h.runtime_s:.1f} s)")
    return _finish(problem, cands, bound, False, proven, cert, t0)


def _finish(problem: Problem, cands, bound: float | None, infeasible: bool, proven: bool, cert: list[str],
            t0: float) -> ExactResult:
    rt = time.perf_counter() - t0
    if not cands:
        return ExactResult("infeasible" if infeasible else "unknown", None, None, bound or None, None, cert, rt)
    ms, plan, method, ev = min(cands, key=lambda c: c[0])
    if bound is not None and bound > ms * (1 + REL_TOL):
        cert.append(f"inconsistent bound {bound:.6g} > schedule {ms:.6g} ms dropped (solver numerics)")
        bound = None
    if bound is not None:
        bound = min(bound, ms)
    res = ExactResult("feasible", plan, ms, bound, method, cert, rt, ev)
    if proven or (res.gap is not None and res.gap <= REL_TOL):
        res.status = "optimal"
    return res


def run(problem: Problem, time_limit_s: float = 60.0, warm_starts: Sequence[Plan] = (), workers: int = 8,
        seed: int = 0, n_samples: int = 1000) -> Result:
    """`solve_exact` as a pipeline `Result` (metrics with intervals, diagnostics with the strong lower bound)."""
    r = solve_exact(problem, time_limit_s=time_limit_s, warm_starts=warm_starts, workers=workers, seed=seed)
    status = {"optimal": "optimal", "feasible": "feasible", "infeasible": "infeasible"}.get(r.status, "time_limit")
    info = SolverInfo(name="exact", status=status, runtime_s=r.runtime_s, seed=seed,  # type: ignore[arg-type]
                      objective=r.makespan_ms,
                      params={"time_limit_s": time_limit_s, "method": r.method, "bound_ms": r.bound_ms,
                              "gap": r.gap, "warm_starts": len(warm_starts)},
                      notes="; ".join(r.certificate) or None)
    plan = r.plan
    return build_result(problem, plan.mapping if plan else None, info, priority=plan.priority if plan else None,
                        insertion=plan.insertion if plan else False, n_samples=n_samples, seed=seed,
                        strong_bound=True)
