"""Relax-and-schedule matheuristic for large instances, with a certified gap.

1. Solve the mapping relaxation (`relaxation.py`) for a share of the budget: its dual bound is a valid lower bound,
   and its incumbent mapping is usually a much better mapping than list-scheduling heuristics find.
2. Schedule that mapping exactly: CP-SAT with the mapping fixed and bus contention on.
3. Hand the mapping to full CP-SAT (mapping free) as a hint for the rest of the budget. Mapping only: with start
   times also hinted, CP-SAT 9.15's presolve declared a false optimum (Tier A large seed 5003: "optimal" 22.315 ms
   where 20.164 ms is feasible in the same model; gone with presolve off or without start hints; DECISIONS
   2026-11-06).
The best simulator-checked schedule from steps 2–3 is returned with bound = max(relaxation bound, CP-SAT bound).
Motivation (DECISIONS 2026-11-04): on 60–100-task graphs, step 2 alone beat 20 s of monolithic CP-SAT on all four
hard instances tried.
"""

from __future__ import annotations

import dataclasses
import time

from estimate.annotate import Problem
from sim import evaluate
from solvers.cp import solve_cp
from solvers.exact.lbbd import LbbdOutcome
from solvers.exact.relaxation import relaxation_bound
from solvers.heuristics.list_sched import heft


def relax_and_schedule(
    problem: Problem,
    time_limit_s: float = 60.0,
    relax_share: float = 0.4,
    fixed_share: float = 0.2,
    workers: int = 8,
    seed: int = 0,
) -> LbbdOutcome:
    t0 = time.perf_counter()
    if set(problem.weights) != {"latency"}:
        raise ValueError("this method minimises the makespan; use CP-SAT for other objectives")
    notes: list[str] = []
    rb = relaxation_bound(problem, time_limit_s=max(1.0, relax_share * time_limit_s), seed=seed)
    if rb.status == "infeasible":
        return LbbdOutcome("infeasible", None, None, None, None, 1, time.perf_counter() - t0, notes=rb.notes)
    bound = rb.bound_ms or 0.0
    cands = []
    hint = None
    if rb.mapping is not None:
        fixed = dataclasses.replace(problem, options={t: [rb.mapping[t]] for t in problem.order}, _norms=None)
        s = solve_cp(fixed, time_limit_s=max(1.0, fixed_share * time_limit_s), workers=workers, seed=seed,
                     contention=True)
        if s.mapping is not None:
            ev = evaluate(problem, s.mapping, priority=s.priority(fixed))
            cands.append((ev, s.mapping, s.priority(fixed), "relaxation mapping"))
            hint = s.mapping
        else:
            notes.append(f"relaxation mapping: schedule {s.status}")
            hint = rb.mapping
    else:
        notes.append("relaxation found no mapping in its share of the budget; full CP-SAT hinted with HEFT")
        hint = heft(problem).mapping
    rest = time_limit_s - (time.perf_counter() - t0)
    c = solve_cp(problem, time_limit_s=max(1.0, rest), workers=workers, seed=seed, contention=True, hint=hint)
    if c.mapping is not None:
        cands.append((evaluate(problem, c.mapping, priority=c.priority(problem)), c.mapping, c.priority(problem),
                      "full CP-SAT"))
    if c.status == "infeasible":
        return LbbdOutcome("infeasible", None, None, None, None, 1, time.perf_counter() - t0,
                           notes=notes + ["CP-SAT proved the instance infeasible"])
    if c.bound is not None:
        bound = max(bound, c.bound * problem.norms["latency"])
    feas = [x for x in cands if x[0].feasible]
    rt = time.perf_counter() - t0
    if not feas:
        return LbbdOutcome("time_limit", None, None, None, bound or None, 1, rt, notes=notes)
    ev, mapping, prio, src = min(feas, key=lambda x: x[0].sim.makespan)
    ms = ev.sim.makespan
    status = "optimal" if c.status == "optimal" or bound >= ms * (1 - 1e-3) else "feasible"
    return LbbdOutcome(status, mapping, prio, ms, min(bound, ms), 1, rt, notes=notes + [f"best from {src}"])
