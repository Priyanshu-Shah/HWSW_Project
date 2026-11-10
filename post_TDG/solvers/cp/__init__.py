"""Exact CP-SAT baseline and label generator (preferred over the MILP beyond ~10 tasks; see the survey §7.1)."""

from __future__ import annotations

from estimate.annotate import Problem
from schemas.result import Result, SolverInfo
from solvers.common import build_result
from solvers.cp.model import CpOutcome, solve_cp
from solvers.heuristics.list_sched import heft
from solvers.ilp.model import minimal_relaxation

__all__ = ["CpOutcome", "solve", "solve_cp"]


def solve(
    problem: Problem,
    time_limit_s: float = 60.0,
    workers: int = 8,
    seed: int = 0,
    n_samples: int = 1000,
    warm_start: bool = True,
    contention: bool = True,
) -> Result:
    """CP-SAT result. `contention=True` (default) models shared-bus serialisation, matching the evaluator; the
    schedule is replayed exactly by passing task + transfer priorities to the simulator (unified mode)."""
    hint = hint_starts = None
    if warm_start and all(problem.options[n] for n in problem.order):
        try:
            h = heft(problem, insertion=False)
            hint = h.mapping
        except ValueError:
            pass
    out = solve_cp(problem, time_limit_s=time_limit_s, workers=workers, seed=seed, hint=hint,
                   hint_starts=hint_starts, contention=contention)
    relaxation, notes = None, list(out.notes)
    if out.status == "infeasible" and all(problem.options[n] for n in problem.order):
        relaxation, more = minimal_relaxation(problem, time_limit_s=time_limit_s, seed=seed)
        notes += more
        if contention and relaxation == []:
            relaxation = None
            notes.append("feasible without bus contention: the infeasibility comes from shared-bus serialisation "
                         "(relaxation is computed by the no-contention ILP)")
    priority = out.priority(problem)  # includes transfers when contention is modelled
    info = SolverInfo(
        name="cp-sat",
        status=out.status,  # type: ignore[arg-type]
        runtime_s=out.runtime_s,
        seed=seed,
        objective=out.objective,
        params={"time_limit_s": time_limit_s, "workers": workers, "cp_makespan_ms": out.makespan,
                "best_bound": out.bound, "gap": out.gap, "contention": contention},
        notes="; ".join(notes) or ("optimal with shared-bus contention" if contention else
                                   "optimal under the no-contention model; metrics re-simulated with contention"),
    )
    return build_result(problem, out.mapping, info, priority=priority, relaxation=relaxation,
                        n_samples=n_samples, seed=seed)
