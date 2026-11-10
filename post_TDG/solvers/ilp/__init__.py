"""Exact ILP baseline (also the label generator for imitation learning later)."""

from __future__ import annotations

from estimate.annotate import Problem
from schemas.result import Result, SolverInfo
from solvers.common import build_result
from solvers.ilp.model import IlpOutcome, solve_ilp

__all__ = ["IlpOutcome", "solve", "solve_ilp"]


def solve(
    problem: Problem,
    solver: str = "highs",
    time_limit_s: float = 60.0,
    seed: int = 0,
    n_samples: int = 1000,
) -> Result:
    out = solve_ilp(problem, solver=solver, time_limit_s=time_limit_s, seed=seed)
    priority = sorted(out.starts, key=lambda n: (out.starts[n], problem.order.index(n))) if out.starts else None
    info = SolverInfo(
        name=f"ilp-{solver}",
        status=out.status,  # type: ignore[arg-type]
        runtime_s=out.runtime_s,
        seed=seed,
        objective=out.objective,
        params={"time_limit_s": time_limit_s, "ilp_makespan_ms": out.makespan},
        notes="; ".join(out.notes) or "optimal under the no-contention model; metrics re-simulated with contention",
    )
    return build_result(problem, out.mapping, info, priority=priority, relaxation=out.relaxation,
                        n_samples=n_samples, seed=seed)
