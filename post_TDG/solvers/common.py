"""Turn a solver's mapping into a contract `Result` (simulate, Monte Carlo metrics, diagnostics)."""

from __future__ import annotations

from collections.abc import Sequence

from diagnostics import diagnose, lower_bound
from estimate.annotate import Problem
from metrics.absolute import DEFAULT_SAMPLES, metrics_for
from schemas.result import (
    Assignment,
    Diagnostics,
    Result,
    ScheduledComm,
    ScheduledTask,
    SolverInfo,
    Violation,
)
from sim.evaluate import _exceeds, evaluate, violation
from sim.scheduler import MappingT


def build_result(
    problem: Problem,
    mapping: MappingT | None,
    solver: SolverInfo,
    priority: Sequence[str] | None = None,
    relaxation: list[Violation] | None = None,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
    insertion: bool = False,
    strong_bound: bool = False,
) -> Result:
    lb = lower_bound(problem, strong=strong_bound)
    base = dict(graph_id=problem.graph.id, platform_id=problem.platform.id, workload=problem.workload, solver=solver)
    if mapping is None:
        diag: Diagnostics = diagnose(lb, None, relaxation)
        return Result(**base, mapping={}, schedule=[], comms=[], feasible=False, metrics=None, diagnostics=diag)

    ev = evaluate(problem, mapping, priority=priority, insertion=insertion)
    metrics = metrics_for(problem, mapping, n_samples=n_samples, seed=seed, priority=priority, insertion=insertion)
    sim = ev.sim
    violations = list(ev.violations)
    tm = metrics.throughput
    if (tm is not None and problem.period_ms is not None and not any(v.constraint == "period_ms" for v in violations)
            and _exceeds(tm.period_ms.nominal, problem.period_ms)):
        # the busy-time bound passed but the exact (max-cycle-ratio) period does not
        violations.append(violation("period_ms.exact", tm.period_ms.nominal, problem.period_ms))
    return Result(
        **base,
        mapping={n: Assignment(pe=o.pe, impl=o.impl) for n, o in mapping.items()},
        schedule=[
            ScheduledTask(node=t.node, pe=t.pe, impl=t.impl, start_ms=t.start, finish_ms=t.finish)
            for t in sorted(sim.tasks.values(), key=lambda t: (t.start, t.node))
        ],
        comms=[
            ScheduledComm(src=c.src, dst=c.dst, interface=c.interface, bytes=c.nbytes, start_ms=c.start,
                          finish_ms=c.finish)
            for c in sim.comms
        ],
        feasible=not violations,
        metrics=metrics,
        diagnostics=diagnose(lb, violations, relaxation, metrics.feasible_fraction),
    )
