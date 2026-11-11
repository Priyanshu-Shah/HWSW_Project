"""Independent schedule checkers (used to test solver outputs and the simulator itself)."""

from __future__ import annotations

from collections.abc import Mapping

from estimate.annotate import CostView, Problem
from sim.scheduler import MappingT, SimResult

_EPS = 1e-6


def validate_starts(
    problem: Problem, mapping: MappingT, starts: Mapping[str, float], costs: CostView | None = None
) -> list[str]:
    """Check start times under the no-contention model: precedence + comm delay, exclusive PEs don't overlap."""
    costs = costs or problem.nominal()
    errors: list[str] = []
    finish = {n: starts[n] + costs.time[(n, mapping[n].impl)] for n in problem.order}
    for n in problem.order:
        if starts[n] < -_EPS:
            errors.append(f"{n} starts before 0")
    for s, d in problem.edges:
        c = problem.comm(mapping[s].pe, mapping[d].pe, costs.bytes[(s, d)])
        if c is None:
            errors.append(f"no route {s}->{d}")
        elif starts[d] < finish[s] + c.time_ms - _EPS:
            errors.append(f"{d} starts {starts[d]:.6f} before {s} data arrives {finish[s] + c.time_ms:.6f}")
    errors += _overlaps(problem, {n: (mapping[n].pe, starts[n], finish[n]) for n in problem.order})
    return errors


def validate_sim(problem: Problem, sim: SimResult, costs: CostView | None = None, contention: bool = True) -> list[str]:
    """Check a simulated schedule, including interface serialisation on shared links."""
    costs = costs or problem.nominal()
    errors: list[str] = []
    for n, t in sim.tasks.items():
        if abs((t.finish - t.start) - costs.time[(n, t.impl)]) > _EPS:
            errors.append(f"{n}: duration does not match cost")
    comm = {(c.src, c.dst): c for c in sim.comms}
    for s, d in problem.edges:
        c = comm.get((s, d))
        if c is None:
            errors.append(f"missing transfer {s}->{d}")
            continue
        if c.start < sim.tasks[s].finish - _EPS:
            errors.append(f"transfer {s}->{d} starts before {s} finishes")
        if sim.tasks[d].start < c.finish - _EPS:
            errors.append(f"{d} starts before transfer from {s} completes")
    errors += _overlaps(problem, {n: (t.pe, t.start, t.finish) for n, t in sim.tasks.items()})
    if contention:
        by_if: dict[str, list[tuple[float, float]]] = {}
        for c in sim.comms:
            if c.interface is not None and problem.interface(c.interface).shared and c.finish > c.start:
                by_if.setdefault(c.interface, []).append((c.start, c.finish))
        for itf, spans in by_if.items():
            spans.sort()
            for (a0, a1), (b0, _) in zip(spans, spans[1:], strict=False):
                if b0 < a1 - _EPS:
                    errors.append(f"overlapping transfers on shared interface {itf} at {b0:.6f}")
    return errors


def _overlaps(problem: Problem, slots: Mapping[str, tuple[str, float, float]]) -> list[str]:
    errors = []
    by_pe: dict[str, list[tuple[float, float, str]]] = {}
    for n, (pe, s, f) in slots.items():
        if problem.is_exclusive(pe) and f > s:
            by_pe.setdefault(pe, []).append((s, f, n))
    for pe, spans in by_pe.items():
        spans.sort()
        for (s0, f0, a), (s1, _, b) in zip(spans, spans[1:], strict=False):
            if s1 < f0 - _EPS:
                errors.append(f"{a} and {b} overlap on exclusive PE {pe}")
    return errors
