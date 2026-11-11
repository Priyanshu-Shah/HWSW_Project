"""Optimistic lower bound: every task on its fastest legal option, zero communication, unlimited parallelism.

If even this bound violates a constraint, no mapping on this platform subset can meet it, so the
problem belongs to task generation / the spec (feedback loop fb1).
"""

from __future__ import annotations

from estimate.annotate import CostView, Problem
from schemas.result import LowerBound, Violation
from sim.evaluate import _exceeds, violation


def _bounds(problem: Problem, v: CostView) -> tuple[float, dict[str, float], float, float]:
    """(critical path, earliest finish per node, II bound, energy bound) for one cost view."""
    t_min = {n: min(v.time[(n, o.impl)] for o in problem.options[n]) for n in problem.order if problem.options[n]}
    finish: dict[str, float] = {}
    for n in problem.order:
        finish[n] = max((finish[p] for p in problem.preds[n]), default=0.0) + t_min.get(n, 0.0)
    latency = max(finish.values(), default=0.0)
    ii = 0.0
    for n in problem.order:
        if problem.options[n]:
            ii = max(
                ii,
                min(v.time[(n, o.impl)] if problem.is_exclusive(o.pe) else v.ii[(n, o.impl)] for o in problem.options[n]),
            )
    energy = sum(min(v.energy[(n, o.impl)] for o in problem.options[n]) for n in problem.order if problem.options[n])
    energy += problem.static_power_mw * latency / 1000.0
    return latency, finish, ii, energy


def _capacity_structural(problem: Problem, v: CostView) -> list[str]:
    msgs: list[str] = []
    forced: dict[str, dict[str, float]] = {}
    for n in problem.order:
        opts = problem.options[n]
        fits = []
        for o in opts:
            cap = problem.platform.pe(o.pe).capacity
            use = v.res[(n, o.impl)]
            fits.append(all(not _exceeds(use.get(r, 0.0), c) for r, c in cap.items()))
        if opts and not any(fits):
            msgs.append(f"node {n}: no legal option fits its PE's capacity even at optimistic resource estimates")
        pes = {o.pe for o in opts}
        if len(pes) == 1:  # node can only go to one PE: its smallest footprint is forced onto it
            pe = next(iter(pes))
            for r in problem.platform.pe(pe).capacity:
                need = min(v.res[(n, o.impl)].get(r, 0.0) for o in opts)
                forced.setdefault(pe, {})[r] = forced.get(pe, {}).get(r, 0.0) + need
    for pe, use in forced.items():
        for r, need in use.items():
            cap = problem.platform.pe(pe).capacity[r]
            if _exceeds(need, cap):
                msgs.append(f"capacity[{pe}].{r}: nodes that can only run on {pe} need {need:g} > {cap:g}")
    return msgs


def _violations(problem: Problem, v: CostView) -> tuple[float, float, float, list[Violation]]:
    lat, finish, ii, energy = _bounds(problem, v)
    viol: list[Violation] = []

    def chk(name: str, value: float, limit: float | None) -> None:
        if limit is not None and _exceeds(value, limit):
            viol.append(violation(name, value, limit))

    chk("deadline_ms", lat, problem.deadline_ms)
    for n, d in problem.task_deadlines.items():
        chk(f"deadline_ms[{n}]", finish[n], d)
    if problem.arrival == "periodic":
        chk("period_ms", lat, problem.period_ms)
    elif problem.arrival == "streaming":
        chk("period_ms", ii, problem.period_ms)
    chk("energy_mj_max", energy, problem.energy_mj_max)
    window = problem.period_ms or lat
    chk("power_mw_max", energy / window * 1000.0 if window else 0.0, problem.power_mw_max)
    return lat, ii, energy, viol


def lower_bound(problem: Problem, strong: bool = False, time_limit_s: float = 10.0) -> LowerBound:
    """Critical-path bounds; with `strong=True` also the mapping relaxation (solvers/exact/relaxation.py), which
    adds PE and bus load, transfers and capacity. Its makespan bound replaces the critical path in the deadline (and
    periodic period) checks when it is larger: a violation then proves no mapping can meet the constraint (fb1)."""
    optimistic = problem.bound_view("lo")
    lat, ii, energy, viol = _violations(problem, problem.nominal())
    lat_opt, _, _, viol_opt = _violations(problem, optimistic)
    structural = list(problem.structural) + _capacity_structural(problem, optimistic)
    certain = [v.constraint for v in viol_opt]
    lat_relax = None
    if strong and not structural:
        lat_relax = _strengthen(problem, optimistic, lat, viol, certain, structural, time_limit_s)
    return LowerBound(
        latency_ms=lat,
        latency_ms_optimistic=lat_opt,
        ii_ms=ii,
        energy_mj=energy,
        violations=viol,
        certain=certain,
        structural=structural,
        latency_ms_relaxation=lat_relax,
    )


def _strengthen(problem: Problem, optimistic: CostView, lat: float, viol: list[Violation], certain: list[str],
                structural: list[str], time_limit_s: float) -> float | None:
    """Upgrade the makespan checks with the mapping relaxation (deadlines dropped, so its optimum is the smallest
    makespan any mapping could reach). Mutates viol / certain / structural; returns the nominal bound."""
    from solvers.exact.relaxation import relaxation_bound  # local: diagnostics must not import solvers at load time

    nom = relaxation_bound(problem, time_limit_s=time_limit_s, deadlines=False)
    if nom.status == "infeasible":
        structural.append("no mapping satisfies the capacity constraints (mapping relaxation infeasible)")
        return None
    if nom.bound_ms is None or nom.bound_ms <= lat:
        return nom.bound_ms
    lo = relaxation_bound(problem, time_limit_s=time_limit_s, deadlines=False, costs=optimistic).bound_ms
    limits = [("deadline_ms", problem.deadline_ms)]
    if problem.arrival == "periodic":
        limits.append(("period_ms", problem.period_ms))
    for name, limit in limits:
        if limit is None or not _exceeds(nom.bound_ms, limit):
            continue
        viol[:] = [v for v in viol if v.constraint != name] + [violation(name, nom.bound_ms, limit)]
        if lo is not None and _exceeds(lo, limit) and name not in certain:
            certain.append(name)
    return nom.bound_ms
