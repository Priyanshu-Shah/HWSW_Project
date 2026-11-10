"""Constraint checking and objective for a simulated mapping."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from estimate.annotate import CostView, Problem
from schemas.result import Violation
from sim.scheduler import MappingT, SimResult, simulate

_TOL = 1e-9


def violation(name: str, value: float, limit: float) -> Violation:
    excess = value - limit
    return Violation(
        constraint=name, value=value, limit=limit, excess=excess, relative=excess / limit if limit else float("inf")
    )


def _exceeds(value: float, limit: float) -> bool:
    return value > limit * (1 + _TOL) + _TOL


def avg_power_mw(problem: Problem, sim: SimResult) -> float:
    window = problem.period_ms if problem.period_ms else sim.makespan
    return sim.energy_mj / window * 1000.0 if window > 0 else 0.0


def check(problem: Problem, sim: SimResult) -> list[Violation]:
    out: list[Violation] = []

    def chk(name: str, value: float, limit: float | None) -> None:
        if limit is not None and _exceeds(value, limit):
            out.append(violation(name, value, limit))

    chk("deadline_ms", sim.makespan, problem.deadline_ms)
    for n, d in problem.task_deadlines.items():
        chk(f"deadline_ms[{n}]", sim.tasks[n].finish, d)
    if problem.arrival == "periodic":
        chk("period_ms", sim.makespan, problem.period_ms)
    elif problem.arrival == "streaming":
        chk("period_ms", sim.ii_ms, problem.period_ms)
    for pe in problem.pes:
        cap = problem.platform.pe(pe).capacity
        use = sim.usage.get(pe, {})
        for r, limit in cap.items():
            chk(f"capacity[{pe}].{r}", use.get(r, 0.0), limit)
    chk("energy_mj_max", sim.energy_mj, problem.energy_mj_max)
    power = avg_power_mw(problem, sim)
    chk("power_mw_max", power, problem.power_mw_max)
    chk("usd_max", sim.usd, problem.usd_max)
    budgets = problem.platform.budgets
    chk("budget.usd", sim.usd, budgets.get("usd"))
    chk("budget.power_mw", power, budgets.get("power_mw"))
    return out


def objective(problem: Problem, sim: SimResult) -> float:
    """Weighted sum of latency, energy and $ cost, each divided by its lower-bound reference."""
    metric = {"latency": sim.makespan, "energy": sim.energy_mj, "usd": sim.usd}
    norms = problem.norms
    return sum(w * metric[k] / norms[k] for k, w in problem.weights.items())


@dataclass
class Evaluation:
    sim: SimResult
    violations: list[Violation]
    objective: float

    @property
    def feasible(self) -> bool:
        return not self.violations


def evaluate(
    problem: Problem,
    mapping: MappingT,
    costs: CostView | None = None,
    priority: Sequence[str] | None = None,
    contention: bool = True,
    insertion: bool = False,
) -> Evaluation:
    sim = simulate(problem, mapping, costs, priority, contention, insertion)
    return Evaluation(sim, check(problem, sim), objective(problem, sim))
