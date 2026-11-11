"""Classical baselines. See docs/survey/partitioning_survey.md §7.2 for why each is here."""

from __future__ import annotations

import time
from collections.abc import Callable

from estimate.annotate import Problem
from schemas.result import Result, SolverInfo
from solvers.common import build_result
from solvers.heuristics.baselines import (
    all_hw,
    all_sw,
    exhaustive,
    greedy_eft,
    simulated_annealing,
)
from solvers.heuristics.evolutionary import ParetoPoint, genetic_algorithm, nsga2
from solvers.heuristics.list_sched import Plan, heft, optimistic_cost_table, peft, upward_ranks
from solvers.heuristics.local_search import kernighan_lin, tabu_search
from solvers.heuristics.magellan import magellan

__all__ = [
    "HEURISTICS",
    "ParetoPoint",
    "Plan",
    "all_hw",
    "all_sw",
    "exhaustive",
    "genetic_algorithm",
    "greedy_eft",
    "heft",
    "kernighan_lin",
    "magellan",
    "nsga2",
    "optimistic_cost_table",
    "peft",
    "plan",
    "run",
    "simulated_annealing",
    "tabu_search",
    "upward_ranks",
]


def _ga(problem: Problem, seed: int) -> Plan:
    mapping, priority = genetic_algorithm(problem, seed=seed)
    return Plan(mapping, priority, insertion=False)


HEURISTICS: dict[str, Callable[[Problem, int], Plan]] = {
    "all_sw": lambda p, s: Plan(all_sw(p)),
    "all_hw": lambda p, s: Plan(all_hw(p)),
    "greedy_eft": lambda p, s: Plan(greedy_eft(p)),
    "heft": lambda p, s: heft(p),
    "peft": lambda p, s: peft(p),
    "kl": lambda p, s: Plan(kernighan_lin(p, seed=s)),
    "kl2": lambda p, s: Plan(kernighan_lin(p, max_moves_per_node=5, restarts=10, seed=s)),
    "tabu": lambda p, s: Plan(tabu_search(p, seed=s)),
    "sa": lambda p, s: Plan(simulated_annealing(p, seed=s)),
    "ga": _ga,
    "magellan": lambda p, s: magellan(p),
}


def plan(problem: Problem, name: str, seed: int = 0) -> Plan:
    return HEURISTICS[name](problem, seed)


def run(problem: Problem, name: str, seed: int = 0, n_samples: int = 1000) -> Result:
    t0 = time.perf_counter()
    p = plan(problem, name, seed)
    info = SolverInfo(name=name, status="heuristic", runtime_s=time.perf_counter() - t0, seed=seed,
                      params={"insertion": p.insertion})
    return build_result(problem, p.mapping, info, priority=p.priority, n_samples=n_samples, seed=seed,
                        insertion=p.insertion)
