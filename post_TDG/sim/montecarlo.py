"""Monte Carlo evaluation of a mapping under bounded cost uncertainty."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from estimate.annotate import Problem, UncKey
from sim.evaluate import avg_power_mw, evaluate
from sim.scheduler import MappingT


@dataclass
class MCSamples:
    latency_ms: np.ndarray
    energy_mj: np.ndarray
    ii_ms: np.ndarray
    avg_power_mw: np.ndarray
    objective: np.ndarray
    feasible: np.ndarray
    seed: int


def draw(problem: Problem, n: int, seed: int) -> list[dict[UncKey, float]]:
    """Factor samples to share across candidates (common random numbers)."""
    return problem.sample_factors(np.random.default_rng(seed), n)


def run(
    problem: Problem,
    mapping: MappingT,
    n: int = 1000,
    seed: int = 0,
    priority: Sequence[str] | None = None,
    contention: bool = True,
    factors: list[dict[UncKey, float]] | None = None,
    insertion: bool = False,
) -> MCSamples:
    factors = factors if factors is not None else draw(problem, n, seed)
    rows = []
    for f in factors:
        ev = evaluate(problem, mapping, problem.view(f), priority, contention, insertion)
        s = ev.sim
        rows.append((s.makespan, s.energy_mj, s.ii_ms, avg_power_mw(problem, s), ev.objective, ev.feasible))
    a = np.array(rows, dtype=float).reshape(-1, 6)
    return MCSamples(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5].astype(bool), seed)
