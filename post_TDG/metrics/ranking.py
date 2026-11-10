"""Relative metrics: rank candidate mappings and measure how stable the order is under cost uncertainty.

All candidates are evaluated on the same cost samples (common random numbers), because the uncertainty is
in the task and edge costs, not in the mapping. A candidate that violates a constraint in a sample ranks
after every candidate that meets all constraints in that sample. Candidates with identical mappings are
merged into one entry labelled "a=b".
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from estimate.annotate import Problem
from metrics.fidelity import kendall_tau
from schemas.result import Ranking
from sim import montecarlo
from sim.evaluate import evaluate
from sim.scheduler import MappingT

INFEASIBLE_PENALTY = 1e6


def _merge_identical(candidates: Mapping[str, MappingT]) -> dict[str, MappingT]:
    groups: dict[tuple, list[str]] = {}
    for label, m in candidates.items():
        groups.setdefault(tuple(sorted((n, o.pe, o.impl) for n, o in m.items())), []).append(label)
    return {"=".join(labels): candidates[labels[0]] for labels in groups.values()}


def rank(
    problem: Problem,
    candidates: Mapping[str, MappingT],
    n_samples: int = 1000,
    seed: int = 0,
) -> Ranking:
    """Order candidates by (infeasible at nominal, nominal objective), lower is better; ties by label."""
    cands = _merge_identical(candidates)
    labels = list(cands)
    nominal = {}
    for k in labels:
        ev = evaluate(problem, cands[k])
        nominal[k] = ev.objective + (0.0 if ev.feasible else INFEASIBLE_PENALTY)
    order = sorted(labels, key=lambda k: (nominal[k], k))
    factors = montecarlo.draw(problem, n_samples, seed)
    rows = []
    for k in order:
        mc = montecarlo.run(problem, cands[k], factors=factors)
        rows.append(mc.objective + INFEASIBLE_PENALTY * (~mc.feasible))
    score = np.array(rows).reshape(len(order), -1)  # (candidates, samples)

    c = len(order)
    pairwise = [[float(np.mean(score[i] < score[j])) if i != j else 1.0 for j in range(c)] for i in range(c)]
    preserved = 0
    taus = []
    for s in range(score.shape[1]):
        col = score[:, s]
        preserved += int(all(col[i] <= col[i + 1] for i in range(c - 1)))
        if c > 1:
            taus.append(kendall_tau(list(range(c)), col.tolist()))
    return Ranking(
        candidates=order,
        objective_nominal=[nominal[k] for k in order],
        pairwise_confidence=pairwise,
        order_preserved_fraction=preserved / n_samples if n_samples else 1.0,
        mean_kendall_tau=float(np.nanmean(taus)) if taus else 1.0,
        n_samples=n_samples,
        seed=seed,
    )
