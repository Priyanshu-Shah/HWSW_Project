"""Exact dynamic programme for chain task graphs with several implementations and one resource budget.

After Wu, Sun & Srikanthan (Computers & OR 2012, multiple-choice HW/SW partitioning) and Shi et al. (Computers &
Electrical Engineering 2016): E(i, j, a) = min_t E(i-1, t, a - a_ij) + c(t -> j) + time_ij, O(n · K² · A) for K
options per task and integer budget A. On a chain execution is sequential, so the makespan is the sum of task times
plus cross-PE transfer times regardless of contention: the DP is exact for our full model (latency objective, one
capacity dimension). Used as an oracle to check the CP-SAT / MILP models on long chains.
"""

from __future__ import annotations

import math

import numpy as np

from estimate.annotate import Option, Problem


def chain_dp(problem: Problem) -> tuple[dict[str, Option], float]:
    """Optimal mapping and makespan. Requires: a chain (each task ≤ 1 predecessor and ≤ 1 successor, connected),
    latency-only objective, and at most one finite (PE, resource) capacity, with integer resource values."""
    order = problem.order
    if any(len(problem.preds[t]) > 1 or len(problem.succs[t]) > 1 for t in order) or \
            len(problem.edges) != len(order) - 1:
        raise ValueError("chain_dp needs a single chain")
    if set(problem.weights) != {"latency"}:
        raise ValueError("chain_dp optimises latency only")
    caps = [(pe, r, c) for pe in problem.pes for r, c in problem.platform.pe(pe).capacity.items()]
    if len(caps) > 1:
        raise ValueError("chain_dp supports one capacity dimension")
    v = problem.nominal()
    cap_pe, cap_r, cap = caps[0] if caps else (None, None, 0.0)
    A = int(math.floor(cap + 1e-9)) if caps else 0
    opts = [problem.options[t] for t in order]

    def use(t: str, o: Option) -> int:
        if o.pe != cap_pe:
            return 0
        x = v.res[(t, o.impl)].get(cap_r, 0.0)
        if abs(x - round(x)) > 1e-9:
            raise ValueError("chain_dp needs integer resource values")
        return int(round(x))

    inf = math.inf
    # E[j] is an array over used area 0..A
    E = []
    for j, o in enumerate(opts[0]):
        row = np.full(A + 1, inf)
        u = use(order[0], o)
        if u <= A:
            row[u] = v.time[(order[0], o.impl)]
        E.append(row)
    back: list[list[np.ndarray]] = []
    for i in range(1, len(order)):
        prev, cur = order[i - 1], order[i]
        nb = v.bytes[(prev, cur)]
        newE, newB = [], []
        for o in opts[i]:
            u, tm = use(cur, o), v.time[(cur, o.impl)]
            best = np.full(A + 1, inf)
            arg = np.full(A + 1, -1, dtype=np.int32)
            if u <= A:
                for t, po in enumerate(opts[i - 1]):
                    c = problem.comm(po.pe, o.pe, nb)
                    if c is None:
                        continue
                    cand = np.full(A + 1, inf)
                    cand[u:] = E[t][: A + 1 - u] + c.time_ms + tm
                    better = cand < best
                    best[better], arg[better] = cand[better], t
            newE.append(best)
            newB.append(arg)
        E, _ = newE, back.append(newB)
    finals = [(row.min(), j, int(row.argmin())) for j, row in enumerate(E)]
    best_val, j, a = min(finals)
    if not math.isfinite(best_val):
        raise ValueError("no feasible mapping within the capacity")
    mapping: dict[str, Option] = {}
    for i in range(len(order) - 1, -1, -1):
        mapping[order[i]] = opts[i][j]
        if i == 0:
            break
        t = int(back[i - 1][j][a])
        a -= use(order[i], opts[i][j])
        j = t
    return mapping, float(best_val)
