"""Baselines for the GCPS model, all scored by LSSP (as in Zheng et al. 2021, "combined with LSSP").

- greedy_hgp: rank by HW gain per unit area (HGP), stop-at-first-overflow cut line (GCPS without the GCN).
- kl_lssp: Kernighan–Lin with single-node flips (Mann, Orbán & Farkas 2007, Alg. 5: gain = change of the LSSP
  schedule length). Permissive area penalty ((r−1)/(q−1))^4 for area ratio r in (1, q], forbidden above q; q = 1.1
  matches the paper's unexplained "KL: N = 1.1" (inferred). Only feasible partitions can become the best. Random
  valid start (Table 8: "Random"). Optional wall-clock budget.
- sa_lssp: simulated annealing, Table 3 parameters T = 1000, L = 300 (moves per temperature), M = 500 (temperature
  levels), r = 0.999 (cooling); flips that break the area budget are rejected (Wiangtong 2002). Random valid start.
"""

from __future__ import annotations

import math
import time

import numpy as np

from solvers.gcps.instance import BinInstance
from solvers.gcps.lssp import greedy_area, lssp


def greedy_hgp(inst: BinInstance) -> np.ndarray:
    return greedy_area((inst.ts - inst.th) / inst.area, inst.area, inst.C)


def _random_valid(inst: BinInstance, rng: np.random.Generator) -> np.ndarray:
    return greedy_area(rng.random(inst.n), inst.area, inst.C)


def kl_lssp(inst: BinInstance, seed: int = 0, q: float = 1.1, max_passes: int = 20,
            budget_s: float | None = None, weight: float = 10.0, candidates: int | None = None) -> np.ndarray:
    """`candidates`: evaluate only the top-k unlocked flips per move by surrogate gain (HW gain for SW->HW, its
    negative for HW->SW); None = all (exact Mann Alg. 5 move selection)."""
    rng = np.random.default_rng(seed)
    t_start = time.perf_counter()
    cur = _random_valid(inst, rng)

    def cost(y: np.ndarray) -> tuple[float, bool]:
        r = float(inst.area @ y) / inst.C
        if r > q:
            return float("inf"), False
        pen = ((r - 1) / (q - 1)) ** 4 if r > 1 else 0.0
        return lssp(inst, y) / inst.t_max + weight * pen, r <= 1 + 1e-12

    best_y, (best_c, _) = cur.copy(), cost(cur)
    for _ in range(max_passes):
        state = best_y.copy()
        locked = np.zeros(inst.n, dtype=bool)
        pass_best_c, pass_best = best_c, None
        for _ in range(inst.n):
            if budget_s is not None and time.perf_counter() - t_start > budget_s:
                break
            cand = None
            free = np.flatnonzero(~locked)
            if candidates is not None and len(free) > candidates:
                gain = np.where(state[free] == 0, inst.ts[free] - inst.th[free], inst.th[free] - inst.ts[free])
                free = free[np.argsort(-gain, kind="stable")[:candidates]]
            for i in free:
                state[i] ^= 1
                c, ok = cost(state)
                state[i] ^= 1
                if c < float("inf") and (cand is None or c < cand[0]):
                    cand = (c, i, ok)
            if cand is None:
                break
            c, i, ok = cand
            state[i] ^= 1
            locked[i] = True
            if ok and c < pass_best_c - 1e-12:
                pass_best_c, pass_best = c, state.copy()
        if pass_best is None:
            break
        best_y, best_c = pass_best, pass_best_c
        if budget_s is not None and time.perf_counter() - t_start > budget_s:
            break
    return best_y


def sa_lssp(inst: BinInstance, seed: int = 0, t0: float = 1000.0, steps_per_temp: int = 300, temps: int = 500,
            r: float = 0.999, budget_s: float | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t_start = time.perf_counter()
    cur = _random_valid(inst, rng)
    cur_t = lssp(inst, cur)
    best, best_t = cur.copy(), cur_t
    used = float(inst.area @ cur)
    temp = t0
    for _ in range(temps):
        for _ in range(steps_per_temp):
            i = int(rng.integers(inst.n))
            new_used = used + (inst.area[i] if not cur[i] else -inst.area[i])
            if new_used > inst.C + 1e-9:
                continue
            cur[i] ^= 1
            t = lssp(inst, cur)
            if t <= cur_t or rng.random() < math.exp(-(t - cur_t) / temp):
                cur_t, used = t, new_used
                if t < best_t:
                    best, best_t = cur.copy(), t
            else:
                cur[i] ^= 1
        temp *= r
        if budget_s is not None and time.perf_counter() - t_start > budget_s:
            break
    return best
