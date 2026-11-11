"""Single-solution metaheuristics: simulated annealing, Kernighan-Lin (KL1 / KL2) and tabu search.

All of them move in the space of *option choices* (one global option id per task) and score a candidate with the
compiled simulator in default mode (upward-rank priority, no insertion). They start from the HEFT mapping, and the HEFT
schedule itself is offered to the tracker, so no metaheuristic ever reports a result worse than HEFT.

Constraint handling: capacity is a hard constraint for the *reported* solution (only capacity-feasible mappings are
recorded as best). During the search an infeasible candidate is scored with a penalty:

* SA / tabu: ``f = makespan * (1 + 20 * excess)``;
* KL: the "permissive" cost of Mann, Orban & Farkas (2007), a fourth-order penalty ``makespan * (1 + 16 * excess^4)``
  that is forbidden (infinite) beyond ``excess > 2``.

``excess`` = sum over (PE, resource) of max(0, usage / capacity - 1).
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..fast import PEN, Fast
from ..instance import Instance
from .base import Result, Tracker, random_feasible


def _warm_start(inst: Instance, fast: Fast, tr: Tracker):
    plan, hmk = fast.heft()
    tr.offer_plan(plan, hmk)
    opt = plan.opt.copy()
    mk, ex = fast.evaluate(opt)
    tr.offer(mk, ex, opt)
    return opt, mk, ex


def gen_moves(inst: Instance, opt: np.ndarray, cand: np.ndarray, cap: int, rng: np.random.Generator):
    """All (task, alternative option) pairs of ``cand`` tasks, or a uniform random sample of ``cap`` of them."""
    if len(cand) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    sizes = np.array([len(inst.task_opts[t]) - 1 for t in cand])
    if sizes.sum() <= cap:
        mt, mo = [], []
        for t in cand:
            for k in inst.task_opts[t]:
                if k != opt[t]:
                    mt.append(t); mo.append(k)
        return np.array(mt, dtype=np.int64), np.array(mo, dtype=np.int64)
    ts = rng.choice(cand, size=cap, p=sizes / sizes.sum())
    mo = np.empty(cap, dtype=np.int64)
    for j, t in enumerate(ts):
        alts = [k for k in inst.task_opts[t] if k != opt[t]]
        mo[j] = alts[int(rng.integers(len(alts)))]
    return ts.astype(np.int64), mo


# ------------------------------------------------------------------------------------------------ SA
def solve_sa(inst: Instance, fast: Fast, seed: int, budget_s: float, chunk_s: float = 1.5e-3) -> Result:
    """Simulated annealing (Kirkpatrick et al. 1983; move set / cooling after Eles et al. 1997).

    Single-task option moves, Metropolis acceptance, geometric cooling driven by the *wall clock*: the temperature at
    time t is T_hi * (T_lo/T_hi)^(t/T_budget), so the schedule adapts to the budget. T_hi is the mean |delta f| of
    random moves, T_lo = 1e-3 T_hi. The chain runs inside numba in ~1.5 ms chunks."""
    tr = Tracker(inst, fast, budget_s)
    rng = np.random.default_rng(seed)
    fast.seed(seed)
    opt, mk, ex = _warm_start(inst, fast, tr)
    if len(fast.movable) == 0:
        return tr.result("sa")
    cur_f = mk * (1 + PEN * ex)
    # calibrate T_hi on a handful of random moves
    mt, mo = gen_moves(inst, opt, fast.movable, 40, rng)
    mks, exs = fast.eval_moves(opt, mt, mo)
    d = np.abs(mks * (1 + PEN * exs) - cur_f)
    d = d[np.isfinite(d) & (d > 0)]
    t_hi = float(d.mean()) if len(d) else 0.05 * cur_f
    t_lo = t_hi * 1e-3
    best_opt = tr.best_plan.opt.copy()
    best_f = tr.best_mk
    t_start = tr.elapsed()
    t_total = max(budget_s - t_start, 1e-6)
    iters = 8
    while not tr.expired():
        now = tr.elapsed()
        frac_a = min((now - t_start) / t_total, 1.0)
        frac_b = min(frac_a + chunk_s / t_total, 1.0)
        ta = t_hi * (t_lo / t_hi) ** frac_a
        tb = t_hi * (t_lo / t_hi) ** frac_b
        t0 = time.perf_counter()
        cur_f, nb, _acc = fast.sa_chunk(opt, cur_f, best_opt, best_f, ta, tb, iters)
        dt = time.perf_counter() - t0
        if nb < best_f:
            best_f = nb
            tr.offer(nb, 0.0, best_opt)
        if dt > 0:
            iters = int(min(max(iters * chunk_s / dt, 4), 4 * iters))
    return tr.result("sa")


# ------------------------------------------------------------------------------------------------ KL
def _kl_cost(mk: np.ndarray, ex: np.ndarray) -> np.ndarray:
    f = mk * (1.0 + 16.0 * ex ** 4)
    return np.where((ex > 2.0) | ~np.isfinite(mk), np.inf, f)


def solve_kl(inst: Instance, fast: Fast, seed: int, budget_s: float, variant: int = 1, cap: int = 256,
             restarts: int = 10) -> Result:
    """Kernighan-Lin for HW/SW partitioning (Kernighan & Lin 1970; Mann, Orban & Farkas 2007; Vahid & Le 1997).

    A *move* changes one task's option; its gain is the change of the re-simulated cost. A pass repeatedly applies the
    best unlocked move (even if it worsens the cost) and locks the moved task, then returns to the best *feasible* state
    visited. Tie-breaking is LIFO within 5% of the best gain. KL1 locks a task after one move; KL2 uses dynamic locking
    (at most 5 moves per task per pass) and 10 random restarts after convergence. When the neighbourhood is larger than
    ``cap`` moves, a random sample of ``cap`` moves is evaluated per step (stated deviation for large graphs)."""
    tr = Tracker(inst, fast, budget_s)
    rng = np.random.default_rng(seed)
    opt, mk, ex = _warm_start(inst, fast, tr)
    movable = fast.movable
    if len(movable) == 0:
        return tr.result(f"kl{variant}")
    lim = 1 if variant == 1 else 5
    cur_f = float(_kl_cost(np.array([mk]), np.array([ex]))[0])
    n_restart = 0
    passes = 0
    while not tr.expired():
        cnt = np.zeros(inst.n, dtype=np.int64)
        pass_best_f = cur_f if ex == 0 else math.inf
        pass_best_opt = opt.copy()
        start_f = pass_best_f
        max_steps = lim * len(movable)
        steps = 0
        while steps < max_steps and not tr.expired():
            cand = movable[cnt[movable] < lim]
            if len(cand) == 0:
                break
            mt, mo = gen_moves(inst, opt, cand, cap, rng)
            if len(mt) == 0:
                break
            mks, exs = fast.eval_moves(opt, mt, mo)
            f = _kl_cost(mks, exs)
            gain = cur_f - f
            if not np.isfinite(gain).any():
                break
            g = np.where(np.isfinite(gain), gain, -np.inf)
            gmax = g.max()
            thr = gmax - 0.05 * abs(gmax)
            j = int(np.where(g >= thr)[0][-1])               # LIFO among near-best moves
            t, k = int(mt[j]), int(mo[j])
            opt[t] = k
            cnt[t] += 1
            cur_f, mk, ex = float(f[j]), float(mks[j]), float(exs[j])
            tr.offer(mk, ex, opt)
            if ex == 0.0 and cur_f < pass_best_f:
                pass_best_f, pass_best_opt = cur_f, opt.copy()
            steps += 1
        passes += 1
        if pass_best_f < start_f - 1e-12:
            opt = pass_best_opt.copy()
            cur_f = pass_best_f
            ex = 0.0
            continue
        # converged
        if variant == 1 or n_restart >= restarts:
            break
        n_restart += 1
        opt = random_feasible(inst, rng)
        mk, ex = fast.evaluate(opt)
        tr.offer(mk, ex, opt)
        cur_f = float(_kl_cost(np.array([mk]), np.array([ex]))[0])
    return tr.result(f"kl{variant}", passes=passes, restarts=n_restart)


# ------------------------------------------------------------------------------------------------ tabu
def solve_tabu(inst: Instance, fast: Fast, seed: int, budget_s: float, cap: int = 256) -> Result:
    """Tabu search with the selection rule of Wiangtong, Cheung & Luk (2002): take the best improving non-tabu move
    unpenalised; otherwise the best non-tabu move with a frequency penalty (0.005 * current cost per earlier move of that
    task); if every move is tabu, the oldest one. Aspiration: a tabu move that gives a new global best is allowed.
    Tenure = sqrt(#movable tasks) + 1, clamped to [7, 20]. Neighbourhood sampled to ``cap`` moves for large graphs."""
    tr = Tracker(inst, fast, budget_s)
    rng = np.random.default_rng(seed)
    opt, mk, ex = _warm_start(inst, fast, tr)
    movable = fast.movable
    if len(movable) == 0:
        return tr.result("tabu")
    tenure = int(min(max(math.isqrt(len(movable)) + 1, 7), 20))
    tabu_until = np.zeros(inst.n, dtype=np.int64)
    freq = np.zeros(inst.n)
    cur_f = mk * (1 + PEN * ex)
    it = 0
    while not tr.expired():
        it += 1
        mt, mo = gen_moves(inst, opt, movable, cap, rng)
        mks, exs = fast.eval_moves(opt, mt, mo)
        f = np.where(np.isfinite(mks), mks * (1 + PEN * exs), np.inf)
        if not np.isfinite(f).any():
            break
        is_tabu = tabu_until[mt] > it
        asp = is_tabu & (exs == 0) & (mks < tr.best_mk * (1 - 1e-12))
        if asp.any():
            j = int(np.argmin(np.where(asp, mks, np.inf)))
        else:
            imp = (~is_tabu) & (f < cur_f)
            if imp.any():
                j = int(np.argmin(np.where(imp, f, np.inf)))
            elif (~is_tabu).any():
                pf = f + 0.005 * cur_f * freq[mt]
                j = int(np.argmin(np.where(~is_tabu, pf, np.inf)))
            else:
                j = int(np.argmin(tabu_until[mt]))
        t, k = int(mt[j]), int(mo[j])
        opt[t] = k
        cur_f = float(f[j])
        tabu_until[t] = it + tenure
        freq[t] += 1
        tr.offer(float(mks[j]), float(exs[j]), opt)
    return tr.result("tabu", iterations=it)
