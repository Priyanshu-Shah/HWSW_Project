"""Compiled evaluator (numba) used inside the search loops. Same semantics as ``sim.simulate`` in default mode
(non-insertion, transfers booked when the consumer is placed, upward-rank or explicit key priority).

``tests/test_fast.py`` checks makespan and capacity excess against the reference simulator on random mappings.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .instance import Instance

PEN = 20.0          # capacity penalty: f = makespan * (1 + PEN * excess)


# -----------------------------------------------------------------------------------------------------------
# kernel
# -----------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _sim(n, pred_ptr, pred_t, pred_e, succ_ptr, succ_t, succ_e, o_pe, o_time, o_res, ctime, cif,
         pe_excl, if_shared, pe_cap, opt, key, use_rank, contention, start, finish):
    m = pe_excl.shape[0]
    R = o_res.shape[1]
    # priority keys (smaller = earlier)
    if use_rank:
        rank = np.zeros(n)
        for i in range(n - 1, -1, -1):
            best = 0.0
            pi = o_pe[opt[i]]
            for q in range(succ_ptr[i], succ_ptr[i + 1]):
                s = succ_t[q]
                v = ctime[succ_e[q], pi, o_pe[opt[s]]] + rank[s]
                if v > best:
                    best = v
            rank[i] = o_time[opt[i]] + best
        kk = -rank
    else:
        kk = key
    # binary heap over (key, idx)
    hk = np.empty(n)
    hi = np.empty(n, dtype=np.int64)
    hs = 0
    remaining = np.empty(n, dtype=np.int64)
    for i in range(n):
        remaining[i] = pred_ptr[i + 1] - pred_ptr[i]
    for i in range(n):
        if remaining[i] == 0:
            # push
            j = hs
            hs += 1
            while j > 0:
                p = (j - 1) // 2
                if hk[p] < kk[i] or (hk[p] == kk[i] and hi[p] < i):
                    break
                hk[j] = hk[p]
                hi[j] = hi[p]
                j = p
            hk[j] = kk[i]
            hi[j] = i
    pe_free = np.zeros(m)
    nif = if_shared.shape[0]
    if_free = np.zeros(nif)
    usage = np.zeros((m, R))
    tf = np.empty(64)
    ti = np.empty(64, dtype=np.int64)
    te = np.empty(64, dtype=np.int64)
    mk = 0.0
    placed = 0
    while hs > 0:
        i = hi[0]
        # pop
        hs -= 1
        lk = hk[hs]
        li = hi[hs]
        j = 0
        while True:
            c = 2 * j + 1
            if c >= hs:
                break
            if c + 1 < hs and (hk[c + 1] < hk[c] or (hk[c + 1] == hk[c] and hi[c + 1] < hi[c])):
                c += 1
            if lk < hk[c] or (lk == hk[c] and li < hi[c]):
                break
            hk[j] = hk[c]
            hi[j] = hi[c]
            j = c
        if hs > 0:
            hk[j] = lk
            hi[j] = li
        # place i
        k = opt[i]
        b = o_pe[k]
        deg = pred_ptr[i + 1] - pred_ptr[i]
        if deg > tf.shape[0]:
            tf = np.empty(deg)
            ti = np.empty(deg, dtype=np.int64)
            te = np.empty(deg, dtype=np.int64)
        for d in range(deg):
            q = pred_ptr[i] + d
            u = pred_t[q]
            # insertion sort by (finish, idx)
            f = finish[u]
            p = d
            while p > 0 and (tf[p - 1] > f or (tf[p - 1] == f and ti[p - 1] > u)):
                tf[p] = tf[p - 1]
                ti[p] = ti[p - 1]
                te[p] = te[p - 1]
                p -= 1
            tf[p] = f
            ti[p] = u
            te[p] = pred_e[q]
        ready = 0.0
        for d in range(deg):
            u = ti[d]
            e = te[d]
            a = o_pe[opt[u]]
            c = ctime[e, a, b]
            if c == np.inf:
                return np.inf, np.inf
            st = tf[d]
            if a != b:
                fi = cif[e, a, b]
                if contention and if_shared[fi]:
                    if if_free[fi] > st:
                        st = if_free[fi]
                    if_free[fi] = st + c
            if st + c > ready:
                ready = st + c
        if pe_excl[b]:
            s0 = ready if ready > pe_free[b] else pe_free[b]
        else:
            s0 = ready
        f0 = s0 + o_time[k]
        if pe_excl[b]:
            pe_free[b] = f0
        start[i] = s0
        finish[i] = f0
        if f0 > mk:
            mk = f0
        for r in range(R):
            usage[b, r] += o_res[k, r]
        placed += 1
        for q in range(succ_ptr[i], succ_ptr[i + 1]):
            s = succ_t[q]
            remaining[s] -= 1
            if remaining[s] == 0:
                j = hs
                hs += 1
                while j > 0:
                    p = (j - 1) // 2
                    if hk[p] < kk[s] or (hk[p] == kk[s] and hi[p] < s):
                        break
                    hk[j] = hk[p]
                    hi[j] = hi[p]
                    j = p
                hk[j] = kk[s]
                hi[j] = s
    excess = 0.0
    for p in range(m):
        for r in range(R):
            cp = pe_cap[p, r]
            if cp < np.inf:
                if cp > 0.0:
                    v = usage[p, r] / cp - 1.0
                    if v > 0.0:
                        excess += v
                elif usage[p, r] > 0.0:
                    excess += 1.0
    if excess < 1e-9:
        excess = 0.0
    return mk, excess


@njit(cache=True)
def _eval_moves(n, pred_ptr, pred_t, pred_e, succ_ptr, succ_t, succ_e, o_pe, o_time, o_res, ctime, cif,
                pe_excl, if_shared, pe_cap, opt, key, use_rank, mv_task, mv_opt, out_mk, out_ex):
    start = np.empty(n)
    finish = np.empty(n)
    for q in range(mv_task.shape[0]):
        t = mv_task[q]
        old = opt[t]
        opt[t] = mv_opt[q]
        mk, ex = _sim(n, pred_ptr, pred_t, pred_e, succ_ptr, succ_t, succ_e, o_pe, o_time, o_res, ctime, cif,
                      pe_excl, if_shared, pe_cap, opt, key, use_rank, True, start, finish)
        out_mk[q] = mk
        out_ex[q] = ex
        opt[t] = old


@njit(cache=True)
def _eval_pop(n, pred_ptr, pred_t, pred_e, succ_ptr, succ_t, succ_e, o_pe, o_time, o_res, ctime, cif,
              pe_excl, if_shared, pe_cap, optm, keym, use_rank, out_mk, out_ex):
    start = np.empty(n)
    finish = np.empty(n)
    for q in range(optm.shape[0]):
        mk, ex = _sim(n, pred_ptr, pred_t, pred_e, succ_ptr, succ_t, succ_e, o_pe, o_time, o_res, ctime, cif,
                      pe_excl, if_shared, pe_cap, optm[q], keym[q], use_rank, True, start, finish)
        out_mk[q] = mk
        out_ex[q] = ex


@njit(cache=True)
def _seed(s):
    np.random.seed(s)


@njit(cache=True)
def _sa_chunk(n, pred_ptr, pred_t, pred_e, succ_ptr, succ_t, succ_e, o_pe, o_time, o_res, ctime, cif,
              pe_excl, if_shared, pe_cap, opt, cur_f, best_opt, best_f, movable, topt_ptr, topt_list,
              t_hi, t_lo, iters, pen):
    """Metropolis chain with geometric cooling t_hi -> t_lo over ``iters`` single-task option moves.

    ``best_*`` track the best *feasible* mapping (capacity excess 0). Returns (cur_f, best_f, accepted)."""
    start = np.empty(n)
    finish = np.empty(n)
    key = np.empty(1)
    acc = 0
    nm = movable.shape[0]
    alpha = (t_lo / t_hi) ** (1.0 / max(iters, 1))
    temp = t_hi
    for _ in range(iters):
        t = movable[np.random.randint(nm)]
        a, b = topt_ptr[t], topt_ptr[t + 1]
        old = opt[t]
        pos = 0
        for z in range(a, b):
            if topt_list[z] == old:
                pos = z - a
                break
        r = np.random.randint(b - a - 1)
        if r >= pos:
            r += 1
        pick = topt_list[a + r]
        opt[t] = pick
        mk, ex = _sim(n, pred_ptr, pred_t, pred_e, succ_ptr, succ_t, succ_e, o_pe, o_time, o_res, ctime, cif,
                      pe_excl, if_shared, pe_cap, opt, key, True, True, start, finish)
        if mk == np.inf:
            opt[t] = old
            temp *= alpha
            continue
        f = mk * (1.0 + pen * ex)
        if f <= cur_f or np.random.random() < np.exp(-(f - cur_f) / temp):
            cur_f = f
            acc += 1
            if ex == 0.0 and mk < best_f:
                best_f = mk
                best_opt[:] = opt
        else:
            opt[t] = old
        temp *= alpha
    return cur_f, best_f, acc


# -----------------------------------------------------------------------------------------------------------
class Fast:
    """Holds the flat arrays of an instance and wraps the compiled kernels."""

    def __init__(self, inst: Instance):
        self.inst = inst
        n = inst.n
        self.n = n
        pp, pt, pe_ = [0], [], []
        sp, st, se = [0], [], []
        for i in range(n):
            for u, e in inst.preds[i]:
                pt.append(u); pe_.append(e)
            pp.append(len(pt))
            for v, e in inst.succs[i]:
                st.append(v); se.append(e)
            sp.append(len(st))
        self.pred_ptr = np.array(pp, dtype=np.int64)
        self.pred_t = np.array(pt, dtype=np.int64)
        self.pred_e = np.array(pe_, dtype=np.int64)
        self.succ_ptr = np.array(sp, dtype=np.int64)
        self.succ_t = np.array(st, dtype=np.int64)
        self.succ_e = np.array(se, dtype=np.int64)
        self.o_pe = np.ascontiguousarray(inst.o_pe, dtype=np.int64)
        self.o_time = np.ascontiguousarray(inst.o_time, dtype=np.float64)
        self.o_res = np.ascontiguousarray(inst.o_res, dtype=np.float64).reshape(inst.K, inst.R)
        self.ctime = np.ascontiguousarray(inst.ctime, dtype=np.float64).reshape(inst.E, inst.m, inst.m)
        self.cif = np.ascontiguousarray(inst.cif, dtype=np.int64).reshape(inst.E, inst.m, inst.m)
        self.pe_excl = np.ascontiguousarray(inst.pe_excl, dtype=np.bool_)
        self.if_shared = np.ascontiguousarray(inst.iface_shared if len(inst.iface_shared) else np.zeros(1, dtype=bool),
                                              dtype=np.bool_)
        self.pe_cap = np.ascontiguousarray(inst.pe_cap, dtype=np.float64).reshape(inst.m, inst.R)
        ptr = [0]; lst: list[int] = []
        for i in range(n):
            lst.extend(inst.task_opts[i]); ptr.append(len(lst))
        self.topt_ptr = np.array(ptr, dtype=np.int64)
        self.topt_list = np.array(lst, dtype=np.int64)
        self.movable = np.array([i for i in range(n) if len(inst.task_opts[i]) > 1], dtype=np.int64)
        self._dummy_key = np.zeros(1)
        self.n_evals = 0
        self.o_energy = np.ascontiguousarray(inst.o_energy, dtype=np.float64)
        # averages used by HEFT / PEFT / GCLP
        m = inst.m
        off = ~np.eye(m, dtype=bool)
        fin = np.isfinite(self.ctime) & off[None, :, :]
        cnt = fin.sum(axis=(1, 2))
        tot = np.where(fin, self.ctime, 0.0).sum(axis=(1, 2))
        self.cbar = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
        self.wbar = np.array([self.o_time[inst.task_opts[i]].mean() for i in range(n)])
        self._heft_buf = None
        # trivial always-feasible fallback: every task on its fastest option of a capacity-free PE (instance preprocessing)
        free_pe = np.isinf(self.pe_cap).all(axis=1) if inst.R else np.ones(inst.m, dtype=bool)
        best = np.empty(n, dtype=np.int64)
        for i in range(n):
            ks = np.array(inst.task_opts[i])
            pool = ks[free_pe[self.o_pe[ks]]]
            if len(pool) == 0:
                pool = ks
            best[i] = pool[np.argmin(self.o_time[pool])]      # first minimum -> lowest option id on ties
        self.allsw_opt = best
        self.allsw_mk, self.allsw_ex = self.evaluate(best)
        self.n_evals = 0

    @property
    def _a(self):
        return (self.n, self.pred_ptr, self.pred_t, self.pred_e, self.succ_ptr, self.succ_t, self.succ_e, self.o_pe,
                self.o_time, self.o_res, self.ctime, self.cif, self.pe_excl, self.if_shared, self.pe_cap)

    def evaluate(self, opt: np.ndarray, key: np.ndarray | None = None, with_times: bool = False):
        """(makespan, capacity excess) of ``opt``; ``key`` = explicit priorities (smaller first) or None for rank."""
        self.n_evals += 1
        start = np.empty(self.n)
        finish = np.empty(self.n)
        use_rank = key is None
        mk, ex = _sim(*self._a, np.ascontiguousarray(opt, dtype=np.int64), self._dummy_key if use_rank else key,
                      use_rank, True, start, finish)
        return (mk, ex, start, finish) if with_times else (mk, ex)

    def eval_moves(self, opt: np.ndarray, mv_task: np.ndarray, mv_opt: np.ndarray):
        out_mk = np.empty(len(mv_task)); out_ex = np.empty(len(mv_task))
        self.n_evals += len(mv_task)
        _eval_moves(*self._a, np.ascontiguousarray(opt, dtype=np.int64), self._dummy_key, True,
                    np.ascontiguousarray(mv_task, dtype=np.int64), np.ascontiguousarray(mv_opt, dtype=np.int64),
                    out_mk, out_ex)
        return out_mk, out_ex

    def eval_pop(self, optm: np.ndarray, keym: np.ndarray | None = None):
        q = optm.shape[0]
        out_mk = np.empty(q); out_ex = np.empty(q)
        self.n_evals += q
        use_rank = keym is None
        _eval_pop(*self._a, np.ascontiguousarray(optm, dtype=np.int64),
                  np.zeros((q, 1)) if use_rank else np.ascontiguousarray(keym, dtype=np.float64), use_rank,
                  out_mk, out_ex)
        return out_mk, out_ex

    # -- compiled HEFT / PEFT -------------------------------------------------------------------------
    def _run_list(self, key, bias, fixed=None, contention=True, insertion=True):
        from .heftk import heft_kernel
        n = self.n
        opt = np.empty(n, dtype=np.int64)
        start = np.empty(n)
        finish = np.empty(n)
        order = np.empty(n, dtype=np.int64)
        fx = np.full(n, -1, dtype=np.int64)
        if fixed:
            for i, k in fixed.items():
                fx[i] = k
        ok = heft_kernel(n, self.pred_ptr, self.pred_t, self.pred_e, self.succ_ptr, self.succ_t, self.succ_e, self.o_pe,
                         self.o_time, self.o_res, self.o_energy, self.ctime, self.cif, self.pe_excl, self.if_shared,
                         self.pe_cap, self.topt_ptr, self.topt_list, key, bias, fx, contention, insertion, opt, start,
                         finish, order)
        if not ok:
            raise ValueError("list scheduler: a task has no routable option")
        return opt, order, float(finish.max()) if n else 0.0

    def rank_u(self) -> np.ndarray:
        from .heftk import rank_u_kernel
        return rank_u_kernel(self.n, self.succ_ptr, self.succ_t, self.succ_e, self.wbar, self.cbar)

    def heft(self, fixed=None, contention=True, insertion=True):
        """Compiled HEFT; returns (Plan, makespan)."""
        from .algos.common import Plan
        key = -self.rank_u()
        opt, order, mk = self._run_list(key, np.zeros(self.inst.K), fixed, contention, insertion)
        return Plan(opt, order.tolist(), insertion), mk

    def peft(self):
        from .algos.common import Plan
        from .heftk import oct_kernel
        octv = oct_kernel(self.n, self.succ_ptr, self.succ_t, self.succ_e, self.o_pe, self.o_time, self.topt_ptr,
                          self.topt_list, self.cbar, self.inst.K)
        rank = np.array([octv[self.inst.task_opts[i]].mean() for i in range(self.n)])
        opt, order, mk = self._run_list(-rank, octv)
        return Plan(opt, order.tolist(), True), mk

    def excess(self, opt: np.ndarray) -> float:
        """Capacity excess of a mapping (same definition as ``sim.cap_excess``), vectorised."""
        u = np.zeros((self.inst.m, self.inst.R))
        np.add.at(u, self.o_pe[opt], self.o_res[opt])
        with np.errstate(divide="ignore", invalid="ignore"):
            over = np.where(np.isfinite(self.pe_cap) & (self.pe_cap > 0), np.maximum(u / self.pe_cap - 1.0, 0.0), 0.0)
        over = np.where(np.isfinite(self.pe_cap) & (self.pe_cap == 0) & (u > 0), 1.0, over)
        t = float(over.sum())
        return 0.0 if t < 1e-9 else t

    def seed(self, s: int):
        _seed(int(s))

    def sa_chunk(self, opt, cur_f, best_opt, best_f, t_hi, t_lo, iters, pen=PEN):
        self.n_evals += iters
        return _sa_chunk(*self._a, opt, cur_f, best_opt, best_f, self.movable, self.topt_ptr, self.topt_list,
                         t_hi, t_lo, iters, pen)


def penalised(mk: float, ex: float, pen: float = PEN) -> float:
    return mk * (1.0 + pen * ex)


def warmup() -> None:
    """Trigger (or load from cache) the numba compilation of every kernel on a tiny instance."""
    from .fixtures import heft_paper_example
    inst = heft_paper_example()
    fa = Fast(inst)
    opt = np.array([inst.task_opts[i][0] for i in range(inst.n)], dtype=np.int64)
    fa.evaluate(opt)
    fa.evaluate(opt, np.arange(inst.n, dtype=float))
    fa.eval_moves(opt, np.array([0]), np.array([inst.task_opts[0][1]]))
    fa.eval_pop(np.array([opt, opt]), np.zeros((2, inst.n)))
    fa.eval_pop(np.array([opt, opt]))
    fa.heft()
    fa.peft()
    fa.heft({0: inst.task_opts[0][0]})
    fa.seed(0)
    fa.sa_chunk(opt.copy(), 1.0, opt.copy(), 1.0, 1.0, 0.1, 5)
