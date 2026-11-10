"""Compiled insertion-based list scheduler for HEFT / PEFT / MAGELLAN completions.

Same semantics as ``algos/heft.py`` + ``sim.Partial(insertion=True)`` (tested for exact equality):

* tasks leave a ready heap by static key (HEFT: -rank_u; PEFT: -rank_oct);
* for each task every option is tried; the choice minimises ``(does not fit, finish + bias[k], energy, k)`` where ``bias``
  is the PEFT optimistic cost (zeros for HEFT) and "fits" respects capacity including the resources still *reserved* for
  fixed tasks that are not placed yet;
* input transfers are booked in order of producer finish time; with ``insertion`` they may use idle gaps of a shared
  interface; tasks use idle gaps of exclusive PEs;
* ``fixed[i] >= 0`` forces task ``i`` onto that option (MAGELLAN completion).
"""

from __future__ import annotations

import numpy as np
from numba import njit

EPS = 1e-9


@njit(cache=True)
def _place(i, k, commit, pred_ptr, pred_t, pred_e, o_pe, o_time, ctime, cif, pe_excl, if_shared, contention,
           opt, finish, start, pe_s, pe_f, pe_c, ifs, iff, ifc, tf, ti, te, bk_if, bk_pos, insertion):
    b = o_pe[k]
    deg = pred_ptr[i + 1] - pred_ptr[i]
    for d in range(deg):
        q = pred_ptr[i] + d
        u = pred_t[q]
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
    nb = 0
    ready = 0.0
    ok = True
    for d in range(deg):
        u = ti[d]
        e = te[d]
        a = o_pe[opt[u]]
        c = ctime[e, a, b]
        if c == np.inf:
            ok = False
            break
        st = tf[d]
        if a != b:
            fi = cif[e, a, b]
            if contention and if_shared[fi]:
                cnt = ifc[fi]
                if insertion:
                    t = st
                    for j in range(cnt):
                        if t + c <= ifs[fi, j] + EPS:
                            break
                        if iff[fi, j] > t:
                            t = iff[fi, j]
                    st = t
                    j = cnt
                    while j > 0 and ifs[fi, j - 1] > st:
                        j -= 1
                else:
                    if cnt > 0 and iff[fi, cnt - 1] > st:
                        st = iff[fi, cnt - 1]
                    j = cnt
                for z in range(cnt, j, -1):
                    ifs[fi, z] = ifs[fi, z - 1]
                    iff[fi, z] = iff[fi, z - 1]
                ifs[fi, j] = st
                iff[fi, j] = st + c
                ifc[fi] = cnt + 1
                bk_if[nb] = fi
                bk_pos[nb] = j
                nb += 1
        if st + c > ready:
            ready = st + c
    fin = np.inf
    s = 0.0
    if ok:
        dur = o_time[k]
        if pe_excl[b]:
            if insertion:
                t = ready
                cnt = pe_c[b]
                for j in range(cnt):
                    if t + dur <= pe_s[b, j] + EPS:
                        break
                    if pe_f[b, j] > t:
                        t = pe_f[b, j]
                s = t
            else:
                cnt = pe_c[b]
                s = ready
                if cnt > 0 and pe_f[b, cnt - 1] > s:
                    s = pe_f[b, cnt - 1]
        else:
            s = ready
        fin = s + dur
    if commit and ok:
        if pe_excl[b]:
            cnt = pe_c[b]
            j = cnt
            while j > 0 and pe_s[b, j - 1] > s:
                j -= 1
            for z in range(cnt, j, -1):
                pe_s[b, z] = pe_s[b, z - 1]
                pe_f[b, z] = pe_f[b, z - 1]
            pe_s[b, j] = s
            pe_f[b, j] = fin
            pe_c[b] = cnt + 1
        start[i] = s
        finish[i] = fin
        opt[i] = k
    else:
        for q in range(nb - 1, -1, -1):          # roll the tentative transfers back
            fi = bk_if[q]
            j = bk_pos[q]
            cnt = ifc[fi]
            for z in range(j, cnt - 1):
                ifs[fi, z] = ifs[fi, z + 1]
                iff[fi, z] = iff[fi, z + 1]
            ifc[fi] = cnt - 1
    return fin


@njit(cache=True)
def heft_kernel(n, pred_ptr, pred_t, pred_e, succ_ptr, succ_t, succ_e, o_pe, o_time, o_res, o_energy, ctime, cif,
                pe_excl, if_shared, pe_cap, topt_ptr, topt_list, key, bias, fixed, contention, insertion,
                opt, start, finish, order):
    m = pe_excl.shape[0]
    R = o_res.shape[1]
    E = ctime.shape[0]
    nif = if_shared.shape[0]
    pe_s = np.zeros((m, n + 1))
    pe_f = np.zeros((m, n + 1))
    pe_c = np.zeros(m, dtype=np.int64)
    ifs = np.zeros((nif, E + 2))
    iff = np.zeros((nif, E + 2))
    ifc = np.zeros(nif, dtype=np.int64)
    usage = np.zeros((m, R))
    reserved = np.zeros((m, R))
    for i in range(n):
        if fixed[i] >= 0:
            for r in range(R):
                reserved[o_pe[fixed[i]], r] += o_res[fixed[i], r]
    tf = np.empty(64)
    ti = np.empty(64, dtype=np.int64)
    te = np.empty(64, dtype=np.int64)
    bk_if = np.empty(64, dtype=np.int64)
    bk_pos = np.empty(64, dtype=np.int64)
    remaining = np.empty(n, dtype=np.int64)
    hk = np.empty(n)
    hi = np.empty(n, dtype=np.int64)
    hs = 0
    for i in range(n):
        remaining[i] = pred_ptr[i + 1] - pred_ptr[i]
    maxdeg = 0
    for i in range(n):
        d = pred_ptr[i + 1] - pred_ptr[i]
        if d > maxdeg:
            maxdeg = d
    if maxdeg > 64:
        tf = np.empty(maxdeg)
        ti = np.empty(maxdeg, dtype=np.int64)
        te = np.empty(maxdeg, dtype=np.int64)
        bk_if = np.empty(maxdeg, dtype=np.int64)
        bk_pos = np.empty(maxdeg, dtype=np.int64)
    for i in range(n):
        if remaining[i] == 0:
            j = hs
            hs += 1
            while j > 0:
                p = (j - 1) // 2
                if hk[p] < key[i] or (hk[p] == key[i] and hi[p] < i):
                    break
                hk[j] = hk[p]
                hi[j] = hi[p]
                j = p
            hk[j] = key[i]
            hi[j] = i
    opt[:] = -1
    cntp = 0
    while hs > 0:
        i = hi[0]
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
        # choose the option
        if fixed[i] >= 0:
            kbest = fixed[i]
            for r in range(R):
                reserved[o_pe[kbest], r] -= o_res[kbest, r]
        else:
            kbest = -1
            bfit = 2
            bfin = np.inf
            ben = np.inf
            for z in range(topt_ptr[i], topt_ptr[i + 1]):
                k = topt_list[z]
                fin = _place(i, k, False, pred_ptr, pred_t, pred_e, o_pe, o_time, ctime, cif, pe_excl, if_shared,
                             contention, opt, finish, start, pe_s, pe_f, pe_c, ifs, iff, ifc, tf, ti, te, bk_if,
                             bk_pos, insertion)
                if fin == np.inf:
                    continue
                pe = o_pe[k]
                fit = 0
                for r in range(R):
                    if usage[pe, r] + reserved[pe, r] + o_res[k, r] > pe_cap[pe, r] * (1.0 + 1e-9):
                        fit = 1
                        break
                cost = fin + bias[k]
                better = False
                if kbest < 0:
                    better = True
                elif fit < bfit:
                    better = True
                elif fit == bfit:
                    if cost < bfin:
                        better = True
                    elif cost == bfin:
                        if o_energy[k] < ben:
                            better = True
                        elif o_energy[k] == ben and k < kbest:
                            better = True
                if better:
                    kbest, bfit, bfin, ben = k, fit, cost, o_energy[k]
            if kbest < 0:
                return False
        fin = _place(i, kbest, True, pred_ptr, pred_t, pred_e, o_pe, o_time, ctime, cif, pe_excl, if_shared,
                     contention, opt, finish, start, pe_s, pe_f, pe_c, ifs, iff, ifc, tf, ti, te, bk_if, bk_pos,
                     insertion)
        if fin == np.inf:
            return False
        pe = o_pe[kbest]
        for r in range(R):
            usage[pe, r] += o_res[kbest, r]
        order[cntp] = i
        cntp += 1
        for q in range(succ_ptr[i], succ_ptr[i + 1]):
            s = succ_t[q]
            remaining[s] -= 1
            if remaining[s] == 0:
                j = hs
                hs += 1
                while j > 0:
                    p = (j - 1) // 2
                    if hk[p] < key[s] or (hk[p] == key[s] and hi[p] < s):
                        break
                    hk[j] = hk[p]
                    hi[j] = hi[p]
                    j = p
                hk[j] = key[s]
                hi[j] = s
    return True


@njit(cache=True)
def rank_u_kernel(n, succ_ptr, succ_t, succ_e, w, cbar):
    r = np.zeros(n)
    for i in range(n - 1, -1, -1):
        best = 0.0
        for q in range(succ_ptr[i], succ_ptr[i + 1]):
            v = cbar[succ_e[q]] + r[succ_t[q]]
            if v > best:
                best = v
        r[i] = w[i] + best
    return r


@njit(cache=True)
def oct_kernel(n, succ_ptr, succ_t, succ_e, o_pe, o_time, topt_ptr, topt_list, cbar, K):
    oct_ = np.zeros(K)
    for i in range(n - 1, -1, -1):
        for z in range(topt_ptr[i], topt_ptr[i + 1]):
            k = topt_list[z]
            worst = 0.0
            for q in range(succ_ptr[i], succ_ptr[i + 1]):
                s = succ_t[q]
                e = succ_e[q]
                best = np.inf
                for y in range(topt_ptr[s], topt_ptr[s + 1]):
                    kw = topt_list[y]
                    comm = 0.0 if o_pe[kw] == o_pe[k] else cbar[e]
                    v = oct_[kw] + o_time[kw] + comm
                    if v < best:
                        best = v
                if best > worst:
                    worst = best
            oct_[k] = worst
    return oct_
