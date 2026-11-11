"""GCLP (Kalavade & Lee 1994) and MIBS (Kalavade & Lee 1997) adapted to our multi-PE, multi-bin model.

The original papers are not available in machine-readable form here, so the parts below are *reconstructed from the
published description* and are flagged as such (same status as MAGELLAN's suitability metric in the group's notes):

* **Node selection:** among nodes whose predecessors are all mapped, the one with the largest static priority
  (HEFT upward rank with mean times and mean communication).
* **Global criticality GC** (recomputed before mapping each node): every still-unmapped node is provisionally put on its
  fastest software option, already mapped nodes keep their choice, and the schedule length L_sw of that mapping is
  computed with the simulator. If L_sw <= target, GC = 0. Otherwise unmapped nodes are moved to their fastest hardware
  option in decreasing order of (time saved) / (1 + area fraction); the smallest such set Gamma that meets the target gives
  GC = sum_{Gamma} t_sw / sum_{unmapped} t_sw, and GC = 1 if even all-hardware misses the target.
* **GCLP decision:** a node is a hardware *extremity* if its speedup is >= ``ext_hi`` and its fastest hardware option
  uses at most ``ext_area`` of the binding capacity; a software extremity if the speedup is <= ``ext_lo``. Extremities
  are mapped to their side regardless of GC. Otherwise GC > 0.5 selects the **time** objective (earliest finish over the
  allowed classes) and GC <= 0.5 the **area** objective (software if a software option exists, else the smallest
  area). GCLP has exactly two classes: software (any CPU option) and the fastest hardware option.
* **MIBS:** same GC and extremity rules, but every implementation bin of a node is a candidate and the choice minimises
  ``GC * tnorm(finish) + (1 - GC) * area_fraction`` where tnorm rescales finish times of the candidates to [0, 1].
* ``target``: ``"deadline"`` (faithful: the constrained problem "meet D with minimum area") or ``"lb"`` (latency variant
  that we add: the target is the critical-path lower bound, so GC stays high and the result minimises latency).

Repellers/attractors of the local phase (neighbour affinity) are not implemented.
"""

from __future__ import annotations

import numpy as np

from ..fast import Fast
from ..instance import INF, Instance
from ..sim import Partial
from .base import Result, Tracker
from .common import Plan, fits, mean_edge_comm, mean_task_time, rank_u


def _classes(inst: Instance):
    sw = [[k for k in inst.task_opts[i] if inst.pe_class(int(inst.o_pe[k])) == "sw"] for i in range(inst.n)]
    hw = [[k for k in inst.task_opts[i] if inst.pe_class(int(inst.o_pe[k])) == "hw"] for i in range(inst.n)]
    return sw, hw


def area_frac(inst: Instance, k: int) -> float:
    pe = int(inst.o_pe[k])
    fr = [inst.o_res[k, r] / inst.pe_cap[pe, r] for r in range(inst.R)
          if np.isfinite(inst.pe_cap[pe, r]) and inst.pe_cap[pe, r] > 0]
    return max(fr) if fr else 0.0


def critical_path_lb(inst: Instance) -> float:
    fin = np.zeros(inst.n)
    for i in range(inst.n):
        s = max((fin[u] for u, _ in inst.preds[i]), default=0.0)
        fin[i] = s + min(inst.o_time[k] for k in inst.task_opts[i])
    return float(fin.max())


def _gclp(inst: Instance, fast: Fast, mibs: bool, target: str, ext_hi=8.0, ext_area=0.25, ext_lo=1.1) -> tuple[Plan, float]:
    sw, hw = _classes(inst)
    n = inst.n
    tsw = np.array([min((inst.o_time[k] for k in sw[i]), default=INF) for i in range(n)])
    thw = np.array([min((inst.o_time[k] for k in hw[i]), default=INF) for i in range(n)])
    sw_best = [min(sw[i], key=lambda k: (inst.o_time[k], k)) if sw[i] else None for i in range(n)]
    hw_best = [min(hw[i], key=lambda k: (inst.o_time[k], k)) if hw[i] else None for i in range(n)]
    prio = rank_u(inst, mean_task_time(inst), mean_edge_comm(inst))
    tgt = inst.deadline if (target == "deadline" and inst.deadline is not None) else critical_path_lb(inst)
    speed = np.where((thw < INF) & (tsw < INF), tsw / np.where(thw > 0, thw, 1e-12), 0.0)
    afr = np.array([area_frac(inst, hw_best[i]) if hw_best[i] is not None else 0.0 for i in range(n)])
    gain = np.where((thw < INF) & (tsw < INF), (tsw - thw) / (1.0 + afr), -INF)
    tsw_eff = np.where(tsw < INF, tsw, thw)
    gorder = np.array(sorted((i for i in range(n) if hw_best[i] is not None and gain[i] > 0),
                             key=lambda i: (-gain[i], i)), dtype=np.int64)
    est = np.array([sw_best[i] if sw_best[i] is not None else hw_best[i] for i in range(n)], dtype=np.int64)
    mapped = np.zeros(n, dtype=bool)
    cur = est.copy()

    def global_criticality() -> float:
        unm = np.where(~mapped)[0]
        if len(unm) == 0:
            return 0.0
        mk, _ = fast.evaluate(cur)
        if mk <= tgt:
            return 0.0
        cand = gorder[~mapped[gorder]]
        denom = float(tsw_eff[unm].sum()) or 1.0

        def mk_with(j: int) -> float:
            b = cur.copy()
            for i in cand[:j]:
                b[i] = hw_best[i]
            return fast.evaluate(b)[0]

        if len(cand) == 0 or mk_with(len(cand)) > tgt:
            return 1.0
        lo, hi = 0, len(cand)
        while lo < hi:
            mid = (lo + hi) // 2
            if mk_with(mid) <= tgt:
                hi = mid
            else:
                lo = mid + 1
        return float(tsw_eff[cand[:lo]].sum()) / denom

    st = Partial(inst, True, False)
    remaining = [len(inst.preds[i]) for i in range(n)]
    ready = [i for i in range(n) if remaining[i] == 0]
    order = []
    stride = max(1, n // 200)          # GC is refreshed every `stride` mapped nodes (stride = 1 for n < 400)
    gc = 0.0
    while ready:
        ready.sort(key=lambda i: (-prio[i], i))
        i = ready.pop(0)
        if len(order) % stride == 0:
            gc = global_criticality()
        fit_hw = [k for k in hw[i] if fits(inst, st.usage, k)]
        pool_hw = fit_hw if mibs else [k for k in [hw_best[i]] if k is not None and k in fit_hw]
        tr = {k: st.trial(i, k) for k in list(sw[i]) + pool_hw}
        tr = {k: t for k, t in tr.items() if t is not None}
        if not tr:  # nothing routable / fitting: any reachable option
            tr = {k: st.trial(i, k) for k in inst.task_opts[i]}
            tr = {k: t for k, t in tr.items() if t is not None}
        sw_c = [k for k in tr if k in sw[i]]
        hw_c = [k for k in tr if k not in sw[i]]
        ext_hw = hw_best[i] is not None and speed[i] >= ext_hi and afr[i] <= ext_area
        ext_sw = bool(sw_c) and speed[i] <= ext_lo
        allowed = hw_c if (ext_hw and hw_c) else sw_c if (ext_sw and sw_c) else list(tr)
        if mibs:
            fins = [tr[k][1] for k in allowed]
            lo_f, hi_f = min(fins), max(fins)
            span = (hi_f - lo_f) or 1.0
            k = min(allowed, key=lambda k: (gc * (tr[k][1] - lo_f) / span + (1 - gc) * area_frac(inst, k),
                                            tr[k][1], k))
        elif gc > 0.5:
            k = min(allowed, key=lambda k: (tr[k][1], k))
        else:
            pool = [k for k in allowed if k in sw_c] or allowed
            k = min(pool, key=lambda k: (area_frac(inst, k), tr[k][1], k))
        st.place(i, k, tr[k])
        mapped[i] = True
        cur[i] = k
        order.append(i)
        for s, _e in inst.succs[i]:
            remaining[s] -= 1
            if remaining[s] == 0:
                ready.append(s)
    return Plan(st.opt.copy(), order, False), float(st.finish.max())


def solve_gclp(inst, fast, seed, budget_s, target="deadline") -> Result:
    tr = Tracker(inst, fast, budget_s)
    plan, mk = _gclp(inst, fast, False, target)
    tr.offer_plan(plan, mk)
    return tr.result("gclp" if target == "deadline" else "gclp_lat")


def solve_mibs(inst, fast, seed, budget_s, target="deadline") -> Result:
    tr = Tracker(inst, fast, budget_s)
    plan, mk = _gclp(inst, fast, True, target)
    tr.offer_plan(plan, mk)
    return tr.result("mibs" if target == "deadline" else "mibs_lat")
