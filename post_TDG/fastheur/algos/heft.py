"""HEFT (Topcuoglu, Hariri & Wu 2002) and PEFT (Arabnejad & Barbosa 2014) on (PE, implementation) options.

* a "processor" is a legal option; averages are taken over a task's options;
* the average edge cost is the mean transfer time over distinct, connected PE pairs;
* extension for the capacity-limited FPGA: options that would overflow a PE are skipped unless none fits.
"""

from __future__ import annotations

import numpy as np

from ..instance import INF, Instance
from ..sim import Partial
from .common import Plan, fits, mean_edge_comm, mean_task_time, rank_u


def _choose(inst: Instance, st: Partial, i: int, cost_fn, reserved: np.ndarray | None = None):
    """Best option for task i by (fits, cost(finish, k), energy, k). ``reserved`` = resources of not-yet-placed
    fixed tasks, which free tasks must leave room for."""
    best = None
    usage = st.usage if reserved is None else st.usage + reserved
    for k in inst.task_opts[i]:
        t = st.trial(i, k)
        if t is None:
            continue
        key = (0 if fits(inst, usage, k) else 1, cost_fn(t[1], k), inst.o_energy[k], k)
        if best is None or key < best[0]:
            best = (key, k, t)
    if best is None:
        raise ValueError(f"task {i}: no reachable option")
    return best[1], best[2]


def heft(inst: Instance, fixed: dict[int, int] | None = None, insertion: bool = True, contention: bool = True) -> Plan:
    """HEFT. ``fixed`` pins some tasks to given option ids (used by MAGELLAN to complete partial mappings)."""
    w, c = mean_task_time(inst), mean_edge_comm(inst)
    rk = rank_u(inst, w, c)
    order = sorted(range(inst.n), key=lambda i: (-rk[i], i))
    st = Partial(inst, contention, insertion)
    reserved = None
    if fixed:
        reserved = np.zeros((inst.m, inst.R))
        for i, k in fixed.items():
            reserved[inst.o_pe[k]] += inst.o_res[k]
    for i in order:
        if fixed and i in fixed:
            k = fixed[i]
            reserved[inst.o_pe[k]] -= inst.o_res[k]
            st.place(i, k)
        else:
            k, t = _choose(inst, st, i, lambda f, k: f, reserved)
            st.place(i, k, t)
    return Plan(st.opt.copy(), list(order), insertion)


def oct_table(inst: Instance, cbar: np.ndarray) -> list[dict[int, float]]:
    """Optimistic cost table OCT(t, option) of PEFT. Indexed by global option id."""
    oct_ = [dict() for _ in range(inst.n)]
    for i in range(inst.n - 1, -1, -1):
        for k in inst.task_opts[i]:
            pk = int(inst.o_pe[k])
            worst = 0.0
            for s, e in inst.succs[i]:
                best = INF
                for kw in inst.task_opts[s]:
                    comm = 0.0 if int(inst.o_pe[kw]) == pk else cbar[e]
                    v = oct_[s][kw] + inst.o_time[kw] + comm
                    if v < best:
                        best = v
                worst = max(worst, best)
            oct_[i][k] = worst
    return oct_


def peft(inst: Instance) -> Plan:
    cbar = mean_edge_comm(inst)
    oct_ = oct_table(inst, cbar)
    rank_oct = np.array([np.mean(list(oct_[i].values())) for i in range(inst.n)])
    st = Partial(inst, True, True)
    remaining = [len(inst.preds[i]) for i in range(inst.n)]
    ready = [i for i in range(inst.n) if remaining[i] == 0]
    order = []
    while ready:
        ready.sort(key=lambda i: (-rank_oct[i], i))
        i = ready.pop(0)
        k, t = _choose(inst, st, i, lambda f, k, i=i: f + oct_[i][k])
        st.place(i, k, t)
        order.append(i)
        for s, _e in inst.succs[i]:
            remaining[s] -= 1
            if remaining[s] == 0:
                ready.append(s)
    return Plan(st.opt.copy(), order, True)


def solve_heft(inst: Instance, fast, seed: int, budget_s: float):
    """Compiled HEFT (identical to ``heft`` above; see tests/test_heftk.py)."""
    from .base import Tracker
    tr = Tracker(inst, fast, budget_s)
    plan, mk = fast.heft()
    tr.offer_plan(plan, mk)
    return tr.result("heft")


def solve_peft(inst: Instance, fast, seed: int, budget_s: float):
    from .base import Tracker
    tr = Tracker(inst, fast, budget_s)
    plan, mk = fast.peft()
    tr.offer_plan(plan, mk)
    return tr.result("peft")
