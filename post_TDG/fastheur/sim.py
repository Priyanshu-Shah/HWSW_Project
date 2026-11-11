"""Reference list-scheduling simulator (pure Python), implementing ``format/model.md`` of the data package.

This is the *specification* of the evaluator. ``fastheur/fast.py`` holds a compiled twin used inside the search loops;
the tests assert that both agree on random mappings.

Semantics (nominal costs):

* exclusive PEs run one task at a time, non-preemptive; with ``insertion=True`` a task may start in an earlier idle gap
  that is long enough (HEFT / PEFT);
* spatial PEs (FPGA fabric) give every mapped task its own hardware: tasks start as soon as their data is there, and the
  resource usage of all tasks on the PE adds up against its capacity;
* a transfer between different PEs uses the fastest interface for its size; transfers on a *shared* interface are
  serialised in the order they are scheduled (no gap insertion); same-PE communication is free;
* default mode: ready tasks are started in priority order (default = upward rank of the mapped graph); a task's input
  transfers are booked when the task is placed, ordered by producer finish time;
* unified mode (priority list that also contains transfers, item ``n + edge_id``): transfers are list-scheduled items of
  their own. A transfer is ready when its producer is placed; a task is ready when all its inputs have arrived.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from .instance import INF, Instance

EPS = 1e-9


@dataclass
class Schedule:
    makespan: float
    start: np.ndarray
    finish: np.ndarray
    comms: list[tuple[int, float, float, int]]      # (edge, start, finish, interface)
    usage: np.ndarray                               # (m, R)
    cap_excess: float                               # sum over (PE, resource) of max(0, use/cap - 1)
    energy: float
    order: list[int] = field(default_factory=list)  # tasks in placement order

    @property
    def capacity_ok(self) -> bool:
        return self.cap_excess <= 1e-9


def mean_comm_time(inst: Instance, e: int) -> float:
    """Average transfer time of edge ``e`` over ordered pairs of distinct, connected PEs."""
    vals = [inst.ctime[e, a, b] for a in range(inst.m) for b in range(inst.m) if a != b and inst.ctime[e, a, b] < INF]
    return float(np.mean(vals)) if vals else 0.0


class Partial:
    """A partial schedule that supports trial placement (earliest start/finish) and commitment."""

    def __init__(self, inst: Instance, contention: bool = True, insertion: bool = False):
        self.inst = inst
        self.contention = contention
        self.insertion = insertion
        self.pe_free = np.zeros(inst.m)
        self.busy: list[list[tuple[float, float]]] = [[] for _ in range(inst.m)]   # sorted (start, finish)
        self.if_free = np.zeros(len(inst.iface_ids))
        self.if_busy: list[list[tuple[float, float]]] = [[] for _ in inst.iface_ids]
        self.start = np.zeros(inst.n)
        self.finish = np.zeros(inst.n)
        self.opt = np.full(inst.n, -1, dtype=np.int64)
        self.comms: list[tuple[int, float, float, int]] = []
        self.usage = np.zeros((inst.m, inst.R))
        self.order: list[int] = []

    # -- helpers ---------------------------------------------------------------------------------------
    def _gap(self, pe: int, ready: float, dur: float) -> float:
        t = ready
        for s, f in self.busy[pe]:
            if t + dur <= s + EPS:
                return t
            if f > t:
                t = f
        return t

    def trial(self, i: int, k: int):
        """Return (start, finish, comm records, interface-clock update) for task ``i`` on option ``k`` or None.

        With ``insertion=True`` transfers on a shared interface may also use idle gaps of that interface; otherwise they
        are appended after the last booked transfer.
        """
        inst = self.inst
        b = int(inst.o_pe[k])
        upd = None                      # interface -> new free time (append mode) or new sorted busy list (gap mode)
        ready = 0.0
        recs = []
        pr = inst.preds[i]
        if len(pr) > 1:
            pr = sorted(pr, key=lambda x: (self.finish[x[0]], x[0]))
        for u, e in pr:
            a = int(inst.o_pe[self.opt[u]])
            c = inst.ctime[e, a, b]
            if c == INF:
                return None
            st = self.finish[u]
            if a != b:
                f = int(inst.cif[e, a, b])
                if self.contention and inst.iface_shared[f]:
                    if upd is None:
                        upd = {}
                    if self.insertion:
                        lst = upd.get(f)
                        if lst is None:
                            lst = upd[f] = list(self.if_busy[f])
                        t = st
                        for bs, bf in lst:
                            if t + c <= bs + EPS:
                                break
                            if bf > t:
                                t = bf
                        st = t
                        j = len(lst)
                        while j > 0 and lst[j - 1][0] > st:
                            j -= 1
                        lst.insert(j, (st, st + c))
                    else:
                        free = upd.get(f, self.if_free[f])
                        if free > st:
                            st = free
                        upd[f] = st + c
                recs.append((e, st, st + c, f))
            ft = st + c
            if ft > ready:
                ready = ft
        dur = inst.o_time[k]
        if inst.pe_excl[b]:
            s = self._gap(b, ready, dur) if self.insertion else max(ready, self.pe_free[b])
        else:
            s = ready
        return s, s + dur, recs, upd

    def place(self, i: int, k: int, trial=None):
        t = trial if trial is not None else self.trial(i, k)
        if t is None:
            raise ValueError(f"no route into task {i} on option {k}")
        s, f, recs, upd = t
        inst = self.inst
        b = int(inst.o_pe[k])
        if upd:
            for fi, v in upd.items():
                if self.insertion:
                    self.if_busy[fi] = v
                else:
                    self.if_free[fi] = v
        if inst.pe_excl[b]:
            self.pe_free[b] = max(self.pe_free[b], f)
            lst = self.busy[b]
            j = len(lst)
            while j > 0 and lst[j - 1][0] > s:
                j -= 1
            lst.insert(j, (s, f))
        self.start[i], self.finish[i], self.opt[i] = s, f, k
        self.comms.extend(recs)
        self.usage[b] += inst.o_res[k]
        self.order.append(i)

    def finish_schedule(self) -> Schedule:
        inst = self.inst
        mk = float(self.finish.max()) if inst.n else 0.0
        placed = self.opt >= 0
        energy = float(inst.o_energy[self.opt[placed]].sum())
        energy += sum(float(inst.cenergy[e, inst.o_pe[self.opt[inst.e_src[e]]], inst.o_pe[self.opt[inst.e_dst[e]]]])
                      for e, *_ in self.comms)
        energy += inst.static_power * mk / 1000.0
        return Schedule(mk, self.start.copy(), self.finish.copy(), list(self.comms), self.usage.copy(),
                        cap_excess(inst, self.usage), energy, list(self.order))


def cap_excess(inst: Instance, usage: np.ndarray) -> float:
    cap = inst.pe_cap
    with np.errstate(divide="ignore", invalid="ignore"):
        over = np.where(np.isfinite(cap), np.maximum(usage / np.where(cap > 0, cap, 1.0) - 1.0, 0.0), 0.0)
        over = np.where((cap == 0) & (usage > 0), 1.0, over)
    tot = float(over.sum())
    return 0.0 if tot < 1e-9 else tot


def upward_rank(inst: Instance, opt: np.ndarray) -> np.ndarray:
    """b-level of the mapped graph: own time + max over successors of (transfer + successor rank)."""
    rank = np.zeros(inst.n)
    pe = inst.o_pe[opt]
    for i in range(inst.n - 1, -1, -1):
        best = 0.0
        for s, e in inst.succs[i]:
            c = inst.ctime[e, pe[i], pe[s]]
            v = c + rank[s]
            if v > best:
                best = v
        rank[i] = inst.o_time[opt[i]] + best
    return rank


def simulate(inst: Instance, opt: np.ndarray, order=None, insertion: bool = False, contention: bool = True) -> Schedule:
    """Schedule the mapping ``opt`` (global option id per task).

    ``order``: None -> upward-rank priority; a sequence of task ids (priority, best first); or a unified list that also
    holds transfers as ``n + edge_id``.
    """
    opt = np.asarray(opt, dtype=np.int64)
    n = inst.n
    if order is not None and any(x >= n for x in order):
        return _simulate_unified(inst, opt, order, contention)
    if order is None:
        rank = upward_rank(inst, opt)
        key = [(-rank[i], i) for i in range(n)]
    else:
        pos = {int(t): j for j, t in enumerate(order)}
        key = [(pos.get(i, n + i), i) for i in range(n)]
    st = Partial(inst, contention, insertion)
    remaining = [len(inst.preds[i]) for i in range(n)]
    heap = [(key[i], i) for i in range(n) if remaining[i] == 0]
    heapq.heapify(heap)
    while heap:
        _, i = heapq.heappop(heap)
        st.place(i, int(opt[i]))
        for s, _e in inst.succs[i]:
            remaining[s] -= 1
            if remaining[s] == 0:
                heapq.heappush(heap, (key[s], s))
    return st.finish_schedule()


def _simulate_unified(inst: Instance, opt: np.ndarray, order, contention: bool) -> Schedule:
    n, E = inst.n, inst.E
    pe = inst.o_pe[opt]
    pos = {int(t): j for j, t in enumerate(order)}
    big = len(order)
    prio = lambda item: (pos.get(item, big + item), item)    # noqa: E731
    start = np.zeros(n); finish = np.zeros(n)
    pe_free = np.zeros(inst.m); if_free = np.zeros(len(inst.iface_ids))
    arrival = np.zeros(n)                       # max arrival time of inputs seen so far
    waiting = [0] * n                           # inputs still outstanding
    comms: list[tuple[int, float, float, int]] = []
    placed_order: list[int] = []
    heap: list = []
    for e in range(E):
        waiting[inst.e_dst[e]] += 1
    for i in range(n):
        if waiting[i] == 0:
            heapq.heappush(heap, (prio(i), i))

    def done(i: int):
        for s, e in inst.succs[i]:
            a, b = int(pe[i]), int(pe[s])
            if a == b:
                arrival[s] = max(arrival[s], finish[i])
                waiting[s] -= 1
                if waiting[s] == 0:
                    heapq.heappush(heap, (prio(s), s))
            else:
                heapq.heappush(heap, (prio(n + e), n + e))

    while heap:
        _, item = heapq.heappop(heap)
        if item < n:
            i = item
            b = int(pe[i])
            s = max(arrival[i], pe_free[b]) if inst.pe_excl[b] else arrival[i]
            f = s + inst.o_time[opt[i]]
            start[i], finish[i] = s, f
            if inst.pe_excl[b]:
                pe_free[b] = f
            placed_order.append(i)
            done(i)
        else:
            e = item - n
            u, v = int(inst.e_src[e]), int(inst.e_dst[e])
            a, b = int(pe[u]), int(pe[v])
            c = inst.ctime[e, a, b]
            if c == INF:
                raise ValueError("no route")
            f_if = int(inst.cif[e, a, b])
            s = finish[u]
            if contention and inst.iface_shared[f_if]:
                s = max(s, if_free[f_if])
                if_free[f_if] = s + c
            comms.append((e, s, s + c, f_if))
            arrival[v] = max(arrival[v], s + c)
            waiting[v] -= 1
            if waiting[v] == 0:
                heapq.heappush(heap, (prio(v), v))
    usage = np.zeros((inst.m, inst.R))
    for i in range(n):
        usage[pe[i]] += inst.o_res[opt[i]]
    mk = float(finish.max()) if n else 0.0
    energy = float(inst.o_energy[opt].sum()) + sum(float(inst.cenergy[e, pe[inst.e_src[e]], pe[inst.e_dst[e]]])
                                                   for e, *_ in comms) + inst.static_power * mk / 1000.0
    return Schedule(mk, start, finish, comms, usage, cap_excess(inst, usage), energy, placed_order)


def deadline_met(inst: Instance, sch: Schedule) -> bool:
    return inst.deadline is None or sch.makespan <= inst.deadline * (1 + 1e-9)


def feasible(inst: Instance, sch: Schedule) -> bool:
    return sch.capacity_ok and deadline_met(inst, sch)
