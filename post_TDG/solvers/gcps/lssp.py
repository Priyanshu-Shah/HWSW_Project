"""LSSP: list scheduling with static priorities, exactly as needed to reproduce Table 1 of Zheng et al. 2021.

Rules (docs/survey/slices/07_gcps_reproduction.md §4; three of them are not stated in the paper but are required to
reproduce its Table 1 cell for cell):
1. priority = bottom level using the partition's execution times and the ORIGINAL edge costs (even same-side ones);
2. the ready task with the highest priority goes next; ties -> lower task index;
3. its incoming edges are placed on the single append-only bus in ascending order of predecessor finish time:
   start = max(pred finish, bus free); the edge costs c if exactly one end is HW, else 0 — zero-cost edges still
   wait for the bus;
4. SW tasks also wait for the CPU (sequential, no gap filling); HW tasks start as soon as their data has arrived.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from solvers.gcps.instance import BinInstance


@dataclass
class LsspSchedule:
    t_sl: float
    start: np.ndarray
    finish: np.ndarray
    bus: list[tuple[int, int, float, float]]  # (i, j, start, finish)


def _adjacency(inst: BinInstance) -> tuple[list[list[tuple[int, float]]], list[list[tuple[int, float]]]]:
    cache = getattr(inst, "_adj_cache", None)
    if cache is None:
        preds: list[list[tuple[int, float]]] = [[] for _ in range(inst.n)]
        succs: list[list[tuple[int, float]]] = [[] for _ in range(inst.n)]
        for i, j, c in inst.edges:
            preds[j].append((i, c))
            succs[i].append((j, c))
        cache = (preds, succs)
        object.__setattr__(inst, "_adj_cache", cache)
    return cache


def lssp(inst: BinInstance, y: np.ndarray, detail: bool = False) -> float | LsspSchedule:
    """Schedule length for partition y (1 = HW, 0 = SW)."""
    n = inst.n
    yl = [bool(v) for v in y]
    ts, th = inst.ts.tolist(), inst.th.tolist()
    exe = [th[i] if yl[i] else ts[i] for i in range(n)]
    preds, succs = _adjacency(inst)
    pri = [0.0] * n
    for i in range(n - 1, -1, -1):  # tasks are in topological order
        pri[i] = exe[i] + max((c + pri[j] for j, c in succs[i]), default=0.0)
    remaining = [len(p) for p in preds]
    heap = [(-pri[i], i) for i in range(n) if remaining[i] == 0]
    heapq.heapify(heap)
    start, finish = [0.0] * n, [0.0] * n
    bus_free = cpu_free = 0.0
    bus: list[tuple[int, int, float, float]] = []
    while heap:
        _, j = heapq.heappop(heap)  # highest priority, ties -> lower index
        t_ready = 0.0
        for i, c in sorted(preds[j], key=lambda e: (finish[e[0]], e[0])):
            st = finish[i] if finish[i] > bus_free else bus_free
            bus_free = st + (c if yl[i] != yl[j] else 0.0)
            if detail:
                bus.append((i, j, st, bus_free))
            if bus_free > t_ready:
                t_ready = bus_free
        st = t_ready if yl[j] else max(t_ready, cpu_free)
        start[j], finish[j] = st, st + exe[j]
        if not yl[j]:
            cpu_free = finish[j]
        for k, _ in succs[j]:
            remaining[k] -= 1
            if remaining[k] == 0:
                heapq.heappush(heap, (-pri[k], k))
    t_sl = max(finish) if n else 0.0
    if detail:
        return LsspSchedule(t_sl, np.array(start), np.array(finish), bus)
    return t_sl


def greedy_area(score: np.ndarray, area: np.ndarray, cap: float) -> np.ndarray:
    """GCPS's cut-line: label tasks HW in decreasing score until the first task that would exceed the area budget;
    that task and all after it are SW (stop-at-first-overflow, p. 334)."""
    y = np.zeros(len(score), dtype=np.int8)
    used = 0.0
    for i in np.argsort(-score, kind="stable"):
        if used + area[i] > cap + 1e-9:
            break
        y[i] = 1
        used += area[i]
    return y
