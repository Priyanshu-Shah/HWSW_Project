"""Shared pieces for the algorithms: Plan, capacity-aware option choice, mean costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..instance import INF, Instance
from ..sim import Partial, Schedule, mean_comm_time, simulate


@dataclass
class Plan:
    """A solver's answer: option per task, optional list order, and whether insertion scheduling is used."""
    opt: np.ndarray
    order: list | None = None
    insertion: bool = False
    info: dict | None = None

    def schedule(self, inst: Instance, contention: bool = True) -> Schedule:
        return simulate(inst, self.opt, self.order, self.insertion, contention)


def fits(inst: Instance, usage: np.ndarray, k: int) -> bool:
    pe = int(inst.o_pe[k])
    return bool(np.all(usage[pe] + inst.o_res[k] <= inst.pe_cap[pe] * (1 + 1e-9)))


def mean_task_time(inst: Instance) -> np.ndarray:
    return np.array([np.mean(inst.o_time[inst.task_opts[i]]) for i in range(inst.n)])


def mean_edge_comm(inst: Instance) -> np.ndarray:
    return np.array([mean_comm_time(inst, e) for e in range(inst.E)])


def rank_u(inst: Instance, w: np.ndarray, c: np.ndarray) -> np.ndarray:
    r = np.zeros(inst.n)
    for i in range(inst.n - 1, -1, -1):
        r[i] = w[i] + max((c[e] + r[s] for s, e in inst.succs[i]), default=0.0)
    return r
