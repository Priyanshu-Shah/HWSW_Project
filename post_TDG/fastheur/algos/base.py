"""Anytime bookkeeping shared by every algorithm.

Protocol
--------
``solve(inst, fast, seed, budget_s) -> Result``. The algorithm runs for at most ``budget_s`` wall-clock seconds and
reports every improvement of its best *feasible* (capacity-respecting) solution in ``Result.trace`` as
``(elapsed_s, makespan_ms)``. The benchmark reads the best value known at 10 ms / 100 ms / 1 s / 10 s from the trace, so
one 10 s run of a budget-independent algorithm yields all four budgets.

What is *not* charged: loading the instance and building the flat arrays of ``Fast`` (shared by every method),
and numba compilation (a warm-up call is made before timing).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..fast import Fast
from ..instance import Instance
from .common import Plan


@dataclass
class Result:
    method: str
    plan: Plan | None
    makespan: float
    trace: list[tuple[float, float]]
    evals: int
    elapsed: float
    extra: dict = field(default_factory=dict)

    def best_at(self, t: float) -> float:
        """Best feasible makespan known at time ``t`` (inf if none)."""
        best = float("inf")
        for tt, v in self.trace:
            if tt <= t and v < best:
                best = v
        return best


class Tracker:
    def __init__(self, inst: Instance, fast: Fast, budget_s: float):
        self.inst, self.fast, self.budget = inst, fast, budget_s
        self.t0 = time.perf_counter()
        self.best_mk = float("inf")
        self.best_plan: Plan | None = None
        self.trace: list[tuple[float, float]] = []
        self.e0 = fast.n_evals
        # Every method starts from the same free fallback (all tasks on their fastest CPU option), so a result exists
        # at any budget. It is charged to the method's clock like everything else.
        self.fallback_mk = fast.allsw_mk
        self.offer(fast.allsw_mk, fast.allsw_ex, fast.allsw_opt, at=0.0)

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def left(self) -> float:
        return self.budget - self.elapsed()

    def expired(self) -> bool:
        return self.elapsed() >= self.budget

    def offer(self, mk: float, ex: float, opt: np.ndarray, order=None, insertion: bool = False, at=None) -> bool:
        if ex == 0.0 and mk < self.best_mk * (1 - 1e-12):
            self.best_mk = float(mk)
            self.best_plan = Plan(np.array(opt, dtype=np.int64), None if order is None else list(order), insertion)
            self.trace.append((self.elapsed() if at is None else at, self.best_mk))
            return True
        return False

    def offer_plan(self, plan: Plan, mk: float | None = None) -> bool:
        """Offer a constructive plan. ``mk`` = the makespan the algorithm itself computed; if omitted the plan is
        re-simulated with the (slow) reference simulator. Either way the benchmark re-verifies the final plan."""
        if mk is None:
            s = plan.schedule(self.inst)
            return self.offer(s.makespan, s.cap_excess, plan.opt, plan.order, plan.insertion)
        return self.offer(mk, self.fast.excess(plan.opt), plan.opt, plan.order, plan.insertion)

    def result(self, method: str, **extra) -> Result:
        return Result(method, self.best_plan, self.best_mk, list(self.trace), self.fast.n_evals - self.e0,
                      self.elapsed(), extra)


# -----------------------------------------------------------------------------------------------------------
def all_sw_opt(inst: Instance) -> np.ndarray:
    """Every task on its fastest option of a non-FPGA-like (capacity-free) PE; the trivial always-feasible start."""
    out = np.empty(inst.n, dtype=np.int64)
    for i in range(inst.n):
        free = [k for k in inst.task_opts[i] if np.all(np.isinf(inst.pe_cap[inst.o_pe[k]]))]
        pool = free or inst.task_opts[i]
        out[i] = min(pool, key=lambda k: (inst.o_time[k], k))
    return out


def repair_capacity(inst: Instance, opt: np.ndarray, rng: np.random.Generator, max_iter: int = 10_000) -> np.ndarray:
    """Make ``opt`` capacity-feasible: while a PE is over capacity, move a random task on it to an option that uses
    fewer resources on the overflowing dimension (preferring other PEs). Returns the repaired copy."""
    opt = opt.copy()
    usage = usage_of(inst, opt)
    cap = inst.pe_cap * (1 + 1e-9)
    for _ in range(max_iter):
        over = usage > cap
        if not over.any():
            return opt
        pe = int(rng.choice(np.where(over.any(axis=1))[0]))
        dims = np.where(over[pe])[0]
        cand = [i for i in range(inst.n) if inst.o_pe[opt[i]] == pe and inst.o_res[opt[i], dims].sum() > 0]
        rng.shuffle(cand)
        moved = False
        for i in cand:
            cur = int(opt[i])
            better = [k for k in inst.task_opts[i]
                      if k != cur and (inst.o_pe[k] != pe or inst.o_res[k, dims].sum() < inst.o_res[cur, dims].sum())]
            if better:
                k = int(better[int(rng.integers(len(better)))])
                usage[pe] -= inst.o_res[cur]
                usage[inst.o_pe[k]] += inst.o_res[k]
                opt[i] = k
                moved = True
                break
        if not moved:
            return opt
    return opt


def random_opt(inst: Instance, rng: np.random.Generator) -> np.ndarray:
    return np.array([inst.task_opts[i][int(rng.integers(len(inst.task_opts[i])))] for i in range(inst.n)],
                    dtype=np.int64)


def random_feasible(inst: Instance, rng: np.random.Generator) -> np.ndarray:
    return repair_capacity(inst, random_opt(inst, rng), rng)


def hw_area(inst: Instance, usage: np.ndarray) -> float:
    """Second objective for NSGA-II: total utilisation of capacity-limited PEs (sum over PEs of the worst
    resource fraction)."""
    tot = 0.0
    for p in range(inst.m):
        fr = [usage[p, r] / inst.pe_cap[p, r] for r in range(inst.R) if np.isfinite(inst.pe_cap[p, r])
              and inst.pe_cap[p, r] > 0]
        if fr:
            tot += max(fr)
    return tot


def usage_of(inst: Instance, opt: np.ndarray) -> np.ndarray:
    u = np.zeros((inst.m, inst.R))
    np.add.at(u, inst.o_pe[opt], inst.o_res[opt])
    return u
