"""Mapping relaxation of the joint mapping + scheduling problem: a lower bound on the optimal makespan.

Keeps the mapping decisions and drops the schedule's ordering. With x[t, k] = 1 iff task t uses option k:
  - one option per task; capacity per (PE, resource);
  - load: C >= sum of task times on each exclusive PE;
  - paths: f[b] >= f[a] + comm(a, b) + d[b], C >= f[t], where comm(a, b) >= c(p, q) * (X[a, p] + X[b, q] - 1)
    links the edge's transfer time to the PE pair (exact for an integer mapping);
  - bus load: C >= sum of transfer times routed over each shared interface;
  - release/tail cuts per exclusive PE and per shared bus (`energetic`): C >= r + load of the activities that
    cannot start before r + their minimum tail;
  - deadlines: C <= deadline_ms, f[t] <= the task's deadline.
Every feasible schedule satisfies these constraints, so the optimum of this model (or its LP relaxation, or the dual
bound of a time-limited solve) is a valid lower bound on the makespan, with or without bus contention. It is also
the master problem of the logic-based Benders decomposition (`solvers/exact/lbbd.py`).
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import pulp

from estimate.annotate import CostView, Option, Problem


@dataclass
class RelaxBound:
    status: str  # optimal | bound (time limit, dual bound only) | infeasible | error
    bound_ms: float | None  # valid lower bound on the optimal makespan; None if infeasible
    mapping: dict[str, Option] | None  # the relaxation's best mapping (integer model only)
    runtime_s: float
    notes: list[str] = field(default_factory=list)


class RelaxationModel:
    """The relaxation as a PuLP model, kept so cuts can be added between solves (Benders master)."""

    def __init__(self, problem: Problem, integer: bool = True, bus: bool = True, energetic: bool = True,
                 costs: CostView | None = None, deadlines: bool = True):
        """Minimises the makespan C whatever the problem's objective weights: the result bounds the makespan.
        `costs` selects the cost view (default nominal); `deadlines=False` drops the deadline constraints, so the
        optimum is the smallest makespan any mapping could reach (used to report how far a deadline is off)."""
        if any(not problem.options[t] for t in problem.order):
            raise ValueError("some task has no legal option")
        self.problem = p = problem
        v = costs or p.nominal()
        cat = pulp.LpBinary if integer else pulp.LpContinuous
        lp = pulp.LpProblem("mapping_relaxation", pulp.LpMinimize)
        idx = {t: i for i, t in enumerate(p.order)}
        x = {(t, k): lp.add_variable(f"x_{idx[t]}_{k}", lowBound=0, upBound=1, cat=cat) for t in p.order for k in range(len(p.options[t]))}
        C = lp.add_variable("C", lowBound=0)
        lp += C
        for t in p.order:
            lp += pulp.lpSum(x[t, k] for k in range(len(p.options[t]))) == 1
        X = {(t, pe): pulp.lpSum(x[t, k] for k, o in enumerate(p.options[t]) if o.pe == pe)
             for t in p.order for pe in {o.pe for o in p.options[t]}}
        dur = {t: pulp.lpSum(v.time[(t, o.impl)] * x[t, k] for k, o in enumerate(p.options[t])) for t in p.order}
        # mapping-independent earliest start (head) and minimum remaining time after a task (tail)
        dmin = {t: min(v.time[(t, o.impl)] for o in p.options[t]) for t in p.order}
        head: dict[str, float] = {}
        for t in p.order:
            head[t] = max((head[q] + dmin[q] for q in p.preds[t]), default=0.0)
        tail: dict[str, float] = {}
        for t in reversed(p.order):
            tail[t] = max((dmin[s] + tail[s] for s in p.succs[t]), default=0.0)
        for pe in p.pes:
            on_pe = [(t, k, o) for t in p.order for k, o in enumerate(p.options[t]) if o.pe == pe]
            for r, cap in p.platform.pe(pe).capacity.items():
                use = [(v.res[(t, o.impl)].get(r, 0.0), x[t, k]) for t, k, o in on_pe]
                if any(u > 0 for u, _ in use):
                    lp += pulp.lpSum(u * xv for u, xv in use) <= cap
            if p.is_exclusive(pe) and on_pe:
                lp += C >= pulp.lpSum(v.time[(t, o.impl)] * x[t, k] for t, k, o in on_pe)
                if energetic:
                    items = [(t, head[t], tail[t],
                              pulp.lpSum(v.time[(t, o.impl)] * x[t, k] for tt, k, o in on_pe if tt == t))
                             for t in dict.fromkeys(t for t, _, _ in on_pe)]
                    self._release_tail_cuts(lp, C, items)
        f = {t: lp.add_variable(f"f_{idx[t]}", lowBound=0) for t in p.order}
        bus_terms: dict[str, list] = {}
        for a, b in p.edges:
            nb = v.bytes[(a, b)]
            tc = lp.add_variable(f"c_{idx[a]}_{idx[b]}", lowBound=0)
            per_iface: dict[str, pulp.LpVariable] = {}
            for pa, pb in itertools.product({o.pe for o in p.options[a]}, {o.pe for o in p.options[b]}):
                if pa == pb:
                    continue
                c = p.comm(pa, pb, nb)
                if c is None:
                    lp += X[a, pa] + X[b, pb] <= 1
                    continue
                if c.time_ms <= 0:
                    continue
                lp += tc >= c.time_ms * (X[a, pa] + X[b, pb] - 1)
                if bus and c.interface is not None and p.interface(c.interface).shared:
                    if c.interface not in per_iface:
                        per_iface[c.interface] = lp.add_variable(f"b_{idx[a]}_{idx[b]}_{len(per_iface)}", lowBound=0)
                        # a transfer cannot start before its producer can finish, and its consumer runs after it
                        bus_terms.setdefault(c.interface, []).append(
                            ((a, b), head[a] + dmin[a], dmin[b] + tail[b], per_iface[c.interface]))
                    lp += per_iface[c.interface] >= c.time_ms * (X[a, pa] + X[b, pb] - 1)
            lp += f[b] >= f[a] + tc + dur[b]
        for t in p.order:
            lp += f[t] >= dur[t]
            lp += C >= f[t]
            if deadlines and t in p.task_deadlines:
                lp += f[t] <= p.task_deadlines[t]
        for terms in bus_terms.values():
            lp += C >= pulp.lpSum(var for *_, var in terms)
            if energetic:  # the shared bus is one more exclusive resource for release/tail reasoning
                self._release_tail_cuts(lp, C, terms)
        if deadlines and p.deadline_ms is not None:
            lp += C <= p.deadline_ms
        self.lp, self.x, self.C, self.integer = lp, x, C, integer
        self.n_cuts = 0

    @staticmethod
    def _release_tail_cuts(lp, C, items: list[tuple]) -> None:
        """Release/tail cuts on one exclusive resource (a PE or a shared bus). `items` = (key, head, tail, expr):
        an activity that cannot start before `head`, needs at least `tail` after it ends, and occupies the resource
        for `expr` (a linear expression, 0 when the mapping does not put it there). Activities that cannot start
        before r all run after r, and the last of them still needs its tail: C >= r + load(head >= r) + min tail
        over that set. The same with tails, and with both thresholds from one activity. Minima over the candidate
        set are <= the minima over the activities actually on the resource, so every cut is valid."""
        items = sorted(items, key=lambda it: (it[1], it[2]))
        seen: set[tuple] = set()
        for _, hu, tu, _ in items:
            for kind, group, extra in (
                ("r", [it for it in items if it[1] >= hu], lambda g: hu + min(it[2] for it in g)),
                ("q", [it for it in items if it[2] >= tu], lambda g: tu + min(it[1] for it in g)),
                ("rq", [it for it in items if it[1] >= hu and it[2] >= tu], lambda g: hu + tu),
            ):
                key = (kind, tuple(it[0] for it in group))
                if len(group) < 2 or key in seen:
                    continue
                seen.add(key)
                if extra(group) > 0:
                    lp += C >= extra(group) + pulp.lpSum(it[3] for it in group)

    def _changed(self, mapping: dict[str, Option]) -> pulp.LpAffineExpression:
        p = self.problem
        return pulp.lpSum(1 - self.x[t, p.options[t].index(mapping[t])] for t in p.order)

    def add_nogood(self, mapping: dict[str, Option], value: float, floor: float) -> None:
        """C >= value unless at least one task changes option; `floor` must be a valid global lower bound."""
        self.lp += self.C >= value - (value - floor) * self._changed(mapping)
        self.n_cuts += 1

    def add_exclusion(self, mapping: dict[str, Option]) -> None:
        """Forbid this exact mapping (e.g. it cannot meet the deadline)."""
        self.lp += self._changed(mapping) >= 1
        self.n_cuts += 1

    def solve(self, time_limit_s: float = 10.0, seed: int = 0, msg: bool = False) -> RelaxBound:
        t0 = time.perf_counter()
        # tight feasibility tolerances: at HiGHS defaults, presolve cut off the true optimum of this model on Tier A
        # seed 195 (reported 232.37 where 227.40 is feasible; DECISIONS 2026-11-07), i.e. an invalid "bound"
        self.lp.solve(pulp.HiGHS(msg=msg, timeLimit=time_limit_s, threads=1, random_seed=seed,
                                 mip_feasibility_tolerance=1e-9, primal_feasibility_tolerance=1e-9,
                                 dual_feasibility_tolerance=1e-9))
        rt = time.perf_counter() - t0
        st = self.lp.sol_status
        if self.lp.status == pulp.LpStatusInfeasible or st == pulp.LpSolutionInfeasible:
            return RelaxBound("infeasible", None, None, rt, ["relaxation infeasible: no mapping can meet the constraints"])
        if st not in (pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible, pulp.LpSolutionNoSolutionFound):
            return RelaxBound("error", None, None, rt, [f"HiGHS status {pulp.LpStatus[self.lp.status]}"])
        optimal = st == pulp.LpSolutionOptimal
        if self.integer:  # the dual bound, not the primal value: HiGHS stops within its relative gap tolerance
            bound = float(self.lp.solverModel.getInfo().mip_dual_bound)
        else:
            bound = float(self.C.value()) if optimal else None
        mapping = None
        if self.integer and st in (pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible):
            p = self.problem
            mapping = {t: p.options[t][max(range(len(p.options[t])), key=lambda k: self.x[t, k].value() or 0.0)]
                       for t in p.order}
        return RelaxBound("optimal" if optimal else "bound", bound, mapping, rt)


def relaxation_bound(problem: Problem, integer: bool = True, time_limit_s: float = 10.0, bus: bool = True,
                     energetic: bool = True, seed: int = 0, costs: CostView | None = None,
                     deadlines: bool = True) -> RelaxBound:
    """Lower bound on the optimal makespan from the mapping relaxation (LP if `integer=False`)."""
    return RelaxationModel(problem, integer=integer, bus=bus, energetic=energetic, costs=costs,
                           deadlines=deadlines).solve(time_limit_s, seed)
