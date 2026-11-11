"""CP-SAT model for joint mapping + scheduling (same semantics as the MILP in solvers/ilp/model.py).

Why: the survey (docs/survey/partitioning_survey.md §3.1, §7.1) found CP with interval variables beats big-M
MILP on disjunctive scheduling as instances grow (Ku & Beck 2016; Naderi et al. 2023; RunSoC 2.0, 2026).

Model:
- one optional interval per legal (task, PE, implementation) option sharing the task's start/end variables,
  with exactly one option present;
- `AddNoOverlap` per exclusive PE; spatial PEs only constrain resources;
- precedence with PE-pair communication delay, enforced only when the endpoints sit on that PE pair;
- capacity, deadline, per-task deadline, period/II, energy, power and $ constraints as in the MILP;
- objective = the same normalised weighted sum as the simulator.

With `contention=False` it matches the MILP: exact *without* shared-bus contention. With `contention=True`
every cross-PE transfer is an optional interval on its interface (present iff the endpoints sit on that PE
pair); transfers on a `shared` interface may not overlap, the consumer starts after its transfer ends, and in
streaming mode the bus busy time is bounded by the period. Transfer order on the bus is free (insertion-like),
so the simulator replays these labels with `insertion=True`. The returned mapping is always re-simulated. Times are integer microseconds, energies integer
nanojoules, resources ×1000: rounding can move results by about 1 µs per task.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from estimate.annotate import CostView, Option, Problem
from schemas.result import Violation

T_SCALE = 1000.0  # ms -> µs
E_SCALE = 1e6  # mJ -> nJ
R_SCALE = 1000.0  # resources
U_SCALE = 100.0  # USD -> cents


@dataclass
class CpOutcome:
    status: str  # optimal | feasible | infeasible | time_limit | error
    mapping: dict[str, Option] | None
    starts: dict[str, float] | None
    makespan: float | None
    objective: float | None
    bound: float | None
    runtime_s: float
    relaxation: list[Violation] | None = None
    notes: list[str] = field(default_factory=list)
    transfer_starts: dict[str, float] = field(default_factory=dict)  # "src->dst" -> start (contention model)

    def priority(self, problem: Problem) -> list[str] | None:
        """Tasks (and, with contention, transfers) in start order: the list that replays this schedule."""
        if self.starts is None:
            return None
        topo = {n: i for i, n in enumerate(problem.order)}
        items = [(t, 0, topo[n], n) for n, t in self.starts.items()]
        items += [(t, 1, topo[k.split("->")[1]], k) for k, t in self.transfer_starts.items()]
        return [k for *_, k in sorted(items)]

    @property
    def gap(self) -> float | None:
        if self.objective is None or self.bound is None:
            return None
        return abs(self.objective - self.bound) / max(abs(self.objective), 1e-12)


def _t(x: float) -> int:
    """ms -> integer µs; any positive duration becomes >= 1 µs (zero-length intervals would not conflict in
    NoOverlap, letting sub-µs tasks share a start time on one CPU)."""
    v = int(round(x * T_SCALE))
    return max(v, 1) if x > 0 else v


def solve_cp(
    problem: Problem,
    costs: CostView | None = None,
    time_limit_s: float = 60.0,
    workers: int = 8,
    seed: int = 0,
    hint: dict[str, Option] | None = None,
    hint_starts: dict[str, float] | None = None,
    log: bool = False,
    canonical: bool = False,
    contention: bool = False,
) -> CpOutcome:
    """Solve to optimality (or until the time limit). `hint`/`hint_starts` warm-start the search.

    Warning: with `hint_starts` on a model whose mapping is free, OR-Tools 9.15 returned a false OPTIMAL (presolve
    + start hints; DECISIONS 2026-11-06). Hint the mapping only unless the mapping is fixed.

    `canonical=True` (latency-only objectives): after an optimal makespan is proven, fix it and re-solve for the
    lexicographically earliest schedule (Σ starts, then lowest option indices), so equal optima map to one label.
    """
    costs = costs or problem.nominal()
    t0 = time.perf_counter()
    if any(not problem.options[n] for n in problem.order):
        return CpOutcome("infeasible", None, None, None, None, None, time.perf_counter() - t0,
                         notes=["some node has no legal option (see lower-bound structural diagnostics)"])
    m = cp_model.CpModel()
    nodes, opts = problem.order, problem.options
    dur = {(n, k): _t(costs.time[(n, o.impl)]) for n in nodes for k, o in enumerate(opts[n])}

    def pes_of(n: str) -> list[str]:
        return sorted({o.pe for o in opts[n]}, key=problem.pes.index)

    comm_max = 0
    for a, b in problem.edges:
        nb = costs.bytes[(a, b)]
        cs = [c.time_ms for p in pes_of(a) for q in pes_of(b) if p != q and (c := problem.comm(p, q, nb))]
        comm_max += _t(max(cs, default=0.0))
    horizon = sum(max(dur[(n, k)] for k in range(len(opts[n]))) for n in nodes) + comm_max + 1

    s = {n: m.NewIntVar(0, horizon, f"s_{i}") for i, n in enumerate(nodes)}
    e = {n: m.NewIntVar(0, horizon, f"e_{i}") for i, n in enumerate(nodes)}
    x = {(n, k): m.NewBoolVar(f"x_{i}_{k}") for i, n in enumerate(nodes) for k in range(len(opts[n]))}
    iv = {(n, k): m.NewOptionalIntervalVar(s[n], dur[(n, k)], e[n], x[(n, k)], f"iv_{i}_{k}")
          for i, n in enumerate(nodes) for k in range(len(opts[n]))}
    for n in nodes:
        m.AddExactlyOne(x[(n, k)] for k in range(len(opts[n])))
    y: dict[tuple[str, str], cp_model.IntVar] = {}
    for i, n in enumerate(nodes):
        for pe in pes_of(n):
            y[(n, pe)] = m.NewBoolVar(f"y_{i}_{pe}")
            m.Add(y[(n, pe)] == sum(x[(n, k)] for k, o in enumerate(opts[n]) if o.pe == pe))

    # precedence + communication
    comm_energy_terms = []
    bus: dict[str, list[tuple[cp_model.IntervalVar, int, cp_model.IntVar]]] = {}
    tvars: dict[tuple[str, str], tuple[cp_model.IntVar, list[cp_model.IntVar]]] = {}
    for ei, (a, b) in enumerate(problem.edges):
        nb = costs.bytes[(a, b)]
        m.Add(s[b] >= e[a])
        if contention:
            ts = m.NewIntVar(0, horizon, f"ts_{ei}")
            te = m.NewIntVar(0, horizon, f"te_{ei}")
            tvars[(a, b)] = (ts, [])
        for p in pes_of(a):
            for q in pes_of(b):
                if p == q:
                    continue
                c = problem.comm(p, q, nb)
                if c is None:
                    m.AddBoolOr([y[(a, p)].Not(), y[(b, q)].Not()])
                    continue
                if contention:
                    z = m.NewBoolVar(f"z_{ei}_{p}_{q}")  # z <-> (a on p and b on q)
                    m.AddBoolAnd([y[(a, p)], y[(b, q)]]).OnlyEnforceIf(z)
                    m.AddBoolOr([y[(a, p)].Not(), y[(b, q)].Not(), z])
                    d = _t(c.time_ms)
                    itv = m.NewOptionalIntervalVar(ts, d, te, z, f"tr_{ei}_{p}_{q}")
                    tvars[(a, b)][1].append(z)
                    m.Add(ts >= e[a]).OnlyEnforceIf(z)
                    m.Add(s[b] >= te).OnlyEnforceIf(z)
                    if c.interface is not None and problem.interface(c.interface).shared:
                        bus.setdefault(c.interface, []).append((itv, d, z))
                    if c.energy_mj > 0:
                        comm_energy_terms.append(int(round(c.energy_mj * E_SCALE)) * z)
                    continue
                if _t(c.time_ms) > 0:
                    m.Add(s[b] >= e[a] + _t(c.time_ms)).OnlyEnforceIf([y[(a, p)], y[(b, q)]])
                if c.energy_mj > 0:
                    z = m.NewBoolVar(f"z_{ei}_{p}_{q}")
                    m.AddBoolOr([y[(a, p)].Not(), y[(b, q)].Not(), z])
                    comm_energy_terms.append(int(round(c.energy_mj * E_SCALE)) * z)
    for iface, items in bus.items():
        if len(items) > 1:
            m.AddNoOverlap([itv for itv, _, _ in items])

    # exclusive PEs
    for pe in problem.pes:
        if problem.is_exclusive(pe):
            ivs = [iv[(n, k)] for n in nodes for k, o in enumerate(opts[n]) if o.pe == pe]
            if len(ivs) > 1:
                m.AddNoOverlap(ivs)

    # capacities
    for pe in problem.pes:
        for r, cap in problem.platform.pe(pe).capacity.items():
            terms = [int(round(costs.res[(n, o.impl)].get(r, 0.0) * R_SCALE)) * x[(n, k)]
                     for n in nodes for k, o in enumerate(opts[n]) if o.pe == pe]
            if terms:
                m.Add(sum(terms) <= int(math.floor(cap * R_SCALE + 1e-6)))

    C = m.NewIntVar(0, horizon, "C")
    m.AddMaxEquality(C, list(e.values()))
    if problem.deadline_ms is not None:
        m.Add(C <= _t(problem.deadline_ms))
    for n, d in problem.task_deadlines.items():
        m.Add(e[n] <= _t(d))
    if problem.period_ms is not None and problem.arrival == "periodic":
        m.Add(C <= _t(problem.period_ms))
    elif problem.period_ms is not None and problem.arrival == "streaming":
        per = _t(problem.period_ms)
        for pe in problem.pes:
            busy = [dur[(n, k)] * x[(n, k)] for n in nodes for k, o in enumerate(opts[n]) if o.pe == pe]
            if problem.is_exclusive(pe) and busy:
                m.Add(sum(busy) <= per)
        for n in nodes:
            for k, o in enumerate(opts[n]):
                if not problem.is_exclusive(o.pe) and _t(costs.ii[(n, o.impl)]) > per:
                    m.Add(x[(n, k)] == 0)
        for items in bus.values():  # shared-bus busy time per frame (the simulator's interface term of the II)
            m.Add(sum(d * z for _, d, z in items) <= per)

    # energy (nJ): task + communication + static power of all allowed PEs over the makespan
    static_coef = int(round(problem.static_power_mw))  # mW · µs = nJ
    E = (sum(int(round(costs.energy[(n, o.impl)] * E_SCALE)) * x[(n, k)] for n in nodes for k, o in enumerate(opts[n]))
         + sum(comm_energy_terms) + static_coef * C)
    if problem.energy_mj_max is not None:
        m.Add(E <= int(math.floor(problem.energy_mj_max * E_SCALE)))
    pmax = min([v for v in (problem.power_mw_max, problem.platform.budgets.get("power_mw")) if v is not None],
               default=None)
    if pmax is not None:
        if problem.period_ms:
            m.Add(E <= int(math.floor(pmax * problem.period_ms * 1e3)))  # mW·ms = µJ = 1e3 nJ
        else:
            m.Add(E <= int(round(pmax)) * C)  # E[nJ] ≤ P[mW]·C[µs]
    umax = min([v for v in (problem.usd_max, problem.platform.budgets.get("usd")) if v is not None], default=None)
    usd = 0
    if umax is not None or "usd" in problem.weights:
        u = {pe: m.NewBoolVar(f"u_{i}") for i, pe in enumerate(problem.pes)}
        for n in nodes:
            for pe in pes_of(n):
                m.AddImplication(y[(n, pe)], u[pe])
        usd = (sum(int(round(problem.platform.pe(pe).usd * U_SCALE)) * u[pe] for pe in problem.pes)
               + sum(int(round(costs.res[(n, o.impl)].get("usd", 0.0) * U_SCALE)) * x[(n, k)]
                     for n in nodes for k, o in enumerate(opts[n])))
        if umax is not None:
            m.Add(usd <= int(math.floor(umax * U_SCALE)))

    norms = problem.norms
    scale = {"latency": 1.0 / T_SCALE, "energy": 1.0 / E_SCALE, "usd": 1.0 / U_SCALE}
    term = {"latency": C, "energy": E, "usd": usd}
    m.Minimize(sum(w * scale[k] / norms[k] * term[k] for k, w in problem.weights.items()))

    if hint:
        for n in nodes:
            for k, o in enumerate(opts[n]):
                m.AddHint(x[(n, k)], o == hint.get(n))
        for n, st in (hint_starts or {}).items():
            m.AddHint(s[n], _t(st))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = log
    st = solver.Solve(m)
    name = {cp_model.OPTIMAL: "optimal", cp_model.FEASIBLE: "feasible", cp_model.INFEASIBLE: "infeasible",
            cp_model.UNKNOWN: "time_limit"}.get(st, "error")
    rt = time.perf_counter() - t0
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return CpOutcome(name, None, None, None, None, None, rt)
    obj, bound = solver.ObjectiveValue(), solver.BestObjectiveBound()
    notes: list[str] = []
    def transfers() -> dict[str, float]:
        return {f"{a}->{b}": solver.Value(ts) / T_SCALE for (a, b), (ts, zs) in tvars.items()
                if any(solver.Value(z) for z in zs)}

    first = ({n: next(o for k, o in enumerate(opts[n]) if solver.Value(x[(n, k)])) for n in nodes},
             {n: solver.Value(s[n]) / T_SCALE for n in nodes}, solver.Value(C) / T_SCALE, transfers())
    if canonical and st == cp_model.OPTIMAL and set(problem.weights) == {"latency"}:
        c_opt = solver.Value(C)
        for n in nodes:
            m.AddHint(s[n], solver.Value(s[n]))
        m.Add(C <= c_opt)
        kmax = max(len(opts[n]) for n in nodes)
        m.Minimize(sum(s.values()) * kmax + sum(k * x[(n, k)] for n in nodes for k in range(len(opts[n]))))
        solver.parameters.max_time_in_seconds = max(1.0, time_limit_s - rt)
        st2 = solver.Solve(m)
        if st2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):  # keep the proven-optimal first solution
            notes.append("canonicalisation found no solution in time; first optimal solution kept")
            return CpOutcome(name, first[0], first[1], first[2], obj, bound, time.perf_counter() - t0, notes=notes,
                             transfer_starts=first[3])
        notes.append("canonicalised" if st2 == cp_model.OPTIMAL else "canonicalisation not proven (time limit)")
    mapping = {n: next(o for k, o in enumerate(opts[n]) if solver.Value(x[(n, k)])) for n in nodes}
    starts = {n: solver.Value(s[n]) / T_SCALE for n in nodes}
    return CpOutcome(name, mapping, starts, solver.Value(C) / T_SCALE, obj, bound, time.perf_counter() - t0,
                     notes=notes, transfer_starts=transfers())
