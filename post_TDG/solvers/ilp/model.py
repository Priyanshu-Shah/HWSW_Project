"""Exact MILP for joint mapping + scheduling (no bus-contention model).

Decision variables
  x[i,k]  1 if node i uses legal option k = (PE, implementation)
  s[i]    start time of node i (ms);  f[i] = s[i] + sum_k x[i,k] * t[i,k]
  cm[e]   communication delay on edge e = (i, j): >= comm(p, q) whenever i is on p and j on q != p
  o[i,j]  order of two independent nodes that may share an exclusive PE (big-M disjunction)
  C       makespan;  E  energy;  u[p]  PE p used ($ cost)

The model is exact for the simulator's semantics *without* contention on shared interfaces and without the
interface term of the initiation interval; the returned mapping is always re-evaluated by the simulator.
Formulation in the style of Niemann & Marwedel (1997) with PE-pair-dependent communication.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import pulp

from estimate.annotate import CostView, Option, Problem
from schemas.result import Violation
from sim.evaluate import violation

_SLACK_EPS = 1e-6


@dataclass
class IlpOutcome:
    status: str  # optimal | feasible | infeasible | time_limit | error
    mapping: dict[str, Option] | None
    starts: dict[str, float] | None
    makespan: float | None
    objective: float | None
    runtime_s: float
    relaxation: list[Violation] | None = None
    notes: list[str] = field(default_factory=list)


def _solver(name: str, time_limit_s: float, msg: bool, seed: int) -> pulp.LpSolver:
    if name == "highs":
        return pulp.HiGHS(msg=msg, timeLimit=time_limit_s, threads=1, random_seed=seed)
    if name == "cbc":
        with warnings.catch_warnings():  # PuLP 3.x warns that the bundled CBC goes away in 4.0
            warnings.simplefilter("ignore", DeprecationWarning)
            return pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_s, threads=1, options=[f"randomSeed {seed + 1}"])
    raise ValueError(f"unknown solver {name!r}; use 'highs' or 'cbc'")


def _ancestors(problem: Problem) -> dict[str, set[str]]:
    anc: dict[str, set[str]] = {}
    for n in problem.order:
        a: set[str] = set()
        for p in problem.preds[n]:
            a |= anc[p] | {p}
        anc[n] = a
    return anc


def _build(problem: Problem, costs: CostView, elastic: bool):  # noqa: C901 - one formulation, kept together
    nodes = problem.order
    idx = {n: i for i, n in enumerate(nodes)}
    lp = pulp.LpProblem("hwsw_partition", pulp.LpMinimize)
    opts = problem.options
    x = {(n, k): lp.add_variable(f"x_{idx[n]}_{k}", cat="Binary") for n in nodes for k in range(len(opts[n]))}
    s = {n: lp.add_variable(f"s_{idx[n]}", lowBound=0) for n in nodes}
    C = lp.add_variable("C", lowBound=0)
    slacks: dict[str, tuple[pulp.LpVariable, float]] = {}

    def slack(name: str, limit: float) -> pulp.LpAffineExpression | float:
        if not elastic:
            return 0.0
        if name not in slacks:
            slacks[name] = (lp.add_variable(f"sl_{len(slacks)}", lowBound=0), limit)
        return slacks[name][0]

    def X(n: str, pe: str) -> pulp.LpAffineExpression:
        return pulp.lpSum(x[n, k] for k, o in enumerate(opts[n]) if o.pe == pe)

    def pes_of(n: str) -> list[str]:
        return sorted({o.pe for o in opts[n]}, key=problem.pes.index)

    dur = {n: pulp.lpSum(x[n, k] * costs.time[(n, o.impl)] for k, o in enumerate(opts[n])) for n in nodes}
    finish = {n: s[n] + dur[n] for n in nodes}
    for n in nodes:
        lp += pulp.lpSum(x[n, k] for k in range(len(opts[n]))) == 1, f"assign_{idx[n]}"
        lp += C >= finish[n], f"makespan_{idx[n]}"

    # communication + precedence
    comm_energy = []
    max_comm_total = 0.0
    for ei, (a, b) in enumerate(problem.edges):
        nb = costs.bytes[(a, b)]
        cm = lp.add_variable(f"cm_{ei}", lowBound=0)
        ce = lp.add_variable(f"ce_{ei}", lowBound=0)
        worst = 0.0
        for p in pes_of(a):
            for q in pes_of(b):
                if p == q:
                    continue
                c = problem.comm(p, q, nb)
                if c is None:
                    lp += X(a, p) + X(b, q) <= 1, f"noroute_{ei}_{p}_{q}"
                    continue
                worst = max(worst, c.time_ms)
                if c.time_ms > 0:
                    lp += cm >= c.time_ms * (X(a, p) + X(b, q) - 1), f"comm_{ei}_{p}_{q}"
                if c.energy_mj > 0:
                    lp += ce >= c.energy_mj * (X(a, p) + X(b, q) - 1), f"commE_{ei}_{p}_{q}"
        max_comm_total += worst
        comm_energy.append(ce)
        lp += s[b] >= finish[a] + cm, f"prec_{ei}"

    # exclusive PEs: independent nodes that may share one must not overlap
    big_m = sum(max(costs.time[(n, o.impl)] for o in opts[n]) for n in nodes) + max_comm_total + 1.0
    anc = _ancestors(problem)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            if a in anc[b] or b in anc[a]:
                continue
            shared = [p for p in pes_of(a) if p in pes_of(b) and problem.is_exclusive(p)]
            if not shared:
                continue
            o = lp.add_variable(f"o_{idx[a]}_{idx[b]}", cat="Binary")
            for p in shared:
                both = 2 - X(a, p) - X(b, p)
                lp += s[b] >= finish[a] - big_m * (1 - o) - big_m * both, f"ord1_{idx[a]}_{idx[b]}_{p}"
                lp += s[a] >= finish[b] - big_m * o - big_m * both, f"ord2_{idx[a]}_{idx[b]}_{p}"

    # capacities
    for pe in problem.pes:
        for r, cap in problem.platform.pe(pe).capacity.items():
            use = pulp.lpSum(
                x[n, k] * costs.res[(n, o.impl)].get(r, 0.0)
                for n in nodes for k, o in enumerate(opts[n]) if o.pe == pe
            )
            lp += use <= cap + slack(f"capacity[{pe}].{r}", cap), f"cap_{pe}_{r}"

    # timing constraints
    if problem.deadline_ms is not None:
        lp += C <= problem.deadline_ms + slack("deadline_ms", problem.deadline_ms), "deadline"
    for n, d in problem.task_deadlines.items():
        lp += finish[n] <= d + slack(f"deadline_ms[{n}]", d), f"tdeadline_{idx[n]}"
    if problem.period_ms is not None and problem.arrival == "periodic":
        lp += C <= problem.period_ms + slack("period_ms", problem.period_ms), "period"
    elif problem.period_ms is not None and problem.arrival == "streaming":
        per = problem.period_ms
        for pe in problem.pes:
            if problem.is_exclusive(pe):
                busy = pulp.lpSum(
                    x[n, k] * costs.time[(n, o.impl)] for n in nodes for k, o in enumerate(opts[n]) if o.pe == pe
                )
                lp += busy <= per + slack("period_ms", per), f"ii_{pe}"
        for n in nodes:
            for k, o in enumerate(opts[n]):
                if not problem.is_exclusive(o.pe):
                    lp += x[n, k] * costs.ii[(n, o.impl)] <= per + slack("period_ms", per), f"iis_{idx[n]}_{k}"

    # energy, power, $
    E = pulp.lpSum(x[n, k] * costs.energy[(n, o.impl)] for n in nodes for k, o in enumerate(opts[n]))
    E += pulp.lpSum(comm_energy) + problem.static_power_mw / 1000.0 * C
    if problem.energy_mj_max is not None:
        lp += E <= problem.energy_mj_max + slack("energy_mj_max", problem.energy_mj_max), "energy"
    pmax = min([v for v in (problem.power_mw_max, problem.platform.budgets.get("power_mw")) if v is not None],
               default=None)
    if pmax is not None:
        pname = "power_mw_max" if problem.power_mw_max == pmax else "budget.power_mw"
        if problem.period_ms:
            lp += E * (1000.0 / problem.period_ms) <= pmax + slack(pname, pmax), "power"
        elif not elastic:
            lp += 1000.0 * E <= pmax * C, "power"
    umax = min([v for v in (problem.usd_max, problem.platform.budgets.get("usd")) if v is not None], default=None)
    usd = None
    if umax is not None or "usd" in problem.weights:
        u = {pe: lp.add_variable(f"u_{pi}", cat="Binary") for pi, pe in enumerate(problem.pes)}
        for n in nodes:
            for pe in pes_of(n):
                lp += u[pe] >= X(n, pe), f"used_{idx[n]}_{pe}"
        usd = pulp.lpSum(u[pe] * problem.platform.pe(pe).usd for pe in problem.pes) + pulp.lpSum(
            x[n, k] * costs.res[(n, o.impl)].get("usd", 0.0) for n in nodes for k, o in enumerate(opts[n])
        )
        if umax is not None:
            uname = "usd_max" if problem.usd_max == umax else "budget.usd"
            lp += usd <= umax + slack(uname, umax), "usd"

    if elastic:
        lp += pulp.lpSum(v / max(abs(lim), 1e-9) for v, lim in slacks.values())
    else:
        norms = problem.norms
        terms = {"latency": C, "energy": E, "usd": usd if usd is not None else 0.0}
        lp += pulp.lpSum(w * terms[k] / norms[k] for k, w in problem.weights.items()) + 1e-6 * C
    return lp, x, s, C, slacks


def _status(lp: pulp.LpProblem) -> str:
    return {1: "optimal", 2: "feasible", -1: "infeasible", 0: "time_limit"}.get(lp.sol_status, "error")


def solve_ilp(
    problem: Problem,
    costs: CostView | None = None,
    solver: str = "highs",
    time_limit_s: float = 60.0,
    seed: int = 0,
    msg: bool = False,
    relax_if_infeasible: bool = True,
) -> IlpOutcome:
    costs = costs or problem.nominal()
    t0 = time.perf_counter()
    if any(not problem.options[n] for n in problem.order):
        return IlpOutcome("infeasible", None, None, None, None, time.perf_counter() - t0,
                          notes=["some node has no legal option (see lower-bound structural diagnostics)"])
    lp, x, s, C, _ = _build(problem, costs, elastic=False)
    lp.solve(_solver(solver, time_limit_s, msg, seed))
    status = _status(lp)
    if status in ("optimal", "feasible"):
        mapping = {}
        for n in problem.order:
            k = max(range(len(problem.options[n])), key=lambda k: x[n, k].value() or 0.0)
            mapping[n] = problem.options[n][k]
        starts = {n: float(s[n].value() or 0.0) for n in problem.order}
        return IlpOutcome(status, mapping, starts, float(C.value()), float(pulp.value(lp.objective)),
                          time.perf_counter() - t0)

    relaxation = None
    notes = []
    if status == "infeasible" and relax_if_infeasible:
        relaxation, notes = minimal_relaxation(problem, costs, solver, time_limit_s, seed, msg)
    return IlpOutcome(status, None, None, None, None, time.perf_counter() - t0, relaxation, notes)


def minimal_relaxation(
    problem: Problem,
    costs: CostView | None = None,
    solver: str = "highs",
    time_limit_s: float = 60.0,
    seed: int = 0,
    msg: bool = False,
) -> tuple[list[Violation] | None, list[str]]:
    """Elastic model: smallest Σ slack/limit that makes the instance feasible. Shared by the ILP and CP solvers."""
    costs = costs or problem.nominal()
    rlp, _, _, _, slacks = _build(problem, costs, elastic=True)
    rlp.solve(_solver(solver, time_limit_s, msg, seed))
    if _status(rlp) not in ("optimal", "feasible"):
        return None, ["elastic relaxation is also infeasible: the failure is structural (routes/options)"]
    return [
        violation(name, lim + float(v.value()), lim)
        for name, (v, lim) in slacks.items()
        if (v.value() or 0.0) > _SLACK_EPS * max(1.0, abs(lim))
    ], []
