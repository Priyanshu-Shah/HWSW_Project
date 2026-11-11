"""List-scheduling simulator: (problem, mapping, costs) -> schedule, latency, energy, II, resource use.

Model (see docs/model.md):
- exclusive PEs run one task at a time, non-preemptive, in list order; with `insertion=True` a task may
  start in an earlier idle gap that is long enough (insertion-based list scheduling, as in HEFT);
- spatial PEs (FPGA fabric) give every task its own hardware, so tasks run concurrently;
- a transfer between different PEs uses the fastest interface for its size; on a `shared` interface
  transfers are serialised in the order they are scheduled (bus contention) when `contention=True`; with
  `insertion=True` a transfer may also use an earlier idle gap on the bus;
- same-PE communication is free.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from estimate.annotate import CostView, Option, Problem

MappingT = Mapping[str, Option]


class InvalidMapping(ValueError):
    """The mapping uses an illegal option or needs a route that does not exist."""


@dataclass(frozen=True)
class TaskSlot:
    node: str
    pe: str
    impl: str
    start: float
    finish: float


@dataclass(frozen=True)
class CommSlot:
    src: str
    dst: str
    interface: str | None
    nbytes: float
    start: float
    finish: float
    energy_mj: float


@dataclass
class SimResult:
    makespan: float
    tasks: dict[str, TaskSlot]
    comms: list[CommSlot]
    energy_mj: float
    ii_ms: float
    usage: dict[str, dict[str, float]]
    usd: float


@dataclass
class _State:
    problem: Problem
    costs: CostView
    contention: bool
    insertion: bool = False
    pe_free: dict[str, float] = field(default_factory=dict)
    pe_busy: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    iface_free: dict[str, float] = field(default_factory=dict)
    iface_busy: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    _trial_busy: dict[str, list[tuple[float, float]]] = field(default_factory=dict, repr=False)
    tasks: dict[str, TaskSlot] = field(default_factory=dict)
    comms: list[CommSlot] = field(default_factory=list)

    def trial(self, node: str, opt: Option) -> tuple[float, float, list[CommSlot], dict[str, float]] | None:
        """Earliest (start, finish) of `node` on `opt` given what is placed; None if a route is missing."""
        p = self.problem
        iface_free = dict(self.iface_free)
        iface_busy = {k: list(v) for k, v in self.iface_busy.items()} if self.insertion else {}
        ready = 0.0
        comms: list[CommSlot] = []
        for u in sorted(p.preds[node], key=lambda x: (self.tasks[x].finish, x)):
            src = self.tasks[u]
            nb = self.costs.bytes[(u, node)]
            c = p.comm(src.pe, opt.pe, nb)
            if c is None:
                return None
            st = src.finish
            if c.interface is not None and self.contention and p.interface(c.interface).shared:
                if self.insertion:
                    busy = iface_busy.setdefault(c.interface, [])
                    st = _gap(busy, st, c.time_ms)
                    busy.append((st, st + c.time_ms))
                    busy.sort()
                else:
                    st = max(st, iface_free.get(c.interface, 0.0))
                iface_free[c.interface] = max(iface_free.get(c.interface, 0.0), st + c.time_ms)
            ft = st + c.time_ms
            comms.append(CommSlot(u, node, c.interface, nb, st, ft, c.energy_mj))
            ready = max(ready, ft)
        dur = self.costs.time[(node, opt.impl)]
        if not p.is_exclusive(opt.pe):
            start = ready
        elif self.insertion:
            start = self._earliest_gap(opt.pe, ready, dur)
        else:
            start = max(ready, self.pe_free.get(opt.pe, 0.0))
        self._trial_busy = iface_busy
        return start, start + dur, comms, iface_free

    def _earliest_gap(self, pe: str, ready: float, dur: float) -> float:
        return _gap(self.pe_busy.get(pe, []), ready, dur)

    def place(self, node: str, opt: Option) -> TaskSlot:
        t = self.trial(node, opt)
        if t is None:
            raise InvalidMapping(f"no interface route into {node} on {opt.pe}")
        start, finish, comms, iface_free = t
        self.iface_free = iface_free
        if self.insertion:
            self.iface_busy = self._trial_busy
        if self.problem.is_exclusive(opt.pe):
            self.pe_free[opt.pe] = max(self.pe_free.get(opt.pe, 0.0), finish)
            busy = self.pe_busy.setdefault(opt.pe, [])
            busy.append((start, finish))
            busy.sort()
        slot = TaskSlot(node, opt.pe, opt.impl, start, finish)
        self.tasks[node] = slot
        self.comms.extend(comms)
        return slot

    def result(self) -> SimResult:
        p, c = self.problem, self.costs
        makespan = max((s.finish for s in self.tasks.values()), default=0.0)
        energy = sum(c.energy[(n, s.impl)] for n, s in self.tasks.items())
        energy += sum(x.energy_mj for x in self.comms)
        energy += p.static_power_mw * makespan / 1000.0

        busy: dict[str, float] = {}
        ii = 0.0
        for n, s in self.tasks.items():
            if p.is_exclusive(s.pe):
                busy[s.pe] = busy.get(s.pe, 0.0) + c.time[(n, s.impl)]
            else:
                ii = max(ii, c.ii[(n, s.impl)])
        for x in self.comms:
            if x.interface is None:
                continue
            if p.interface(x.interface).shared:
                key = "if:" + x.interface
                busy[key] = busy.get(key, 0.0) + (x.finish - x.start)
            else:
                ii = max(ii, x.finish - x.start)
        ii = max([ii, *busy.values()])

        usage: dict[str, dict[str, float]] = {}
        usd = 0.0
        for n, s in self.tasks.items():
            u = usage.setdefault(s.pe, {})
            for r, v in c.res[(n, s.impl)].items():
                u[r] = u.get(r, 0.0) + v
        for pe, u in usage.items():
            usd += p.platform.pe(pe).usd + u.get("usd", 0.0)
        return SimResult(makespan, dict(self.tasks), list(self.comms), energy, ii, usage, usd)


def _gap(busy: list[tuple[float, float]], ready: float, dur: float) -> float:
    """Earliest start >= ready of a `dur`-long slot that fits between the sorted busy intervals."""
    t = ready
    for s, f in busy:
        if t + dur <= s + 1e-12:
            return t
        t = max(t, f)
    return t


def check_mapping(problem: Problem, mapping: MappingT) -> None:
    missing = [n for n in problem.order if n not in mapping]
    if missing:
        raise InvalidMapping(f"unmapped nodes: {missing}")
    for n in problem.order:
        if mapping[n] not in problem.options[n]:
            raise InvalidMapping(f"node {n}: {mapping[n]} is not a legal option {problem.options[n]}")


def upward_rank(problem: Problem, mapping: MappingT, costs: CostView) -> dict[str, float]:
    """b-level: longest path to a sink including the node's own time and mapped communication."""
    rank: dict[str, float] = {}
    for n in reversed(problem.order):
        best = 0.0
        for s in problem.succs[n]:
            c = problem.comm(mapping[n].pe, mapping[s].pe, costs.bytes[(n, s)])
            best = max(best, (c.time_ms if c else float("inf")) + rank[s])
        rank[n] = costs.time[(n, mapping[n].impl)] + best
    return rank


def simulate(
    problem: Problem,
    mapping: MappingT,
    costs: CostView | None = None,
    priority: Sequence[str] | None = None,
    contention: bool = True,
    insertion: bool = False,
) -> SimResult:
    """Schedule all nodes. `priority` (best first) overrides the default upward-rank list order.

    If `priority` also contains transfer items `"src->dst"`, transfers are list-scheduled as items of their own
    (unified mode): a transfer is ready once its producer is placed and takes its interface in priority order;
    a task is ready once all its incoming data has arrived. This reproduces any schedule whose per-resource orders
    are given, e.g. a CP-SAT schedule with bus contention.
    """
    check_mapping(problem, mapping)
    costs = costs or problem.nominal()
    if priority is not None and any(TRANSFER_SEP in x for x in priority):
        return _simulate_unified(problem, mapping, costs, priority, contention)
    topo = {n: i for i, n in enumerate(problem.order)}
    if priority is not None:
        pos = {n: i for i, n in enumerate(priority)}
        key = lambda n: (pos.get(n, len(pos)), topo[n])  # noqa: E731
    else:
        rank = upward_rank(problem, mapping, costs)
        key = lambda n: (-rank[n], topo[n])  # noqa: E731

    state = _State(problem, costs, contention, insertion)
    remaining = {n: len(problem.preds[n]) for n in problem.order}
    ready = [n for n in problem.order if remaining[n] == 0]
    while ready:
        ready.sort(key=key)
        n = ready.pop(0)
        state.place(n, mapping[n])
        for s in problem.succs[n]:
            remaining[s] -= 1
            if remaining[s] == 0:
                ready.append(s)
    return state.result()


TRANSFER_SEP = "->"


def transfer_id(src: str, dst: str) -> str:
    return f"{src}{TRANSFER_SEP}{dst}"


def _simulate_unified(
    problem: Problem, mapping: MappingT, costs: CostView, priority: Sequence[str], contention: bool
) -> SimResult:
    """List scheduling over tasks AND cross-PE transfers, each resource served in priority order (non-insertion)."""
    p = problem
    topo = {n: i for i, n in enumerate(p.order)}
    pos = {x: i for i, x in enumerate(priority)}
    state = _State(p, costs, contention)
    comm = {}
    for u, v in p.edges:
        c = p.comm(mapping[u].pe, mapping[v].pe, costs.bytes[(u, v)])
        if c is None:
            raise InvalidMapping(f"no interface route for {u}->{v}")
        comm[(u, v)] = c
    cross = {e for e, c in comm.items() if c.interface is not None}
    arrival: dict[tuple[str, str], float] = {}
    waiting = {n: len(p.preds[n]) for n in p.order}  # incoming data not yet available

    def key(item: str) -> tuple:
        if TRANSFER_SEP in item:
            u, v = item.split(TRANSFER_SEP)
            return (pos.get(item, len(pos) + topo[v]), 1, topo[u], topo[v])
        return (pos.get(item, len(pos) + topo[item]), 0, topo[item], 0)

    def data_arrived(u: str, v: str, t: float) -> None:
        arrival[(u, v)] = t
        waiting[v] -= 1
        if waiting[v] == 0:
            ready.append(v)

    ready = [n for n in p.order if waiting[n] == 0]
    while ready:
        ready.sort(key=key)
        item = ready.pop(0)
        if TRANSFER_SEP in item:
            u, v = item.split(TRANSFER_SEP)
            c = comm[(u, v)]
            st = state.tasks[u].finish
            if contention and p.interface(c.interface).shared:
                st = max(st, state.iface_free.get(c.interface, 0.0))
                state.iface_free[c.interface] = st + c.time_ms
            state.comms.append(CommSlot(u, v, c.interface, costs.bytes[(u, v)], st, st + c.time_ms, c.energy_mj))
            data_arrived(u, v, st + c.time_ms)
            continue
        n, opt = item, mapping[item]
        rdy = max((arrival[(u, n)] for u in p.preds[n]), default=0.0)
        dur = costs.time[(n, opt.impl)]
        start = max(rdy, state.pe_free.get(opt.pe, 0.0)) if p.is_exclusive(opt.pe) else rdy
        if p.is_exclusive(opt.pe):
            state.pe_free[opt.pe] = start + dur
        state.tasks[n] = TaskSlot(n, opt.pe, opt.impl, start, start + dur)
        for s_ in p.succs[n]:
            if (n, s_) in cross:
                ready.append(transfer_id(n, s_))
            else:
                state.comms.append(CommSlot(n, s_, None, costs.bytes[(n, s_)], start + dur, start + dur, 0.0))
                data_arrived(n, s_, start + dur)
    return state.result()
