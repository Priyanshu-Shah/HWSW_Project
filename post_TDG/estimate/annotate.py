"""Bind / cost annotation: resolve parameterised IR costs into concrete numbers for one case study.

`annotate(graph, platform)` evaluates every cost expression at the graph's workload, applies the
case study's PE subset (action masking) and compatibility rules, and returns a `Problem` that the
simulator, lower bound and solvers all consume. Costs under uncertainty are exposed as `CostView`s:
nominal, interval ends, or a Monte Carlo draw of multiplicative factors.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from estimate.expr import evaluate
from schemas.common import Param
from schemas.ir import TaskGraph, check_dag
from schemas.platform import Interface, Platform

# Uncertainty keys: ("t"|"e"|"p"|"ii", node, impl), ("r", node, impl, resource), ("b", src, dst)
UncKey = tuple[str, ...]


@dataclass(frozen=True)
class Option:
    """A legal (PE, implementation) choice for a node."""

    pe: str
    impl: str


@dataclass(frozen=True)
class Comm:
    time_ms: float
    energy_mj: float
    interface: str | None


@dataclass
class CostView:
    """Concrete costs for one realisation of the uncertain parameters."""

    time: dict[tuple[str, str], float]
    energy: dict[tuple[str, str], float]
    ii: dict[tuple[str, str], float]
    res: dict[tuple[str, str], dict[str, float]]
    bytes: dict[tuple[str, str], float]


@dataclass
class _Nominal:
    value: float
    unc: tuple[float, float]


@dataclass
class Problem:
    graph: TaskGraph
    platform: Platform
    workload: dict[str, float]
    pes: list[str]
    order: list[str]
    edges: list[tuple[str, str]]
    back_edges: list[tuple[str, str, int]]
    preds: dict[str, list[str]]
    succs: dict[str, list[str]]
    options: dict[str, list[Option]]
    arrival: Literal["one_shot", "periodic", "streaming"]
    period_ms: float | None
    buffers: int
    deadline_ms: float | None
    task_deadlines: dict[str, float]
    energy_mj_max: float | None
    power_mw_max: float | None
    usd_max: float | None
    weights: dict[str, float]
    structural: list[str]
    _time: dict[tuple[str, str], _Nominal] = field(repr=False)
    _energy: dict[tuple[str, str], _Nominal | None] = field(repr=False)
    _power: dict[tuple[str, str], _Nominal | None] = field(repr=False)
    _ii: dict[tuple[str, str], _Nominal | None] = field(repr=False)
    _res: dict[tuple[str, str], dict[str, _Nominal]] = field(repr=False)
    _bytes: dict[tuple[str, str], _Nominal] = field(repr=False)
    _norms: dict[str, float] | None = field(default=None, repr=False)

    # -- structure ---------------------------------------------------------------------------
    @property
    def static_power_mw(self) -> float:
        """Static power of every allowed PE (the board is powered for the whole iteration)."""
        return sum(self.platform.pe(p).static_power_mw for p in self.pes)

    def pe_kind(self, pe: str) -> str:
        return self.platform.pe(pe).kind

    def is_exclusive(self, pe: str) -> bool:
        return self.platform.pe(pe).concurrency == "exclusive"

    def interface(self, iface_id: str) -> Interface:
        for i in self.platform.interfaces:
            if i.id == iface_id:
                return i
        raise KeyError(iface_id)

    def comm(self, src_pe: str, dst_pe: str, nbytes: float) -> Comm | None:
        """Fastest route for `nbytes` between two PEs; zero cost on the same PE; None if unreachable."""
        if src_pe == dst_pe:
            return Comm(0.0, 0.0, None)
        best: Interface | None = None
        for itf in self.platform.interfaces_between(src_pe, dst_pe):
            if best is None or itf.transfer_ms(nbytes) < best.transfer_ms(nbytes):
                best = itf
        if best is None:
            return None
        return Comm(best.transfer_ms(nbytes), best.transfer_mj(nbytes), best.id)

    # -- cost views -------------------------------------------------------------------------
    def uncertainty(self) -> dict[UncKey, tuple[float, float]]:
        """All uncertain quantities with non-trivial intervals, in a deterministic order."""
        out: dict[UncKey, tuple[float, float]] = {}
        for (n, i), v in self._time.items():
            out[("t", n, i)] = v.unc
            if (e := self._energy[(n, i)]) is not None:
                out[("e", n, i)] = e.unc
            if (p := self._power[(n, i)]) is not None:
                out[("p", n, i)] = p.unc
            if (ii := self._ii[(n, i)]) is not None:
                out[("ii", n, i)] = ii.unc
            for r, rv in self._res[(n, i)].items():
                out[("r", n, i, r)] = rv.unc
        for (s, d), v in self._bytes.items():
            out[("b", s, d)] = v.unc
        return {k: u for k, u in out.items() if u != (1.0, 1.0)}

    def view(self, factors: Mapping[UncKey, float] | None = None) -> CostView:
        f = factors or {}

        def g(key: UncKey) -> float:
            return f.get(key, 1.0)

        time = {k: v.value * g(("t", *k)) for k, v in self._time.items()}
        energy: dict[tuple[str, str], float] = {}
        ii: dict[tuple[str, str], float] = {}
        for k in self._time:
            e, p, iiv = self._energy[k], self._power[k], self._ii[k]
            if e is not None:
                energy[k] = e.value * g(("e", *k))
            elif p is not None:
                energy[k] = p.value * g(("p", *k)) * time[k] / 1000.0
            else:
                energy[k] = 0.0
            ii[k] = iiv.value * g(("ii", *k)) if iiv is not None else time[k]
        res = {k: {r: v.value * g(("r", *k, r)) for r, v in rv.items()} for k, rv in self._res.items()}
        nbytes = {k: v.value * g(("b", *k)) for k, v in self._bytes.items()}
        return CostView(time, energy, ii, res, nbytes)

    def nominal(self) -> CostView:
        return self.view()

    def bound_view(self, side: Literal["lo", "hi"]) -> CostView:
        idx = 0 if side == "lo" else 1
        return self.view({k: u[idx] for k, u in self.uncertainty().items()})

    def sample_factors(self, rng: np.random.Generator, n: int) -> list[dict[UncKey, float]]:
        """Draw n independent factor sets, uniform within each interval (deterministic key order)."""
        unc = self.uncertainty()
        keys = list(unc)
        if not keys:
            return [{} for _ in range(n)]
        lo = np.array([unc[k][0] for k in keys])
        hi = np.array([unc[k][1] for k in keys])
        draws = rng.uniform(lo, hi, size=(n, len(keys)))
        return [dict(zip(keys, row.tolist(), strict=True)) for row in draws]

    # -- normalisation for the objective -------------------------------------------------------
    @property
    def norms(self) -> dict[str, float]:
        """Lower-bound reference values used to make objective terms dimensionless."""
        if self._norms is None:
            v = self.nominal()
            lat = critical_path(self, {n: min(v.time[(n, o.impl)] for o in self.options[n]) for n in self.order
                                       if self.options[n]})
            energy = sum(min(v.energy[(n, o.impl)] for o in self.options[n]) for n in self.order if self.options[n])
            energy += self.static_power_mw * lat / 1000.0
            self._norms = {"latency": max(lat, 1e-9), "energy": max(energy, 1e-9), "usd": 1.0}
        return self._norms


def critical_path(problem: Problem, node_time: Mapping[str, float]) -> float:
    """Longest path with the given node times and zero communication (unlimited parallelism)."""
    finish: dict[str, float] = {}
    for n in problem.order:
        start = max((finish[p] for p in problem.preds[n]), default=0.0)
        finish[n] = start + node_time.get(n, 0.0)
    return max(finish.values(), default=0.0)


def _nom(p: Param | None, env: Mapping[str, float]) -> _Nominal | None:
    if p is None:
        return None
    return _Nominal(evaluate(p.expr, env), p.unc)


def _limit(x: float | str | None, env: Mapping[str, float]) -> float | None:
    return None if x is None else evaluate(x, env)


def annotate(graph: TaskGraph, platform: Platform, workload: Mapping[str, float] | None = None) -> Problem:
    """Resolve costs and legal options. `workload` overrides entries of graph.workload."""
    env = {**graph.workload, **(workload or {})}
    pe_ids = [p.id for p in platform.pes]
    allowed = graph.platform.allowed_pes or pe_ids
    unknown = set(allowed) - set(pe_ids)
    if unknown:
        raise ValueError(f"allowed_pes not on platform {platform.id}: {sorted(unknown)}")
    pes = [p for p in pe_ids if p in allowed]  # platform order, for determinism

    edges = [(e.src, e.dst) for e in graph.edges if e.distance == 0]
    back_edges = [(e.src, e.dst, e.distance) for e in graph.edges if e.distance > 0]
    order = check_dag([n.id for n in graph.nodes], edges)
    preds: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
    succs: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
    for s, d in edges:
        preds[d].append(s)
        succs[s].append(d)

    options: dict[str, list[Option]] = {}
    structural: list[str] = []
    t: dict[tuple[str, str], _Nominal] = {}
    e: dict[tuple[str, str], _Nominal | None] = {}
    pw: dict[tuple[str, str], _Nominal | None] = {}
    ii: dict[tuple[str, str], _Nominal | None] = {}
    res: dict[tuple[str, str], dict[str, _Nominal]] = {}
    for node in graph.nodes:
        opts: list[Option] = []
        candidates = [node.pinned_pe] if node.pinned_pe else pes
        if node.pinned_pe and node.pinned_pe not in pes:
            structural.append(f"node {node.id}: pinned to {node.pinned_pe}, which is not an allowed PE")
            candidates = []
        for impl in node.implementations:
            key = (node.id, impl.id)
            t[key] = _nom(impl.cost.time_ms, env)  # type: ignore[assignment]
            e[key] = _nom(impl.cost.energy_mj, env)
            pw[key] = _nom(impl.cost.power_mw, env)
            ii[key] = _nom(impl.cost.ii_ms, env)
            res[key] = {r: _nom(p, env) for r, p in impl.cost.resources.items()}  # type: ignore[misc]
            for pe in candidates:
                if platform.pe(pe).type == impl.pe_type:
                    opts.append(Option(pe, impl.id))
        if not opts and not (node.pinned_pe and node.pinned_pe not in pes):
            types = sorted({i.pe_type for i in node.implementations})
            structural.append(f"node {node.id}: no allowed PE matches implementation types {types}")
        options[node.id] = opts

    nbytes = {(ed.src, ed.dst): _nom(ed.bytes, env) for ed in graph.edges}  # keyed by (src, dst) for both kinds
    task_deadlines = {n.id: evaluate(n.deadline_ms, env) for n in graph.nodes if n.deadline_ms is not None}

    for key, v in {**t, **{k: x for k, x in nbytes.items() if x is not None}}.items():
        if v.value < 0:
            raise ValueError(f"negative cost for {key}: {v.value}")

    return Problem(
        graph=graph,
        platform=platform,
        workload=dict(env),
        pes=pes,
        order=order,
        edges=edges,
        back_edges=back_edges,
        preds=preds,
        succs=succs,
        options=options,
        arrival=graph.arrival.kind,
        period_ms=_limit(graph.arrival.period_ms, env),
        buffers=graph.arrival.buffers,
        deadline_ms=_limit(graph.constraints.deadline_ms, env),
        task_deadlines=task_deadlines,
        energy_mj_max=_limit(graph.constraints.energy_mj_max, env),
        power_mw_max=_limit(graph.constraints.power_mw_max, env),
        usd_max=_limit(graph.constraints.usd_max, env),
        weights=dict(graph.objective.weights),
        structural=structural,
        _time=t,
        _energy=e,
        _power=pw,
        _ii=ii,
        _res=res,
        _bytes=nbytes,  # type: ignore[arg-type]
    )
