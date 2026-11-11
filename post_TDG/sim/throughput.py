"""Exact streaming throughput and steady-state latency of a mapped task graph (max-plus / dataflow analysis).

Why (docs/survey/partitioning_survey.md §3.4, §7.3): the per-PE busy-time bottleneck we report as `ii_ms` is the
resource-MII lower bound of Chatha & Vemuri (TVLSI 2002). The achievable period also depends on cross-frame
state (recurrences), finite frame buffers, and transfers on shared links. The standard exact evaluator is the
maximum cycle ratio of the mapped graph under self-timed execution with a fixed per-resource order
(Ghamarian et al. ACSD 2006; Stuijk et al. DAC 2007, SDF3).

Construction (one actor per task, one per cross-PE transfer; edge weight = duration of its source actor):
- data edge u -> v of the same frame: 0 tokens (through a transfer actor when u, v are on different PEs);
- buffer back-edge v -> u carrying `buffers` tokens for each same-frame data edge (B frames of buffering);
- cross-frame edge with distance d: d tokens on the edge into its destination;
- exclusive PE: a cycle through its tasks in the order of the single-frame schedule, 1 token on the wrap edge;
- spatial (FPGA) task: self-loop of weight ii_ms with 1 token (a kernel accepts a new frame every ii);
- shared interface: a cycle through its transfers in schedule order with 1 token; other transfers self-loop.

Period = max over cycles of (Σ weights / Σ tokens), computed with Howard's policy iteration (Cochet-Terrasson
et al. 1998; Dasdan 2004). A cycle with zero tokens means deadlock (period = inf).

With the order fixed, the system is monotone: raising any duration never shortens the period or latency, so
evaluating at the low/high corners of every cost interval gives sound bounds (unlike dynamic list scheduling,
which can show Graham anomalies).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from estimate.annotate import CostView, Problem
from sim.scheduler import MappingT, SimResult, simulate

_EPS = 1e-9


@dataclass(frozen=True)
class Orders:
    """Static orders taken from a single-frame schedule (frozen across cost samples)."""

    pe: dict[str, list[str]]  # exclusive PE -> tasks in start order
    iface: dict[str, list[tuple[str, str]]]  # shared interface -> transfers (src, dst) in start order


@dataclass
class ThroughputResult:
    period_ms: float
    load_bound_ms: float
    critical_cycle: list[str]
    steady_latency_ms: float | None
    input_period_ms: float | None
    buffers: int


def orders_from(problem: Problem, sim: SimResult) -> Orders:
    pe: dict[str, list[str]] = {}
    for t in sorted(sim.tasks.values(), key=lambda t: (t.start, t.finish, t.node)):
        if problem.is_exclusive(t.pe):
            pe.setdefault(t.pe, []).append(t.node)
    iface: dict[str, list[tuple[str, str]]] = {}
    for c in sorted(sim.comms, key=lambda c: (c.start, c.finish, c.src, c.dst)):
        if c.interface is not None and problem.interface(c.interface).shared:
            iface.setdefault(c.interface, []).append((c.src, c.dst))
    return Orders(pe, iface)


def _graph(problem: Problem, mapping: MappingT, costs: CostView, orders: Orders, buffers: int):
    """Actors (name -> duration) and edges (src, dst, weight, tokens)."""
    dur: dict[str, float] = {}
    edges: list[tuple[str, str, float, int]] = []
    for n in problem.order:
        dur[n] = costs.time[(n, mapping[n].impl)]

    def link(u: str, v: str, tokens: int, back_buffer: bool) -> None:
        pu, pv = mapping[u].pe, mapping[v].pe
        nb = costs.bytes[(u, v)]
        if pu == pv:
            edges.append((u, v, dur[u], tokens))
            if back_buffer:
                edges.append((v, u, dur[v], buffers))
            return
        c = problem.comm(pu, pv, nb)
        if c is None:
            raise ValueError(f"no route for {u}->{v}")
        k = f"{u}->{v}"
        dur[k] = c.time_ms
        edges.append((u, k, dur[u], 0))
        edges.append((k, v, dur[k], tokens))
        if back_buffer:
            edges.append((v, u, dur[v], buffers))
        if c.interface is None or not problem.interface(c.interface).shared:
            edges.append((k, k, dur[k], 1))

    for u, v in problem.edges:
        link(u, v, 0, back_buffer=True)
    for u, v, d in problem.back_edges:
        link(u, v, d, back_buffer=False)

    for seq in orders.pe.values():
        for a, b in zip(seq, seq[1:], strict=False):
            edges.append((a, b, dur[a], 0))
        edges.append((seq[-1], seq[0], dur[seq[-1]], 1))
    for n in problem.order:
        if not problem.is_exclusive(mapping[n].pe):
            edges.append((n, n, costs.ii[(n, mapping[n].impl)], 1))
    for seq_t in orders.iface.values():
        seq = [f"{a}->{b}" for a, b in seq_t if f"{a}->{b}" in dur]
        if not seq:
            continue
        for a, b in zip(seq, seq[1:], strict=False):
            edges.append((a, b, dur[a], 0))
        edges.append((seq[-1], seq[0], dur[seq[-1]], 1))
    return dur, edges


def max_cycle_ratio(nodes: Sequence[str], edges: Sequence[tuple[str, str, float, int]]) -> tuple[float, list[str]]:
    """Howard's policy iteration for max over cycles of Σw/Σt. Returns (ratio, critical cycle nodes).

    Returns (inf, cycle) if a cycle with zero tokens and positive weight exists (deadlock); (0, []) if acyclic.
    """
    out: dict[str, list[tuple[str, float, int]]] = {n: [] for n in nodes}
    for u, v, w, t in edges:
        out[u].append((v, w, t))
    # deadlock check: cycles in the zero-token subgraph
    zero = {n: [v for v, _, t in out[n] if t == 0] for n in nodes}
    color: dict[str, int] = {}
    for root in nodes:
        if color.get(root):
            continue
        stack = [(root, iter(zero[root]))]
        color[root] = 1
        path = [root]
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                color[node] = 2
                stack.pop()
                path.pop()
            elif color.get(nxt) == 1:
                return float("inf"), path[path.index(nxt):]
            elif not color.get(nxt):
                color[nxt] = 1
                stack.append((nxt, iter(zero[nxt])))
                path.append(nxt)
    # prune nodes that cannot reach a cycle (no out-edges, iteratively)
    alive = set(nodes)
    changed = True
    while changed:
        changed = False
        for n in list(alive):
            if not any(v in alive for v, _, _ in out[n]):
                alive.discard(n)
                changed = True
    if not alive:
        return 0.0, []
    out = {n: [(v, w, t) for v, w, t in out[n] if v in alive] for n in alive}
    pol = {n: max(out[n], key=lambda e: e[1]) for n in alive}
    eta: dict[str, float] = {}
    x: dict[str, float] = {}
    for _ in range(10_000):
        # value determination on the functional graph of the policy
        eta.clear()
        x.clear()
        state: dict[str, int] = {}
        for start in alive:
            if start in state:
                continue
            path = []
            n = start
            while n not in state:
                state[n] = 1
                path.append(n)
                n = pol[n][0]
            if state[n] == 1 and n not in eta:  # found a new cycle starting at n
                cyc = path[path.index(n):]
                w = sum(pol[c][1] for c in cyc)
                t = sum(pol[c][2] for c in cyc)
                r = w / t if t > 0 else float("inf")
                eta[n], x[n] = r, 0.0
                for c in reversed(cyc[1:]):
                    v, wc, tc = pol[c]
                    eta[c], x[c] = r, wc - r * tc + x[v]
            for c in reversed(path):
                if c not in eta:
                    v, wc, tc = pol[c]
                    eta[c], x[c] = eta[v], wc - eta[v] * tc + x[v]
            for c in path:
                state[c] = 2
        # improvement
        changed = False
        for n in alive:
            best_eta = max(eta[v] for v, _, _ in out[n])
            if best_eta > eta[n] + _EPS:
                pol[n] = max((e for e in out[n] if eta[e[0]] >= best_eta - _EPS), key=lambda e: e[1])
                changed = True
        if not changed:
            for n in alive:
                best = None
                for e in out[n]:
                    v, w, t = e
                    if abs(eta[v] - eta[n]) <= _EPS:
                        val = w - eta[n] * t + x[v]
                        if val > x[n] + _EPS * max(1.0, abs(x[n])) and (best is None or val > best[0]):
                            best = (val, e)
                if best is not None:
                    pol[n] = best[1]
                    changed = True
        if not changed:
            break
    lam = max(eta.values())
    # critical cycle: follow the policy from a node with maximal eta
    n = next(k for k, v in eta.items() if abs(v - lam) <= _EPS)
    seen: list[str] = []
    while n not in seen:
        seen.append(n)
        n = pol[n][0]
    return lam, seen[seen.index(n):]


def steady_latency(
    problem: Problem,
    dur: Mapping[str, float],
    edges: Sequence[tuple[str, str, float, int]],
    input_period: float,
    iterations: int | None = None,
) -> float:
    """Self-timed execution with frame k released at k·input_period; max latency over the second half of a run.

    Latency of frame k = finish of its last task - release(k). Unbounded (inf) if the input is faster than the
    graph's period.
    """
    nodes = list(dur)
    ins: dict[str, list[tuple[str, float, int]]] = {n: [] for n in nodes}
    for u, v, w, t in edges:
        ins[v].append((u, w, t))
    # topological order of the zero-token subgraph (exists when not deadlocked)
    indeg = {n: 0 for n in nodes}
    for u, v, _, t in edges:
        if t == 0:
            indeg[v] += 1
    order, ready = [], [n for n in nodes if indeg[n] == 0]
    succ0: dict[str, list[str]] = {n: [] for n in nodes}
    for u, v, _, t in edges:
        if t == 0:
            succ0[u].append(v)
    while ready:
        n = ready.pop()
        order.append(n)
        for v in succ0[n]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    if len(order) != len(nodes):
        return float("inf")
    sources = {n for n in problem.order if not problem.preds[n]}
    k_max = iterations or max(40, 4 * len(nodes))
    start: list[dict[str, float]] = []
    lat = []
    for k in range(k_max):
        release = k * input_period
        s: dict[str, float] = {}
        for n in order:
            t0 = release if n in sources else 0.0
            for u, w, tok in ins[n]:
                if tok == 0:
                    t0 = max(t0, s[u] + w)
                elif k - tok >= 0:
                    t0 = max(t0, start[k - tok][u] + w)
            s[n] = t0
        start.append(s)
        lat.append(max(s[n] + dur[n] for n in problem.order) - release)
    tail = lat[k_max // 2:]
    if tail[-1] > tail[0] + max(1e-6, 1e-6 * tail[0]) and tail[-1] > lat[k_max // 4] + 1e-6:
        return float("inf")  # still growing: input faster than throughput
    return max(tail)


def throughput(
    problem: Problem,
    mapping: MappingT,
    costs: CostView | None = None,
    orders: Orders | None = None,
    priority: Sequence[str] | None = None,
    buffers: int = 2,
    input_period_ms: float | None = None,
    insertion: bool = False,
    contention: bool = True,
) -> ThroughputResult:
    """Exact period (max cycle ratio) and steady-state latency for a mapping.

    `orders` freezes the per-resource order (pass the nominal one when evaluating cost samples); by default it is
    taken from the single-frame schedule produced with these costs. `input_period_ms` defaults to the problem's
    arrival period.
    """
    if buffers < 1:
        raise ValueError("buffers must be >= 1")
    costs = costs or problem.nominal()
    sim = simulate(problem, mapping, costs, priority, contention, insertion)
    orders = orders or orders_from(problem, sim)
    dur, edges = _graph(problem, mapping, costs, orders, buffers)
    period, cycle = max_cycle_ratio(list(dur), edges)
    p_in = input_period_ms if input_period_ms is not None else problem.period_ms
    latency = None
    if p_in is not None:
        latency = steady_latency(problem, dur, edges, p_in) if p_in >= period - 1e-9 else float("inf")
    return ThroughputResult(period, sim.ii_ms, cycle, latency, p_in, buffers)
