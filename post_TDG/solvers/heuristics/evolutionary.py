"""Genetic algorithm and NSGA-II with a schedule decoder.

Encoding (Wang, Siegel, Roychowdhury & Maciejewski, JPDC 1997, two-part chromosome; random-key variant):
- gene part 1: index of the node's legal (PE, implementation) option, so type/compatibility is feasible by
  construction (Erbas et al. 2006: per-gene feasible domains);
- gene part 2: a real-valued priority key per node; the decoder list-schedules ready nodes in decreasing key,
  so every key vector decodes to a valid topological order.

Constraint handling:
- repair of PE-capacity overflow before evaluation, written back to the genome (Erbas et al. 2006: GAs
  without repair did not converge; Hou et al. 2018: repair "dispersion correction");
- constraint-domination for selection (Deb et al. 2002): feasible beats infeasible; among infeasible, smaller
  total relative violation wins; among feasible, better objective / Pareto rank wins.

Seeding (Wang et al. seed with a heuristic): HEFT, greedy EFT, all-SW, and random individuals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from estimate.annotate import Option, Problem
from sim.evaluate import Evaluation, evaluate
from sim.scheduler import InvalidMapping
from solvers.heuristics.baselines import all_sw, greedy_eft
from solvers.heuristics.list_sched import heft, upward_ranks

_BIG = 1e18


@dataclass
class Individual:
    genes: np.ndarray  # option index per node (problem.order)
    keys: np.ndarray  # priority key per node
    ev: Evaluation | None = None
    violation: float = _BIG
    objs: tuple[float, ...] = ()


def _decode(problem: Problem, ind: Individual) -> tuple[dict[str, Option], list[str]]:
    mapping = {n: problem.options[n][int(g)] for n, g in zip(problem.order, ind.genes, strict=True)}
    topo = {n: i for i, n in enumerate(problem.order)}
    priority = sorted(problem.order, key=lambda n: (-ind.keys[topo[n]], topo[n]))
    return mapping, priority


def _repair(problem: Problem, ind: Individual) -> None:
    """Move nodes off over-capacity PEs to their fastest option that still fits (in place)."""
    v = problem.nominal()
    idx = {n: i for i, n in enumerate(problem.order)}
    for _ in range(len(problem.order)):
        mapping, _ = _decode(problem, ind)
        usage: dict[str, dict[str, float]] = {}
        for n, o in mapping.items():
            u = usage.setdefault(o.pe, {})
            for r, val in v.res[(n, o.impl)].items():
                u[r] = u.get(r, 0.0) + val
        over = [(pe, r) for pe, u in usage.items() for r, cap in problem.platform.pe(pe).capacity.items()
                if u.get(r, 0.0) > cap * (1 + 1e-9)]
        if not over:
            return
        pe, r = over[0]
        best_move: tuple[float, str, int] | None = None
        for n, o in mapping.items():
            if o.pe != pe or v.res[(n, o.impl)].get(r, 0.0) <= 0:
                continue
            for k, alt in enumerate(problem.options[n]):
                if alt.pe == pe and v.res[(n, alt.impl)].get(r, 0.0) >= v.res[(n, o.impl)].get(r, 0.0):
                    continue
                if alt.pe != pe:
                    cap = problem.platform.pe(alt.pe).capacity
                    au = usage.get(alt.pe, {})
                    if any(au.get(rr, 0.0) + v.res[(n, alt.impl)].get(rr, 0.0) > c * (1 + 1e-9) for rr, c in cap.items()):
                        continue
                freed = v.res[(n, o.impl)].get(r, 0.0) - (v.res[(n, alt.impl)].get(r, 0.0) if alt.pe == pe else 0.0)
                cost = (v.time[(n, alt.impl)] - v.time[(n, o.impl)]) / max(freed, 1e-12)
                if best_move is None or cost < best_move[0]:
                    best_move = (cost, n, k)
        if best_move is None:
            return
        ind.genes[idx[best_move[1]]] = best_move[2]


def _evaluate(problem: Problem, ind: Individual, objectives: Sequence[str]) -> None:
    _repair(problem, ind)
    mapping, priority = _decode(problem, ind)
    try:
        ev = evaluate(problem, mapping, priority=priority)
    except InvalidMapping:
        ind.ev, ind.violation, ind.objs = None, _BIG, tuple(_BIG for _ in objectives)
        return
    ind.ev = ev
    ind.violation = sum(max(0.0, x.relative) for x in ev.violations)
    metric = {"latency": ev.sim.makespan, "energy": ev.sim.energy_mj, "usd": ev.sim.usd, "objective": ev.objective}
    ind.objs = tuple(metric[o] for o in objectives)


def _from_mapping(problem: Problem, mapping: dict[str, Option], order: Sequence[str] | None = None) -> Individual:
    genes = np.array([problem.options[n].index(mapping[n]) for n in problem.order], dtype=int)
    if order is None:
        rank = upward_ranks(problem)
        keys = np.array([rank[n] for n in problem.order], dtype=float)
        keys = keys / (keys.max() or 1.0)
    else:
        pos = {n: i for i, n in enumerate(order)}
        keys = np.array([1.0 - pos[n] / max(len(order), 1) for n in problem.order], dtype=float)
    return Individual(genes, keys)


def _random(problem: Problem, rng: np.random.Generator) -> Individual:
    genes = np.array([rng.integers(len(problem.options[n])) for n in problem.order], dtype=int)
    return Individual(genes, rng.random(len(problem.order)))


def _seeds(problem: Problem) -> list[Individual]:
    out = []
    plan = heft(problem, insertion=False)
    out.append(_from_mapping(problem, plan.mapping, plan.priority))
    out.append(_from_mapping(problem, greedy_eft(problem)))
    out.append(_from_mapping(problem, all_sw(problem)))
    return out


def _vary(problem: Problem, a: Individual, b: Individual, rng: np.random.Generator, pc: float, pm: float,
          sigma: float) -> Individual:
    n = len(problem.order)
    genes, keys = a.genes.copy(), a.keys.copy()
    if rng.random() < pc:
        mask = rng.random(n) < 0.5
        genes[mask] = b.genes[mask]
        mask = rng.random(n) < 0.5
        keys[mask] = b.keys[mask]
    for i, node in enumerate(problem.order):
        if rng.random() < pm and len(problem.options[node]) > 1:
            genes[i] = rng.integers(len(problem.options[node]))
    keys = keys + rng.normal(0.0, sigma, n) * (rng.random(n) < pm * 2)
    return Individual(genes, keys)


def _better_single(a: Individual, b: Individual) -> bool:
    if (a.violation == 0) != (b.violation == 0):
        return a.violation == 0
    if a.violation > 0:
        return a.violation < b.violation
    return a.objs[0] < b.objs[0]


def genetic_algorithm(
    problem: Problem,
    pop_size: int = 40,
    generations: int = 60,
    pc: float = 0.9,
    pm: float | None = None,
    sigma: float = 0.1,
    seed: int = 0,
) -> tuple[dict[str, Option], list[str]]:
    """Single-objective GA (problem objective) with constraint-domination tournaments and (μ+λ) survival."""
    rng = np.random.default_rng(seed)
    pm = pm if pm is not None else 1.0 / max(len(problem.order), 1)
    objectives = ("objective",)
    pop = _seeds(problem)
    while len(pop) < pop_size:
        pop.append(_random(problem, rng))
    for ind in pop:
        _evaluate(problem, ind, objectives)

    def tournament() -> Individual:
        i, j = rng.integers(len(pop), size=2)
        return pop[i] if _better_single(pop[i], pop[j]) else pop[j]

    for _ in range(generations):
        children = []
        for _ in range(pop_size):
            child = _vary(problem, tournament(), tournament(), rng, pc, pm, sigma)
            _evaluate(problem, child, objectives)
            children.append(child)
        merged = pop + children
        merged.sort(key=lambda x: (x.violation > 0, x.violation if x.violation > 0 else x.objs[0]))
        pop = merged[:pop_size]
    return _decode(problem, pop[0])


# --- NSGA-II ------------------------------------------------------------------------------------------


def _dominates(a: Individual, b: Individual) -> bool:
    """Constraint-domination (Deb et al. 2002)."""
    if a.violation == 0 and b.violation > 0:
        return True
    if a.violation > 0 and b.violation > 0:
        return a.violation < b.violation
    if a.violation > 0:
        return False
    return all(x <= y for x, y in zip(a.objs, b.objs, strict=True)) and any(
        x < y for x, y in zip(a.objs, b.objs, strict=True))


def _fronts(pop: list[Individual]) -> list[list[int]]:
    s: list[list[int]] = [[] for _ in pop]
    count = [0] * len(pop)
    fronts: list[list[int]] = [[]]
    for i, a in enumerate(pop):
        for j, b in enumerate(pop):
            if i != j and _dominates(a, b):
                s[i].append(j)
            elif i != j and _dominates(b, a):
                count[i] += 1
        if count[i] == 0:
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in s[i]:
                count[j] -= 1
                if count[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return fronts[:-1]


def _crowding(pop: list[Individual], front: list[int]) -> dict[int, float]:
    d = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: float("inf") for i in front}
    for m in range(len(pop[front[0]].objs)):
        srt = sorted(front, key=lambda i: pop[i].objs[m])
        lo, hi = pop[srt[0]].objs[m], pop[srt[-1]].objs[m]
        d[srt[0]] = d[srt[-1]] = float("inf")
        if hi > lo:
            for a, b, c in zip(srt, srt[1:], srt[2:], strict=False):
                d[b] += (pop[c].objs[m] - pop[a].objs[m]) / (hi - lo)
    return d


@dataclass
class ParetoPoint:
    mapping: dict[str, Option]
    priority: list[str]
    objectives: tuple[float, ...]
    feasible: bool


def nsga2(
    problem: Problem,
    objectives: Sequence[str] = ("latency", "energy"),
    pop_size: int = 40,
    generations: int = 60,
    pc: float = 0.9,
    pm: float | None = None,
    sigma: float = 0.1,
    seed: int = 0,
) -> list[ParetoPoint]:
    """NSGA-II (Deb et al. 2002) with constraint-domination. Returns the final non-dominated set (deduplicated)."""
    rng = np.random.default_rng(seed)
    pm = pm if pm is not None else 1.0 / max(len(problem.order), 1)
    pop = _seeds(problem)
    while len(pop) < pop_size:
        pop.append(_random(problem, rng))
    for ind in pop:
        _evaluate(problem, ind, objectives)

    def rank_and_crowd(p: list[Individual]) -> tuple[dict[int, int], dict[int, float]]:
        rank, crowd = {}, {}
        for r, f in enumerate(_fronts(p)):
            crowd.update(_crowding(p, f))
            rank.update({i: r for i in f})
        return rank, crowd

    for _ in range(generations):
        rank, crowd = rank_and_crowd(pop)

        def tournament() -> Individual:
            i, j = (int(x) for x in rng.integers(len(pop), size=2))
            return pop[i] if (rank[i], -crowd[i]) <= (rank[j], -crowd[j]) else pop[j]

        children = []
        for _ in range(pop_size):
            child = _vary(problem, tournament(), tournament(), rng, pc, pm, sigma)
            _evaluate(problem, child, objectives)
            children.append(child)
        merged = pop + children
        nxt: list[Individual] = []
        for f in _fronts(merged):
            if len(nxt) + len(f) <= pop_size:
                nxt.extend(merged[i] for i in f)
            else:
                cd = _crowding(merged, f)
                nxt.extend(merged[i] for i in sorted(f, key=lambda i: -cd[i])[: pop_size - len(nxt)])
                break
        pop = nxt
    front = [pop[i] for i in _fronts(pop)[0]]
    out: dict[tuple, ParetoPoint] = {}
    for ind in front:
        mapping, priority = _decode(problem, ind)
        k = tuple((n, o.pe, o.impl) for n, o in mapping.items())
        if k not in out:
            out[k] = ParetoPoint(mapping, priority, ind.objs, ind.violation == 0)
    return sorted(out.values(), key=lambda p: p.objectives)
