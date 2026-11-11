"""HEFT and PEFT list scheduling, as published, extended with (PE, implementation) options and resource checks.

HEFT: H. Topcuoglu, S. Hariri, M.-Y. Wu, "Performance-effective and low-complexity task scheduling for
heterogeneous computing", IEEE TPDS 13(3), 2002.
PEFT: H. Arabnejad, J. G. Barbosa, "List scheduling algorithm for heterogeneous systems by an optimistic cost
table", IEEE TPDS 25(3), 2014.

Mapping to our model:
- a "processor" choice is a legal (PE, implementation) option; average costs are taken over a node's options;
- the average communication cost of an edge is the mean transfer time over distinct (PE, PE) pairs the two
  endpoints can use (same-PE pairs excluded, as in HEFT's average over processor pairs);
- extension (not in the papers): an option is skipped if it would exceed a PE capacity, unless no option fits;
- scheduling is insertion-based (HEFT's default) through the simulator's `insertion=True` mode.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from estimate.annotate import CostView, Option, Problem
from sim.scheduler import _State


@dataclass
class Plan:
    """A heuristic's output: mapping plus the list order and scheduling mode that reproduce its schedule."""

    mapping: dict[str, Option]
    priority: list[str] | None = None
    insertion: bool = False


def _avg_time(problem: Problem, v: CostView, n: str) -> float:
    return float(np.mean([v.time[(n, o.impl)] for o in problem.options[n]]))


def _avg_comm(problem: Problem, v: CostView, a: str, b: str) -> float:
    nb = v.bytes[(a, b)]
    pes_a = {o.pe for o in problem.options[a]}
    pes_b = {o.pe for o in problem.options[b]}
    times = [c.time_ms for p in pes_a for q in pes_b if p != q and (c := problem.comm(p, q, nb)) is not None]
    return float(np.mean(times)) if times else 0.0


def upward_ranks(problem: Problem, v: CostView | None = None) -> dict[str, float]:
    """HEFT rank_u(i) = w̄_i + max over successors j of (c̄_ij + rank_u(j))."""
    v = v or problem.nominal()
    rank: dict[str, float] = {}
    for n in reversed(problem.order):
        rank[n] = _avg_time(problem, v, n) + max(
            (_avg_comm(problem, v, n, s) + rank[s] for s in problem.succs[n]), default=0.0
        )
    return rank


def optimistic_cost_table(problem: Problem, v: CostView | None = None) -> dict[tuple[str, str], float]:
    """PEFT OCT(t_i, p_k) = max_succ j min_{p_w} [OCT(t_j, p_w) + w(t_j, p_w) + c̄_ij·(p_w ≠ p_k)].

    w(t_j, p_w) is the fastest implementation of t_j on PE p_w. Defined for every PE t_i can use.
    """
    v = v or problem.nominal()
    oct_: dict[tuple[str, str], float] = {}
    for n in reversed(problem.order):
        pes = sorted({o.pe for o in problem.options[n]}, key=problem.pes.index)
        for pk in pes:
            worst = 0.0
            for s in problem.succs[n]:
                cbar = _avg_comm(problem, v, n, s)
                best = min(
                    oct_[(s, o.pe)] + v.time[(s, o.impl)] + (cbar if o.pe != pk else 0.0)
                    for o in problem.options[s]
                )
                worst = max(worst, best)
            oct_[(n, pk)] = worst
    return oct_


def _fits(problem: Problem, usage: dict[str, dict[str, float]], opt: Option, res: dict[str, float]) -> bool:
    cap = problem.platform.pe(opt.pe).capacity
    use = usage.get(opt.pe, {})
    return all(use.get(r, 0.0) + res.get(r, 0.0) <= c * (1 + 1e-9) for r, c in cap.items())


def _list_map(
    problem: Problem,
    rank: dict[str, float],
    extra: Callable[[str, Option], float],
    insertion: bool,
    fixed: dict[str, Option] | None = None,
) -> Plan:
    """List-schedule in rank order; `fixed` nodes keep their given option, the others take the best fitting one."""
    fixed = fixed or {}
    v = problem.nominal()
    topo = {n: i for i, n in enumerate(problem.order)}
    state = _State(problem, v, contention=True, insertion=insertion)
    usage: dict[str, dict[str, float]] = {}
    mapping: dict[str, Option] = {}
    order: list[str] = []
    remaining = {n: len(problem.preds[n]) for n in problem.order}
    ready = [n for n in problem.order if remaining[n] == 0]
    while ready:
        ready.sort(key=lambda n: (-round(rank[n], 9), topo[n]))  # ties (within float noise) by graph order
        n = ready.pop(0)
        best: tuple[tuple[int, float, float, int], Option] | None = None
        for i, o in enumerate([fixed[n]] if n in fixed else problem.options[n]):
            t = state.trial(n, o)
            if t is None:
                continue
            key = (0 if _fits(problem, usage, o, v.res[(n, o.impl)]) else 1, t[1] + extra(n, o),
                   v.energy[(n, o.impl)], i)
            if best is None or key < best[0]:
                best = (key, o)
        if best is None:
            raise ValueError(f"list scheduler: no reachable option for node {n}")
        o = best[1]
        state.place(n, o)
        mapping[n] = o
        order.append(n)
        u = usage.setdefault(o.pe, {})
        for r, val in v.res[(n, o.impl)].items():
            u[r] = u.get(r, 0.0) + val
        for s in problem.succs[n]:
            remaining[s] -= 1
            if remaining[s] == 0:
                ready.append(s)
    return Plan(mapping, order, insertion)


def heft(problem: Problem, insertion: bool = True) -> Plan:
    """HEFT: decreasing rank_u; each task to the option with the earliest finish time."""
    return _list_map(problem, upward_ranks(problem), lambda n, o: 0.0, insertion)


def peft(problem: Problem, insertion: bool = True) -> Plan:
    """PEFT: decreasing mean OCT; each task to the option minimising EFT + OCT(task, PE)."""
    oct_ = optimistic_cost_table(problem)
    rank = {n: float(np.mean([oct_[(n, p)] for p in {o.pe for o in problem.options[n]}])) for n in problem.order}
    return _list_map(problem, rank, lambda n, o: oct_[(n, o.pe)], insertion)


def heft_complete(problem: Problem, fixed: dict[str, Option], insertion: bool = True) -> Plan:
    """HEFT with some nodes' options fixed: the free nodes are mapped greedily around them."""
    return _list_map(problem, upward_ranks(problem), lambda n, o: 0.0, insertion, fixed)
