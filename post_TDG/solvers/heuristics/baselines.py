"""Classical baselines: greedy earliest-finish-time, all-SW, all-HW, simulated annealing, exhaustive search."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np

from estimate.annotate import Option, Problem
from sim.evaluate import Evaluation, evaluate
from sim.scheduler import _State

MappingD = dict[str, Option]


def _fits(problem: Problem, usage: dict[str, dict[str, float]], node: str, opt: Option, res: dict) -> bool:
    cap = problem.platform.pe(opt.pe).capacity
    use = usage.get(opt.pe, {})
    return all(use.get(r, 0.0) + res.get(r, 0.0) <= c * (1 + 1e-9) for r, c in cap.items())


def greedy_eft(problem: Problem, prefer: Callable[[Option], bool] | None = None) -> MappingD:
    """HEFT-style: nodes in decreasing mean upward rank; each takes the option with the earliest finish.

    `prefer` restricts choices to options it accepts, falling back to all options for nodes with none.
    Options that would exceed a PE capacity are skipped unless nothing else is legal.
    """
    v = problem.nominal()
    mean_t = {n: float(np.mean([v.time[(n, o.impl)] for o in problem.options[n]])) for n in problem.order}
    rank: dict[str, float] = {}
    for n in reversed(problem.order):
        rank[n] = mean_t[n] + max((rank[s] for s in problem.succs[n]), default=0.0)
    topo = {n: i for i, n in enumerate(problem.order)}

    state = _State(problem, v, contention=True)
    usage: dict[str, dict[str, float]] = {}
    mapping: MappingD = {}
    remaining = {n: len(problem.preds[n]) for n in problem.order}
    ready = [n for n in problem.order if remaining[n] == 0]
    while ready:
        ready.sort(key=lambda n: (-rank[n], topo[n]))
        n = ready.pop(0)
        opts = [o for o in problem.options[n] if prefer is None or prefer(o)] or list(problem.options[n])
        best: tuple[tuple[float, float, int, int], Option] | None = None
        for i, o in enumerate(opts):
            t = state.trial(n, o)
            if t is None:
                continue
            fits = _fits(problem, usage, n, o, v.res[(n, o.impl)])
            key = (0 if fits else 1, t[1], v.energy[(n, o.impl)], i)
            if best is None or key < best[0]:
                best = (key, o)
        if best is None:
            raise ValueError(f"greedy: no reachable option for node {n}")
        o = best[1]
        state.place(n, o)
        mapping[n] = o
        u = usage.setdefault(o.pe, {})
        for r, val in v.res[(n, o.impl)].items():
            u[r] = u.get(r, 0.0) + val
        for s in problem.succs[n]:
            remaining[s] -= 1
            if remaining[s] == 0:
                ready.append(s)
    return mapping


def all_sw(problem: Problem) -> MappingD:
    """Everything on CPU cores (greedy across cores); nodes with no CPU option keep their best option."""
    return greedy_eft(problem, prefer=lambda o: problem.pe_kind(o.pe) == "cpu")


def all_hw(problem: Problem) -> MappingD:
    """Every node that has a non-CPU option uses one; the rest stay on CPUs."""
    return greedy_eft(problem, prefer=lambda o: problem.pe_kind(o.pe) != "cpu")


def penalised(problem: Problem, ev: Evaluation, weight: float = 10.0) -> float:
    return ev.objective + weight * sum(max(0.0, v.relative) for v in ev.violations)


def simulated_annealing(
    problem: Problem,
    seed: int = 0,
    n_iter: int = 3000,
    start: MappingD | None = None,
    t_final_ratio: float = 1e-3,
) -> MappingD:
    """Single-node option moves, Metropolis acceptance, geometric cooling; returns the best mapping seen."""
    rng = np.random.default_rng(seed)
    movable = [n for n in problem.order if len(problem.options[n]) > 1]
    cur = dict(start or greedy_eft(problem))
    cur_score = penalised(problem, evaluate(problem, cur))
    best, best_score = dict(cur), cur_score
    if not movable:
        return best
    t0 = max(0.1 * cur_score, 1e-6)
    alpha = t_final_ratio ** (1.0 / max(n_iter, 1))
    temp = t0
    for _ in range(n_iter):
        n = movable[int(rng.integers(len(movable)))]
        choices = [o for o in problem.options[n] if o != cur[n]]
        cand = dict(cur)
        cand[n] = choices[int(rng.integers(len(choices)))]
        try:
            score = penalised(problem, evaluate(problem, cand))
        except ValueError:  # unreachable route
            temp *= alpha
            continue
        if score <= cur_score or rng.random() < math.exp(-(score - cur_score) / temp):
            cur, cur_score = cand, score
            if score < best_score - 1e-12:
                best, best_score = dict(cand), score
        temp *= alpha
    return best


def exhaustive(problem: Problem, contention: bool = True, limit: int = 200_000) -> tuple[MappingD, Evaluation]:
    """Best mapping by penalised objective over all option combinations (list-scheduled). Small graphs only."""
    sizes = [len(problem.options[n]) for n in problem.order]
    total = math.prod(sizes)
    if total > limit:
        raise ValueError(f"{total} combinations exceed limit {limit}")
    best: tuple[float, MappingD, Evaluation] | None = None
    for combo in itertools.product(*(problem.options[n] for n in problem.order)):
        m = dict(zip(problem.order, combo, strict=True))
        try:
            ev = evaluate(problem, m, contention=contention)
        except ValueError:
            continue
        score = penalised(problem, ev)
        if best is None or score < best[0] - 1e-12:
            best = (score, m, ev)
    if best is None:
        raise ValueError("no mapping with complete routes exists")
    return best[1], best[2]
