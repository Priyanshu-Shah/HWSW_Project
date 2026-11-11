"""Kernighan–Lin / Fiduccia–Mattheyses and tabu search over single-node option moves.

Both search the same neighbourhood as our simulated annealing: move one node to another legal
(PE, implementation) option. A move's gain is computed by re-running the list-scheduling simulator
(Mann, Orbán, Farkas 2007, Alg. 5: "gain = change of the scheduled objective"), so precedence, contention
and resources are all accounted for.

KL follows Mann, Orbán & Farkas, "Evaluating the Kernighan–Lin heuristic for hardware/software
partitioning", IJAMCS 17(2), 2007:
- passes of moves; each move takes the best available candidate even if it worsens the cost, then locks it;
- a *permissive* cost: constraint violations are penalised by a 4th-order polynomial up to a threshold q,
  beyond which the move is forbidden; only feasible solutions can become the best;
- LIFO tie-breaking among near-best moves (within (1-ε) of the best improvement, ε = 0.95), preferring nodes
  whose neighbours moved most recently;
- dynamic locking: a moved node's graph neighbours are unlocked again, up to `max_moves_per_node` per pass
  (Mann's KL1 uses about 5; classic KL is 1);
- KL2 = several restarts from random starts; the best result is kept.

Tabu search follows the structure in Eles, Peng, Kuchcinski, Doboli, DAES 1997 (best non-tabu move each
iteration, tenure on recently moved nodes, aspiration when a move beats the global best, frequency-based
diversification). The exact tenure/diversification constants of that paper are not reproduced (unverified);
ours are parameters.
"""

from __future__ import annotations

import math

import numpy as np

from estimate.annotate import Option, Problem
from sim.evaluate import Evaluation, evaluate
from sim.scheduler import InvalidMapping
from solvers.heuristics.baselines import greedy_eft, penalised

MappingD = dict[str, Option]
_INF = float("inf")


def _neighbours(problem: Problem) -> dict[str, set[str]]:
    return {n: set(problem.preds[n]) | set(problem.succs[n]) for n in problem.order}


def _eval(problem: Problem, m: MappingD) -> Evaluation | None:
    try:
        return evaluate(problem, m)
    except InvalidMapping:
        return None


def kl_cost(ev: Evaluation, q: float = 2.0, weight: float = 10.0) -> float:
    """Permissive cost: objective + weight·Σ((r-1)/(q-1))^4 for each violated constraint with ratio r = value/limit;
    +inf if any ratio exceeds q."""
    pen = 0.0
    for v in ev.violations:
        r = 1.0 + v.relative
        if r > q:
            return _INF
        pen += ((r - 1.0) / (q - 1.0)) ** 4
    return ev.objective + weight * pen


def _surrogate(problem: Problem, m: MappingD, n: str, o: Option) -> float:
    """Cheap local estimate of a move's effect: change in node time plus change in incident transfer times."""
    v = problem.nominal()
    d = v.time[(n, o.impl)] - v.time[(n, m[n].impl)]
    for a, b in [(p, n) for p in problem.preds[n]] + [(n, s) for s in problem.succs[n]]:
        nb = v.bytes[(a, b)]
        old = problem.comm(m[a].pe, m[b].pe, nb)
        new = problem.comm(o.pe if a == n else m[a].pe, o.pe if b == n else m[b].pe, nb)
        d += (new.time_ms if new else 1e9) - (old.time_ms if old else 1e9)
    return d


def _kl_pass(
    problem: Problem,
    cur: MappingD,
    max_moves_per_node: int,
    candidate_limit: int | None,
    eps: float,
    nbrs: dict[str, set[str]],
) -> tuple[MappingD, float, bool]:
    """One KL/FM pass from `cur`; returns the best mapping seen in the pass (feasible preferred)."""
    movable = [n for n in problem.order if len(problem.options[n]) > 1]
    state = dict(cur)
    ev = _eval(problem, state)
    state_cost = kl_cost(ev) if ev else _INF
    best = (0 if ev and ev.feasible else 1, state_cost if ev and ev.feasible else penalised(problem, ev) if ev else _INF,
            dict(state))
    unlocked = {n: True for n in movable}
    moves = {n: 0 for n in movable}
    last_move: dict[str, int] = {}
    t = 0
    while any(unlocked.values()):
        cands: list[tuple[float, str, Option, Evaluation]] = []
        for n in movable:
            if not unlocked[n]:
                continue
            opts = [o for o in problem.options[n] if o != state[n]]
            if candidate_limit is not None:
                opts = sorted(opts, key=lambda o: _surrogate(problem, state, n, o))[:candidate_limit]
            for o in opts:
                cand = dict(state)
                cand[n] = o
                e = _eval(problem, cand)
                if e is None:
                    continue
                c = kl_cost(e)
                if c < _INF:
                    cands.append((c, n, o, e))
        if not cands:
            break
        top = min(c for c, *_ in cands)
        improvement = state_cost - top
        window = top + (1.0 - eps) * max(improvement, 0.0) + 1e-12
        near = [x for x in cands if x[0] <= window]
        topo = {n: i for i, n in enumerate(problem.order)}
        c, n, o, e = max(
            near, key=lambda x: (max((last_move.get(m, -1) for m in nbrs[x[1]]), default=-1), -x[0], -topo[x[1]])
        )
        state[n] = o
        state_cost = c
        t += 1
        last_move[n] = t
        moves[n] += 1
        unlocked[n] = False
        if max_moves_per_node > 1:  # dynamic locking: neighbours become movable again
            for m in nbrs[n]:
                if m in unlocked and moves[m] < max_moves_per_node:
                    unlocked[m] = True
        key = (0 if e.feasible else 1, e.objective if e.feasible else penalised(problem, e))
        if key < best[:2]:
            best = (*key, dict(state))
    return best[2], best[1], best[0] == 0


def kernighan_lin(
    problem: Problem,
    start: MappingD | None = None,
    max_passes: int = 20,
    max_moves_per_node: int = 1,
    candidate_limit: int | None = None,
    restarts: int = 0,
    seed: int = 0,
    eps: float = 0.95,
) -> MappingD:
    """KL1 (restarts=0) or KL2 (restarts>0, extra random starts). Returns the best mapping found."""
    rng = np.random.default_rng(seed)
    nbrs = _neighbours(problem)
    starts = [dict(start or greedy_eft(problem))]
    for _ in range(restarts):
        starts.append({n: problem.options[n][int(rng.integers(len(problem.options[n])))] for n in problem.order})
    best: tuple[int, float, MappingD] | None = None
    for s in starts:
        cur = s
        ev = _eval(problem, cur)
        cur_key = (0 if ev and ev.feasible else 1, (ev.objective if ev.feasible else penalised(problem, ev)) if ev
                   else _INF)
        for _ in range(max_passes):
            cand, val, feas = _kl_pass(problem, cur, max_moves_per_node, candidate_limit, eps, nbrs)
            key = (0 if feas else 1, val)
            improved = key[0] < cur_key[0] or (key[0] == cur_key[0] and key[1] < cur_key[1] - 1e-12)
            if not improved:
                break
            cur, cur_key = cand, key
        if best is None or cur_key < best[:2]:
            best = (*cur_key, cur)
    assert best is not None
    return best[2]


def tabu_search(
    problem: Problem,
    start: MappingD | None = None,
    n_iter: int = 400,
    tenure: int | None = None,
    neighbourhood: int | None = None,
    freq_weight: float = 0.005,
    seed: int = 0,
) -> MappingD:
    """Tabu search over single-node option moves, with the selection rule of Wiangtong, Cheung & Luk (DAES 2002,
    Fig. 4):

    A. if some non-tabu move improves the current solution, take the best one by raw cost (no penalty);
    B. otherwise take the non-tabu move with the lowest cost + frequency penalty, where the penalty is relative:
       freq_weight * |current cost| * (times the node has moved)  (their Q = 80 ns on 6k-30k ns makespans);
    C. if every move is tabu, take the move of the node whose tabu status expires first (the oldest).
    Aspiration: a tabu move that beats the best solution so far is allowed. Tenure defaults to sqrt(movable) + 1
    clamped to [7, 20] (Reeves' guidance quoted by Wiangtong); 400 iterations as in Wiangtong.
    """
    rng = np.random.default_rng(seed)
    movable = [n for n in problem.order if len(problem.options[n]) > 1]
    cur = dict(start or greedy_eft(problem))
    ev = _eval(problem, cur)
    if ev is None or not movable:
        return cur

    def key(e: Evaluation) -> tuple[int, float]:
        return (0 if e.feasible else 1, e.objective if e.feasible else penalised(problem, e))

    best_key, best = key(ev), dict(cur)
    cur_score = penalised(problem, ev)
    tenure = tenure if tenure is not None else min(20, max(7, int(round(math.sqrt(len(movable)))) + 1))
    tabu_until = {n: -1 for n in movable}
    freq = {n: 0 for n in movable}
    all_moves = [(n, o) for n in movable for o in problem.options[n]]
    for it in range(n_iter):
        moves = [(n, o) for n, o in all_moves if o != cur[n]]
        if neighbourhood is not None and len(moves) > neighbourhood:
            idx = rng.choice(len(moves), size=neighbourhood, replace=False)
            moves = [moves[i] for i in sorted(idx)]
        free: list[tuple[float, str, Option, Evaluation]] = []  # non-tabu or aspirating
        tabu: list[tuple[int, float, str, Option, Evaluation]] = []
        for n, o in moves:
            cand = dict(cur)
            cand[n] = o
            e = _eval(problem, cand)
            if e is None:
                continue
            sc = penalised(problem, e)
            if tabu_until[n] >= it and not key(e) < best_key:
                tabu.append((tabu_until[n], sc, n, o, e))
            else:
                free.append((sc, n, o, e))
        improving = [x for x in free if x[0] < cur_score - 1e-12]
        if improving:  # A
            sc, n, o, e = min(improving, key=lambda x: x[0])
        elif free:  # B
            pen = freq_weight * abs(cur_score)
            sc, n, o, e = min(free, key=lambda x: x[0] + pen * freq[x[1]])
        elif tabu:  # C
            _, sc, n, o, e = min(tabu, key=lambda x: (x[0], x[1]))
        else:
            continue
        cur[n] = o
        cur_score = sc
        tabu_until[n] = it + tenure
        freq[n] += 1
        if key(e) < best_key:
            best_key, best = key(e), dict(cur)
    return best
