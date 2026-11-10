"""MAGELLAN-flat: steepest descent with permanent fixing and greedy re-completion of the free tasks.

After Chatha & Vemuri, "MAGELLAN: multiway hardware-software partitioning and scheduling for latency minimization
of hierarchical control-dataflow task graphs", CODES 2001 (§5, leaf mode), flattened to our DAG model:
- the "scheduler" that maps every free task is HEFT with capacity checks (`heft_complete`), replacing the paper's
  suitability/threshold rules, which are defined only in an earlier paper (inferred substitution);
- a move forces one free task onto one of its (PE, implementation) options; it is scored by re-completing all
  other free tasks with the scheduler and simulating;
- the best improving move is applied and that task is FIXED for the rest of the run; stop when nothing improves.
Hierarchical/loop/call moves of the original (unrolling, pipelining, sharing) are out of scope for flat DAGs.
Cost: O(n^2 * K) schedule completions for n tasks with K options each.
"""

from __future__ import annotations

from estimate.annotate import Option, Problem
from sim.evaluate import evaluate
from sim.scheduler import InvalidMapping
from solvers.heuristics.baselines import penalised
from solvers.heuristics.list_sched import Plan, heft_complete


def magellan(problem: Problem, insertion: bool = True, max_fixes: int | None = None) -> Plan:
    def score(fixed: dict[str, Option]) -> tuple[tuple[int, float], Plan] | None:
        try:
            pl = heft_complete(problem, fixed, insertion)
            ev = evaluate(problem, pl.mapping, priority=pl.priority, insertion=pl.insertion)
        except (InvalidMapping, ValueError):
            return None
        return (0 if ev.feasible else 1, ev.objective if ev.feasible else penalised(problem, ev)), pl

    fixed: dict[str, Option] = {}
    cur = score(fixed)
    if cur is None:
        raise ValueError("MAGELLAN: no completion exists even with nothing fixed")
    cur_key, cur_plan = cur
    limit = max_fixes if max_fixes is not None else len(problem.order)
    for _ in range(limit):
        best = None
        for n in problem.order:
            if n in fixed:
                continue
            for o in problem.options[n]:
                r = score({**fixed, n: o})
                if r is not None and (best is None or r[0] < best[0]):
                    best = (r[0], r[1], n, o)
        if best is None or not best[0] < cur_key:
            break
        cur_key, cur_plan = best[0], best[1]
        fixed[best[2]] = best[3]
    return cur_plan
