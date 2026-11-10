"""Logic-based Benders decomposition (LBBD) for joint mapping + scheduling, latency objective.

After Hooker (Oper. Res. 2007) and Emeretlis et al. (ACM TECS 2016 / TODAES 2018): the master chooses the mapping,
the subproblem schedules it.
  master:     the mapping relaxation (`relaxation.py`: per-PE load, communication-aware paths, shared-bus load,
              capacity, deadlines) plus Benders cuts, solved by HiGHS. Its optimum is a valid lower bound.
  subproblem: CP-SAT with the mapping fixed and bus contention modelled, i.e. the exact schedule of that mapping.
  cut:        no-good  C >= S_k - (S_k - LB_k) * (#tasks whose option differs from mapping k), where S_k is the
              subproblem's lower bound and LB_k the master bound before the cut (valid for every mapping). An
              infeasible subproblem (deadline) excludes the mapping outright.
Stops when the master bound reaches the best schedule (optimal), the master runs out of mappings, or time runs out
(then the result carries both the best schedule and the bound). The final mapping is re-evaluated by the simulator.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

from estimate.annotate import Option, Problem
from sim import evaluate
from solvers.cp import solve_cp
from solvers.exact.relaxation import RelaxationModel


@dataclass
class LbbdOutcome:
    status: str  # optimal | feasible | infeasible | time_limit (no schedule found)
    mapping: dict[str, Option] | None
    priority: list[str] | None
    makespan: float | None  # simulator makespan of the best mapping (exact durations, bus contention)
    bound: float | None  # valid lower bound on the optimal makespan
    iterations: int
    runtime_s: float
    history: list[tuple[float, float, float | None]] = field(default_factory=list)  # (time s, bound, best)
    notes: list[str] = field(default_factory=list)

    @property
    def gap(self) -> float | None:
        if self.makespan is None or self.bound is None or self.makespan <= 0:
            return None
        return max(0.0, (self.makespan - self.bound) / self.makespan)


def lbbd(
    problem: Problem,
    time_limit_s: float = 60.0,
    sub_time_s: float = 10.0,
    master_time_s: float = 5.0,
    first_master_share: float = 0.4,
    workers: int = 8,
    rel_tol: float = 1e-3,  # 0.1%: CP-SAT schedules in whole µs, so finer proofs are not meaningful
    max_iterations: int = 1000,
    seed: int = 0,
) -> LbbdOutcome:
    t0 = time.perf_counter()
    if set(problem.weights) != {"latency"}:
        raise ValueError("this method minimises the makespan; use CP-SAT for other objectives")
    master = RelaxationModel(problem, integer=True)
    best: tuple[float, dict[str, Option], list[str]] | None = None
    bound, history, notes = 0.0, [], []

    def left() -> float:
        return time_limit_s - (time.perf_counter() - t0)

    it = 0
    while it < max_iterations and left() > 0.5:
        it += 1
        # capped: on hard instances HiGHS cannot close the master, but its incumbent mapping is still worth scheduling.
        # The first solve gets a larger share: its mapping is usually the best one LBBD will see.
        cap = max(master_time_s, first_master_share * time_limit_s) if it == 1 else master_time_s
        mr = master.solve(time_limit_s=max(1.0, min(cap, left() - 1.0)), seed=seed)
        if mr.status == "infeasible":
            if best is None:
                return LbbdOutcome("infeasible", None, None, None, None, it, time.perf_counter() - t0, history,
                                   ["master infeasible: no mapping meets the constraints" if it == 1 else
                                    "every mapping meeting the relaxation was scheduled and failed the deadline"])
            bound = best[0]  # no mapping left: the incumbent is optimal
            break
        if mr.bound_ms is not None:
            bound = max(bound, mr.bound_ms)
        if best is not None and bound >= best[0] * (1 - rel_tol):
            break
        if mr.mapping is None:  # master hit the time limit without a mapping
            break
        m = mr.mapping
        fixed = dataclasses.replace(problem, options={t: [m[t]] for t in problem.order}, _norms=None)
        sub = solve_cp(fixed, time_limit_s=max(1.0, min(sub_time_s, left())), workers=workers, seed=seed,
                       contention=True)
        if sub.status == "infeasible":
            master.add_exclusion(m)  # this mapping cannot meet the constraints
        elif sub.mapping is not None:
            ev = evaluate(problem, m, priority=sub.priority(fixed))
            if ev.feasible and (best is None or ev.sim.makespan < best[0]):
                best = (ev.sim.makespan, m, sub.priority(fixed))
            s_lo = sub.makespan if sub.status == "optimal" else sub.bound * fixed.norms["latency"]
            # CP-SAT rounds each duration to whole µs: a margin of 1 µs per task keeps the cut below the exact optimum
            margin = 1e-3 * len(problem.order)
            master.add_nogood(m, value=max(bound, min(s_lo, ev.sim.makespan) - margin), floor=bound)
        else:  # subproblem timed out without a schedule: no valid cut, so stop
            notes.append("subproblem found no schedule within its time limit")
            break
        history.append((time.perf_counter() - t0, bound, best[0] if best else None))
    rt = time.perf_counter() - t0
    if best is None:
        return LbbdOutcome("time_limit", None, None, None, bound or None, it, rt, history, notes)
    status = "optimal" if bound >= best[0] * (1 - rel_tol) else "feasible"
    return LbbdOutcome(status, best[1], best[2], best[0], min(bound, best[0]), it, rt, history, notes)
