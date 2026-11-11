"""MAGELLAN-flat (Chatha & Vemuri 2001): steepest descent with fixing.

Each step tries every option for every still-free task, completes all other free tasks with capacity-aware HEFT
(fixed tasks keep their options and their resources stay reserved), and permanently fixes the (task, option) pair with
the best completed schedule length, provided it improves on the current best completion. Stops when no pair improves
or the budget ends. The paper's task-suitability metric is replaced by earliest finish time (inferred, as in the
group's notes). The very first completion (nothing fixed) is HEFT itself, so the result is never worse than HEFT.

Cost is O(n^2 K) HEFT completions; the budget cuts it off, returning the best completion found so far.
"""

from __future__ import annotations

import numpy as np

from ..fast import Fast
from ..instance import Instance
from ..sim import cap_excess
from .base import Result, Tracker, usage_of


def fast_excess(inst, opt) -> float:
    return cap_excess(inst, usage_of(inst, opt))



def solve_magellan(inst: Instance, fast: Fast, seed: int, budget_s: float) -> Result:
    tr = Tracker(inst, fast, budget_s)
    base, cur_mk = fast.heft()
    tr.offer_plan(base, cur_mk)
    fixed: dict[int, int] = {}
    # order in which tasks are considered: decreasing HEFT order (the plan's list order), as candidates for fixing
    free = list(base.order)
    n_comp = 1
    while free and not tr.expired():
        best = None
        for i in free:
            for k in inst.task_opts[i]:
                if tr.expired():
                    break
                trial_fixed = dict(fixed)
                trial_fixed[i] = k
                try:
                    plan, mk = fast.heft(trial_fixed)
                except ValueError:
                    continue
                n_comp += 1
                ex = fast_excess(inst, plan.opt)
                if ex > 0:
                    continue
                tr.offer(mk, 0.0, plan.opt, plan.order, True)
                if best is None or mk < best[0] - 1e-12:
                    best = (mk, i, k)
            if tr.expired():
                break
        if best is None or best[0] >= cur_mk - 1e-12:
            break
        cur_mk, i, k = best
        fixed[i] = k
        free.remove(i)
    return tr.result("magellan", completions=n_comp, fixed=len(fixed))
