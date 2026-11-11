"""Choose the feedback loop from the lower bound and the best mapping found.

fb1 (back to task generation): the lower bound already violates a constraint, or a structural problem
     (no compatible PE, a task too large for every PE) means no mapping can succeed.
fb2 (back to Bind/Partition): the lower bound passes but the best mapping found still fails.
none: the best mapping meets every constraint at nominal costs.
"""

from __future__ import annotations

from schemas.result import Diagnostics, LowerBound, Violation


def _fmt(v: Violation) -> str:
    return f"{v.constraint}: {v.value:.4g} vs limit {v.limit:.4g} (excess {v.excess:.4g}, {100 * v.relative:.1f}%)"


def diagnose(
    lb: LowerBound,
    best_violations: list[Violation] | None,
    relaxation: list[Violation] | None = None,
    feasible_fraction: float | None = None,
) -> Diagnostics:
    """`best_violations` is None when no mapping was found at all."""
    if lb.structural or lb.violations:
        reasons = [f"structural: {s}" for s in lb.structural]
        for v in lb.violations:
            tag = "even at optimistic costs" if v.constraint in lb.certain else "at nominal costs only"
            reasons.append(f"lower bound violates {_fmt(v)} {tag}; no mapping on this platform subset can fix it")
        return Diagnostics(loop="fb1", reasons=reasons, lower_bound=lb, violations=best_violations or [],
                           relaxation=relaxation)
    if best_violations is None or best_violations:
        reasons = ["lower bound meets all constraints, but the best mapping found does not"]
        reasons += [f"best mapping violates {_fmt(v)}" for v in best_violations or []]
        reasons += [f"minimal relaxation: {_fmt(v)}" for v in relaxation or []]
        return Diagnostics(loop="fb2", reasons=reasons, lower_bound=lb, violations=best_violations or [],
                           relaxation=relaxation)
    reasons = ["best mapping meets all constraints at nominal costs"]
    if feasible_fraction is not None and feasible_fraction < 1.0:
        reasons.append(f"only {100 * feasible_fraction:.1f}% of Monte Carlo samples meet all constraints")
    return Diagnostics(loop="none", reasons=reasons, lower_bound=lb, violations=[], relaxation=relaxation)
