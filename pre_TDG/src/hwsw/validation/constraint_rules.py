"""R10-R11: the objective is latency only (D3), and the deadline must be reachable per frame."""

from hwsw.validation.issue import Issue, error, warning

MVP_OBJECTIVE = {"latency": 1.0}


def check_objective_and_deadline(tdg: dict) -> list[Issue]:
    return [*_objective_is_latency_only(tdg), *_deadline_fits_in_period(tdg)]


def _objective_is_latency_only(tdg: dict) -> list[Issue]:
    if tdg["objective"] == MVP_OBJECTIVE:
        return []
    return [
        error(
            "R10",
            "objective",
            f"objective must be {MVP_OBJECTIVE} in the MVP (D3), got {tdg['objective']}",
        )
    ]


def _deadline_fits_in_period(tdg: dict) -> list[Issue]:
    deadline = tdg.get("constraints", {}).get("deadline_s")
    period = tdg["arrival"].get("period_s")
    if deadline is None or period is None or deadline <= period:
        return []
    return [
        warning(
            "R11",
            "constraints.deadline_s",
            f"deadline {deadline} s is longer than the period {period} s; with frames processed "
            f"one after another (D3) the rate cannot be held. Use deadline_s <= period_s",
        )
    ]
