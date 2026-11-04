"""R5: the TDG's platform exists, and every hardware design point fits on that board."""

from hwsw.platforms import Board
from hwsw.validation.issue import Issue, error


def check_platform(tdg: dict, board: Board) -> list[Issue]:
    if tdg["platform"] != board.id:
        return [
            error(
                "R5",
                "platform",
                f"TDG targets '{tdg['platform']}' but is being checked against board '{board.id}'",
            )
        ]
    if "costs" not in tdg:
        return []
    return [*_points_fit_on_board(tdg["costs"], board), *_interfaces_exist(tdg["costs"], board)]


def _points_fit_on_board(costs: dict, board: Board) -> list[Issue]:
    issues = []
    for cost in costs["nodes"]:
        for point in cost.get("hw", []):
            for resource, amount in point["resources"].items():
                capacity = board.fpga.capacity.get(resource, 0)
                if amount > capacity:
                    issues.append(
                        error(
                            "R5",
                            f"costs node {cost['id']} hw point {point['point']}",
                            f"needs {amount} {resource} but board '{board.id}' has {capacity}",
                        )
                    )
    return issues


def _interfaces_exist(costs: dict, board: Board) -> list[Issue]:
    return [
        error(
            "R5",
            f"costs edge {cost['id']}",
            f"interface '{cost['interface']}' is not defined by board '{board.id}' "
            f"(available: {', '.join(sorted(board.interfaces))})",
        )
        for cost in costs.get("edges", [])
        if "interface" in cost and cost["interface"] not in board.interfaces
    ]
