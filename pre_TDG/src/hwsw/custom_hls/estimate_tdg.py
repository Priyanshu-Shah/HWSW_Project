"""graph-only TDG + board  ->  the same TDG with its cost map (`costs`)."""

import copy

from hwsw.custom_hls.constants import HlsConstants, load_constants
from hwsw.custom_hls.node_costing import cost_node
from hwsw.custom_hls.templates import TemplateParameterError
from hwsw.platforms import Board


class CostEstimationError(Exception):
    """One or more nodes could not be costed. `problems` has one readable line per node."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("\n".join(problems))


def estimate_costs(tdg: dict, board: Board, constants: HlsConstants | None = None) -> dict:
    """A copy of `tdg` with `costs` filled in. `tdg` should already pass the validator."""
    if tdg["platform"] != board.id:
        raise CostEstimationError(
            [f"TDG targets platform '{tdg['platform']}' but was given board '{board.id}'"]
        )
    constants = constants or load_constants()

    node_costs, problems = [], []
    for node in tdg["nodes"]:
        try:
            node_costs.append(cost_node(node, board, constants))
        except TemplateParameterError as bad_parameters:
            problems.append(f"node {node['id']}: {bad_parameters}")
    if problems:
        raise CostEstimationError(problems)

    costed = copy.deepcopy(tdg)
    costed["costs"] = {"nodes": node_costs, "edges": []}
    return costed
