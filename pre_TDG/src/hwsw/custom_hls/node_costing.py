"""The cost-map entry (`costs.nodes[]`) of one node."""

from hwsw.custom_hls.constants import HlsConstants
from hwsw.custom_hls.estimate_values import make_estimate, round_significant
from hwsw.custom_hls.hw_design_points import HardwareDesignPoint, hardware_design_points
from hwsw.custom_hls.loop_nest import LoopNest, parse_loop_nest
from hwsw.custom_hls.loop_nest_analysis import analyze_loop_nest
from hwsw.custom_hls.sw_time import software_time_s
from hwsw.custom_hls.templates import expand_template
from hwsw.platforms import Board


def cost_node(node: dict, board: Board, constants: HlsConstants) -> dict:
    if node["kind"] == "opaque":
        return _cost_opaque_node(node, constants)

    analysis = analyze_loop_nest(_loop_nest_of(node), constants)
    cost: dict = {
        "id": node["id"],
        "ops_estimated": analysis.ops_in_unit(node.get("ops_unit", "ops")),
    }
    if node["affinity"] != "hw_only":
        cost["sw"] = {
            "time_s": make_estimate(
                software_time_s(analysis, board, constants),
                constants.sw_uncertainty,
                "profile_model",
            )
        }
    if node["affinity"] != "sw_only":
        cost["hw"] = [
            _hardware_point_entry(point, constants)
            for point in hardware_design_points(analysis, board, constants)
        ]
    return cost


def _cost_opaque_node(node: dict, constants: HlsConstants) -> dict:
    """Opaque nodes carry their own software time; copy it into the cost map (TDG open item 2)."""
    opaque = node["opaque"]
    return {
        "id": node["id"],
        "sw": {
            "time_s": make_estimate(
                opaque["sw_time_s"], constants.opaque_uncertainty, opaque["source"]
            )
        },
    }


def _loop_nest_of(node: dict) -> LoopNest:
    if node["kind"] == "template":
        template = node["template"]
        return parse_loop_nest(expand_template(template["op"], template["params"]))
    return parse_loop_nest(node["loop_nest"])


def _hardware_point_entry(point: HardwareDesignPoint, constants: HlsConstants) -> dict:
    return {
        "point": point.point_id,
        "parallelism": point.parallelism,
        "time_s": make_estimate(point.time_s, constants.hw_uncertainty, "roofline"),
        "ii_s": round_significant(point.ii_s),
        "resources": point.resources,
    }
