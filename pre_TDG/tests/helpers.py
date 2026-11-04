"""Small helpers and example data shared by the tests."""

from hwsw.custom_hls.constants import HlsConstants
from hwsw.custom_hls.loop_nest import parse_loop_nest
from hwsw.custom_hls.loop_nest_analysis import analyze_loop_nest

# The worked example of TDG/decisions.md D1: a 3x3 convolution over a 2048x2048 int16 image.
CONV_3X3_2048 = {
    "arrays": {
        "img": {"shape": [2048, 2048], "dtype": "int16", "dir": "in"},
        "kernel": {"shape": [3, 3], "dtype": "int16", "dir": "in", "const": True},
        "out": {"shape": [2046, 2046], "dtype": "int16", "dir": "out"},
    },
    "loops": [
        {"var": "y", "range": 2046},
        {"var": "x", "range": 2046},
        {"var": "ky", "range": 3},
        {"var": "kx", "range": 3},
    ],
    "body": {"mac": 1},
    "access": {"img": {"pattern": "sliding_window", "reuse": ["ky", "kx"]}},
    "control": {"branches": 0},
    "hints": {"unrollable": ["ky", "kx"]},
}


def analyze(raw_loop_nest: dict, constants: HlsConstants | None = None):
    return analyze_loop_nest(parse_loop_nest(raw_loop_nest), constants or HlsConstants())


def rules_of(report) -> set[str]:
    return {issue.rule for issue in report.issues}


def find_node(tdg: dict, node_id: str) -> dict:
    return next(node for node in tdg["nodes"] if node["id"] == node_id)


def find_node_cost(tdg: dict, node_id: str) -> dict:
    return next(cost for cost in tdg["costs"]["nodes"] if cost["id"] == node_id)
