"""reduce: combine `elements` values into `outputs` results (sum, mean, max or min)."""

import math

from hwsw.custom_hls.templates.template_errors import TemplateParameterError


def expand(params: dict) -> dict:
    elements, dtype = params["elements"], params["dtype"]
    outputs = params.get("outputs", 1)
    if outputs > elements:
        raise TemplateParameterError(f"reduce: outputs={outputs} exceeds elements={elements}")

    combine_op = "add" if params["mode"] in ("sum", "mean") else "cmp"
    return {
        "arrays": {
            "values": {"shape": [elements], "dtype": dtype, "dir": "in"},
            "result": {"shape": [outputs], "dtype": dtype, "dir": "out"},
        },
        "loops": [
            {"var": "o", "range": outputs},
            {"var": "i", "range": math.ceil(elements / outputs)},
        ],
        "body": {combine_op: 1},
        "access": {"values": {"pattern": "streaming"}, "result": {"pattern": "streaming"}},
    }
