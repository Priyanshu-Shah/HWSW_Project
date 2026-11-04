"""elementwise: the same small computation on every element (activation, scale, add, normalize)."""


def expand(params: dict) -> dict:
    elements, dtype = params["elements"], params["dtype"]
    input_names = [f"in{index}" for index in range(params["inputs"])]
    return {
        "arrays": {
            **{name: {"shape": [elements], "dtype": dtype, "dir": "in"} for name in input_names},
            "out": {"shape": [elements], "dtype": dtype, "dir": "out"},
        },
        "loops": [{"var": "i", "range": elements}],
        "body": {"add": params["ops_per_element"]},  # the cheapest arithmetic op stands in for any
        "access": {name: {"pattern": "streaming"} for name in [*input_names, "out"]},
    }
