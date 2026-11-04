"""pool: max or average pooling. Input [H, W, C], output [Ho, Wo, C]."""

from hwsw.custom_hls.templates.batching import batch_loops, batch_shape
from hwsw.custom_hls.templates.template_errors import TemplateParameterError


def expand(params: dict) -> dict:
    height, width, channels = params["H"], params["W"], params["C"]
    kernel, stride = params["K"], params["stride"]
    batch, dtype = params.get("batch", 1), params["dtype"]

    out_height = (height - kernel) // stride + 1
    out_width = (width - kernel) // stride + 1
    if out_height < 1 or out_width < 1:
        raise TemplateParameterError(
            f"pool: window K={kernel} does not fit the {height}x{width} input"
        )

    window_op = "cmp" if params["mode"] == "max" else "add"
    return {
        "arrays": {
            "input": {
                "shape": [*batch_shape(batch), height, width, channels],
                "dtype": dtype,
                "dir": "in",
            },
            "output": {
                "shape": [*batch_shape(batch), out_height, out_width, channels],
                "dtype": dtype,
                "dir": "out",
            },
        },
        "loops": [
            *batch_loops(batch),
            {"var": "oy", "range": out_height},
            {"var": "ox", "range": out_width},
            {"var": "c", "range": channels},
            {"var": "ky", "range": kernel},
            {"var": "kx", "range": kernel},
        ],
        "body": {window_op: 1},
        "access": {
            "input": {"pattern": "sliding_window", "reuse": ["ky", "kx"]},
            "output": {"pattern": "streaming"},
        },
        "hints": {"unrollable": ["ky", "kx"]},
    }
