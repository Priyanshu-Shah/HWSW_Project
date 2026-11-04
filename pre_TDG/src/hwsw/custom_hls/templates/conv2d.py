"""conv2d: input [H, W, C_in] (HWC), weights [C_out, C_in/groups, K, K], output [Ho, Wo, C_out]."""

from hwsw.custom_hls.templates.batching import batch_loops, batch_shape
from hwsw.custom_hls.templates.template_errors import TemplateParameterError


def expand(params: dict) -> dict:
    height, width = params["H"], params["W"]
    in_channels, out_channels = params["C_in"], params["C_out"]
    kernel, stride = params["K"], params["stride"]
    padding, groups = params.get("padding", 0), params.get("groups", 1)
    batch, dtype = params.get("batch", 1), params["dtype"]

    if in_channels % groups or out_channels % groups:
        raise TemplateParameterError(
            f"conv2d: groups={groups} must divide C_in={in_channels} and C_out={out_channels}"
        )
    out_height = (height + 2 * padding - kernel) // stride + 1
    out_width = (width + 2 * padding - kernel) // stride + 1
    if out_height < 1 or out_width < 1:
        raise TemplateParameterError(
            f"conv2d: kernel K={kernel} does not fit the {height}x{width} input (padding={padding})"
        )

    return {
        "arrays": {
            "input": {
                "shape": [*batch_shape(batch), height, width, in_channels],
                "dtype": dtype,
                "dir": "in",
            },
            "weights": {
                "shape": [out_channels, in_channels // groups, kernel, kernel],
                "dtype": dtype,
                "dir": "in",
                "const": True,
            },
            "output": {
                "shape": [*batch_shape(batch), out_height, out_width, out_channels],
                "dtype": dtype,
                "dir": "out",
            },
        },
        "loops": [
            *batch_loops(batch),
            {"var": "oc", "range": out_channels},
            {"var": "oy", "range": out_height},
            {"var": "ox", "range": out_width},
            {"var": "ic", "range": in_channels // groups},
            {"var": "ky", "range": kernel},
            {"var": "kx", "range": kernel},
        ],
        "body": {"mac": 1},
        "access": {
            "input": {"pattern": "sliding_window", "reuse": ["ky", "kx"]},
            "weights": {"pattern": "reuse"},
            "output": {"pattern": "streaming"},
        },
        "hints": {"unrollable": ["ky", "kx"]},
    }
