"""Templates: standard operations, each expanded into a D1 loop-nest description."""

from collections.abc import Callable

from hwsw.custom_hls.templates import conv2d, elementwise, gemm, memcpy, pool, reduce
from hwsw.custom_hls.templates.template_errors import TemplateParameterError

_EXPANDERS: dict[str, Callable[[dict], dict]] = {
    "conv2d": conv2d.expand,
    "gemm": gemm.expand,
    "pool": pool.expand,
    "elementwise": elementwise.expand,
    "reduce": reduce.expand,
    "memcpy": memcpy.expand,
}

TEMPLATE_OPS = tuple(_EXPANDERS)


def expand_template(op: str, params: dict) -> dict:
    """The D1 loop-nest description (a dict) for template `op` with `params`."""
    if op not in _EXPANDERS:
        raise TemplateParameterError(
            f"unknown template op '{op}'; known: {', '.join(TEMPLATE_OPS)}"
        )
    return _EXPANDERS[op](params)


__all__ = ["TEMPLATE_OPS", "TemplateParameterError", "expand_template"]
