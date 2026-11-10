"""Safe evaluation of cost expressions over workload parameters.

Allowed: numbers, workload parameter names, + - * / // % **, unary +/-, parentheses and the
functions in FUNCTIONS. Anything else (attributes, subscripts, other calls) is rejected.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Mapping

FUNCTIONS: dict[str, Callable[..., float]] = {
    "min": min,
    "max": max,
    "ceil": math.ceil,
    "floor": math.floor,
    "log2": math.log2,
    "sqrt": math.sqrt,
}

_BINOPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}


class ExprError(ValueError):
    pass


def evaluate(expr: float | int | str, env: Mapping[str, float]) -> float:
    """Evaluate a number or expression string with workload parameters `env`."""
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return float(expr)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"bad expression {expr!r}: {exc.msg}") from None
    return float(_eval(tree.body, env, expr))


def _eval(node: ast.AST, env: Mapping[str, float], src: str) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ExprError(f"unknown workload parameter {node.id!r} in {src!r}")
        return env[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left, env, src), _eval(node.right, env, src))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval(node.operand, env, src)
        return v if isinstance(node.op, ast.UAdd) else -v
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in FUNCTIONS
        and not node.keywords
    ):
        return FUNCTIONS[node.func.id](*(_eval(a, env, src) for a in node.args))
    raise ExprError(f"unsupported syntax {type(node).__name__} in {src!r}")
