"""Evaluate the size formulas the LLM writes, e.g. "1280*720*3" (D4: the TDG holds literals).

Models are bad at multiplying large numbers, so they write the formula and Python does the
arithmetic. Only numbers and + - * / // % ** with parentheses are accepted; nothing is executed.
"""

import ast
import operator

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_MAX_EXPONENT = 64


def evaluate_formula(formula: str | int | float) -> float:
    if isinstance(formula, bool):
        raise ValueError(f"expected a number or a formula, got {formula}")
    if isinstance(formula, int | float):
        return formula
    try:
        tree = ast.parse(formula.replace("^", "**").replace("×", "*"), mode="eval")
        return _evaluate(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError) as failure:
        raise ValueError(
            f"'{formula}' is not a plain arithmetic formula of numbers ({failure})"
        ) from failure


def evaluate_count(formula: str | int | float) -> int:
    """A formula whose value must be a whole, non-negative number (bytes, op counts)."""
    value = evaluate_formula(formula)
    if value < 0 or abs(value - round(value)) > 1e-6 * max(1.0, abs(value)):
        raise ValueError(f"'{formula}' = {value:g}, but it must be a whole number >= 0")
    return round(value)


def _evaluate(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        value = _evaluate(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("exponent too large")
        return _BINARY[type(node.op)](left, right)
    raise ValueError(f"'{ast.unparse(node)}' is not allowed")
