"""Build the `{value, lo, hi, source}` estimate objects of the cost map."""


def make_estimate(value: float, uncertainty: float, source: str) -> dict:
    """`lo`/`hi` are `value` -/+ `uncertainty` (a fraction, e.g. 0.3 for 30%)."""
    return {
        "value": round_significant(value),
        "lo": round_significant(value * (1 - uncertainty)),
        "hi": round_significant(value * (1 + uncertainty)),
        "source": source,
    }


def round_significant(number: float) -> float:
    """6 significant digits: keeps the JSON readable and free of float noise."""
    return float(f"{number:.6g}")
