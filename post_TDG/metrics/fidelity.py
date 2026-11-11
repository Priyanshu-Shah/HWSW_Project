"""Metric-quality measures: MAPE for absolute estimates, Kendall's tau-b for rankings ("fidelity")."""

from __future__ import annotations

import math
from collections.abc import Sequence


def mape(predicted: Sequence[float], measured: Sequence[float]) -> float:
    """Mean absolute percentage error, in percent. Measured values must be non-zero."""
    if len(predicted) != len(measured) or not measured:
        raise ValueError("need two equal-length, non-empty sequences")
    if any(m == 0 for m in measured):
        raise ValueError("MAPE is undefined when a measured value is 0")
    return 100.0 * sum(abs(p - m) / abs(m) for p, m in zip(predicted, measured, strict=True)) / len(measured)


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    """Kendall's tau-b between two score vectors (handles ties). Returns nan if either is constant."""
    if len(a) != len(b):
        raise ValueError("sequences must have equal length")
    concordant = discordant = ties_a = ties_b = 0
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
    return (concordant - discordant) / denom if denom else float("nan")
