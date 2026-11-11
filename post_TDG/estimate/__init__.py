"""Bind step (cost estimation / performance annotation).

Phase 0: evaluates hand-written parameterised costs. The calibrated FPGA-resource estimator trained on
QuickEst (docs/data_cards/quickest.md) will plug in here later.
"""

from estimate.annotate import Comm, CostView, Option, Problem, annotate, critical_path

__all__ = ["Comm", "CostView", "Option", "Problem", "annotate", "critical_path"]
