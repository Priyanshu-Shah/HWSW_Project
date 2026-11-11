from __future__ import annotations

import numpy as np
import pytest

from conftest import TOY_ALL_SW, TOY_B_HW, TOY_C_HW, golden_variant
from metrics import kendall_tau, mape, metrics_for, rank


def test_kendall_tau_known_values():
    assert kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert kendall_tau([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(4 / 6)
    assert np.isnan(kendall_tau([1, 1, 1], [1, 2, 3]))


def test_kendall_tau_matches_scipy_with_ties():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.integers(0, 4, 12).tolist(), rng.integers(0, 4, 12).tolist()
        expected = stats.kendalltau(a, b).statistic
        if not np.isnan(expected):
            assert kendall_tau(a, b) == pytest.approx(expected)


def test_mape():
    assert mape([110, 90], [100, 100]) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        mape([1], [0])


def test_absolute_interval_for_deterministic_and_uncertain_mappings(toy):
    det = metrics_for(toy, TOY_B_HW, n_samples=200, seed=0)
    assert det.latency_ms.lo == det.latency_ms.hi == det.latency_ms.nominal == pytest.approx(6.0)
    unc = metrics_for(toy, TOY_ALL_SW, n_samples=2000, seed=0)
    assert unc.latency_ms.nominal == pytest.approx(10.0)
    assert 9.0 <= unc.latency_ms.p05 < unc.latency_ms.p50 < unc.latency_ms.p95 <= 11.0
    assert unc.feasible_fraction == 0.0  # deadline 8 < 9


def test_ranking_clear_order(toy):
    r = rank(toy, {"all_sw": TOY_ALL_SW, "b_hw": TOY_B_HW, "c_hw": TOY_C_HW}, n_samples=500, seed=0)
    assert r.candidates == ["b_hw", "c_hw", "all_sw"]
    assert r.order_preserved_fraction == 1.0
    assert r.mean_kendall_tau == pytest.approx(1.0)
    assert r.pairwise_confidence[0][2] == 1.0


def test_ranking_confidence_under_overlap():
    # C on FPGA now takes 5 ms -> that mapping is exactly 10 ms; all-SW is 6 + B, B ~ U[3, 5] -> 50/50 order.
    def edit(d):
        c = next(n for n in d["nodes"] if n["id"] == "C")
        c["implementations"][1]["cost"]["time_ms"] = 5
        d["constraints"] = {}

    p = golden_variant("toy_forkjoin", edit)
    r = rank(p, {"all_sw": TOY_ALL_SW, "c_hw": TOY_C_HW}, n_samples=4000, seed=0)
    i, j = r.candidates.index("all_sw"), r.candidates.index("c_hw")
    assert r.pairwise_confidence[i][j] == pytest.approx(0.5, abs=0.03)
    assert r.pairwise_confidence[i][j] + r.pairwise_confidence[j][i] == pytest.approx(1.0, abs=1e-9)


def test_identical_mappings_are_merged(toy):
    r = rank(toy, {"x": TOY_B_HW, "y": TOY_B_HW, "z": TOY_ALL_SW}, n_samples=50, seed=0)
    assert r.candidates == ["x=y", "z"]
