from __future__ import annotations

import pytest

from conftest import golden_variant
from estimate import Option
from estimate.expr import ExprError, evaluate


def test_expr_arithmetic_and_functions():
    env = {"w": 1280, "h": 720}
    assert evaluate("w * h", env) == 921600
    assert evaluate("ceil(2 * 6000 / 96)", {}) == 125
    assert evaluate("max(1, min(3, 2)) ** 2 + -1", {}) == 3
    assert evaluate(4, {}) == 4.0


@pytest.mark.parametrize("src", ["__import__('os')", "w.real", "w[0]", "open('x')", "lambda: 1", "w if w else 1"])
def test_expr_rejects_unsafe_syntax(src):
    with pytest.raises(ExprError):
        evaluate(src, {"w": 1})


def test_expr_unknown_parameter():
    with pytest.raises(ExprError, match="unknown workload parameter"):
        evaluate("fps * 2", {})


def test_collision_costs_are_functions_of_workload(collision):
    v = collision.nominal()
    assert v.time[("capture", "sw")] == pytest.approx(0.5 + 1e-6 * 1280 * 720)
    assert v.bytes[("capture", "preprocess")] == 2 * 1280 * 720
    assert collision.period_ms == pytest.approx(1000 / 30)
    # power-based energy: 400 mW * 1.4216 ms = 0.56864 mJ
    assert v.energy[("capture", "sw")] == pytest.approx(0.4 * (0.5 + 0.9216))


def test_workload_override_changes_costs():
    p = golden_variant("collision_detection", lambda d: d["workload"].update(width=640, height=480))
    assert p.nominal().time[("capture", "sw")] == pytest.approx(0.5 + 1e-6 * 640 * 480)


def test_options_respect_pinning_and_types(collision):
    assert collision.options["capture"] == [Option("cpu0", "sw")]
    assert set(collision.options["preprocess"]) == {Option("cpu0", "sw"), Option("cpu1", "sw"), Option("fpga0", "hw")}
    assert {o.pe for o in collision.options["track"]} == {"cpu0", "cpu1"}


def test_allowed_pes_mask_the_platform():
    p = golden_variant("collision_detection", lambda d: d["platform"].update(allowed_pes=["cpu0", "fpga0"]))
    assert p.pes == ["cpu0", "fpga0"]
    assert all(o.pe != "cpu1" for opts in p.options.values() for o in opts)
    assert p.static_power_mw == 150 + 200


def test_structural_problems_are_reported_not_raised():
    p = golden_variant("collision_detection", lambda d: d["platform"].update(allowed_pes=["fpga0"]))
    text = " ".join(p.structural)
    assert "capture: pinned to cpu0" in text
    assert "track: no allowed PE" in text


def test_comm_uses_fastest_interface(collision):
    nb = 1_000_000
    assert collision.comm("cpu0", "cpu1", nb).interface == "l2_shared"
    c = collision.comm("cpu0", "fpga0", nb)
    assert c.interface == "axi_hp_dma"
    assert c.time_ms == pytest.approx(20 / 1000 + nb / 600e3)
    assert collision.comm("cpu0", "cpu0", nb).time_ms == 0


def test_bound_views_bracket_nominal(collision):
    lo, nom, hi = collision.bound_view("lo"), collision.nominal(), collision.bound_view("hi")
    for k in nom.time:
        assert lo.time[k] <= nom.time[k] <= hi.time[k]
