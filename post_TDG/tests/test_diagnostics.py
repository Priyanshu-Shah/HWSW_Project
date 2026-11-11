"""Lower bound and the fb1 / fb2 decision."""

from __future__ import annotations

import pytest

from conftest import golden_variant
from diagnostics import lower_bound
from solvers import ilp


def deadline(name: str, value: float):
    return golden_variant(name, lambda d: d["constraints"].update(deadline_ms=value))


def test_toy_lower_bound(toy):
    lb = lower_bound(toy)
    assert lb.latency_ms == pytest.approx(4.0)  # A 2 + max(B 1, C 1) + D 1, no comm
    assert lb.violations == [] and lb.structural == []


def test_collision_lower_bound(collision):
    lb = lower_bound(collision)
    # capture 1.4216 + preprocess(hw) 2.304 + detect(hw) 16.384 + track 2.8 + ttc 1.0
    assert lb.latency_ms == pytest.approx(1.4216 + 2.304 + 16.384 + 2.8 + 1.0)
    assert lb.latency_ms_optimistic < lb.latency_ms


def test_fb1_when_lower_bound_misses_deadline():
    r = ilp.solve(deadline("toy_forkjoin", 3.0), n_samples=0)
    assert r.diagnostics.loop == "fb1"
    (v,) = r.diagnostics.lower_bound.violations
    assert (v.constraint, v.value, v.limit) == ("deadline_ms", pytest.approx(4.0), 3.0)
    assert "deadline_ms" in r.diagnostics.lower_bound.certain


def test_fb2_with_relaxation_when_mapping_is_the_problem():
    # LB = 4 <= 5, but the best mapping needs 6 ms: the ILP is infeasible and the relaxation says by how much
    r = ilp.solve(deadline("toy_forkjoin", 5.0), n_samples=0)
    assert r.solver.status == "infeasible"
    assert r.diagnostics.loop == "fb2"
    (v,) = r.diagnostics.relaxation
    assert v.constraint == "deadline_ms"
    assert v.excess == pytest.approx(1.0, abs=1e-6)


def test_fb1_nominal_only_vs_certain():
    lb_certain = lower_bound(deadline("collision_detection", 20.0))  # optimistic LB 20.46 > 20
    lb_nominal = lower_bound(deadline("collision_detection", 22.0))  # 20.46 < 22 < 23.91
    assert "deadline_ms" in lb_certain.certain
    assert lb_nominal.violations and "deadline_ms" not in lb_nominal.certain


def test_collision_fb2_between_bound_and_optimum():
    r = ilp.solve(deadline("collision_detection", 25.0), n_samples=0)
    assert r.diagnostics.loop == "fb2"
    (v,) = r.diagnostics.relaxation
    assert v.constraint == "deadline_ms"
    assert 25.0 + v.excess == pytest.approx(27.023, abs=1e-2)


def test_structural_problem_goes_to_fb1():
    p = golden_variant("collision_detection", lambda d: d["platform"].update(allowed_pes=["fpga0"]))
    r = ilp.solve(p, n_samples=0)
    assert r.diagnostics.loop == "fb1"
    assert r.mapping == {}
    assert any("pinned to cpu0" in s for s in r.diagnostics.lower_bound.structural)


def test_task_too_big_for_fpga_is_structural():
    def edit(d):
        b = next(n for n in d["nodes"] if n["id"] == "B")
        b["implementations"] = [i for i in b["implementations"] if i["id"] == "hw"]
        b["implementations"][0]["cost"]["resources"]["LUT"] = 1200  # capacity is 1000

    lb = lower_bound(golden_variant("toy_forkjoin", edit))
    assert any("node B: no legal option fits" in s for s in lb.structural)


def test_streaming_period_violation_in_lower_bound():
    # 1000 fps -> 1 ms period; detect alone needs >= 16 ms on its fastest PE
    p = golden_variant("collision_detection", lambda d: d["workload"].update(fps=1000))
    lb = lower_bound(p)
    assert [v.constraint for v in lb.violations] == ["period_ms"]
