"""Simulator against hand-computed schedules on the toy fork-join (1000 bytes = 1 ms on bus0)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import TOY_ALL_SW, TOY_B_HW, TOY_BOTH_HW, TOY_C_HW, m
from estimate import Option
from sim import InvalidMapping, evaluate, simulate
from sim import montecarlo
from sim.validate import validate_sim


@pytest.mark.parametrize(
    "mapping, makespan",
    [
        (TOY_ALL_SW, 10.0),  # A 0-2, B,C serial on cpu0 2-9, D 9-10
        (TOY_B_HW, 6.0),  # A 0-2; A->B 2-3; B 3-4; C cpu 2-5; B->D 4-5; D 5-6
        (TOY_C_HW, 7.0),  # A 0-2; A->C 2-3; C 3-4; B cpu 2-6; C->D 4-5; D 6-7
    ],
)
def test_toy_makespans(toy, mapping, makespan):
    sim = simulate(toy, mapping)
    assert sim.makespan == pytest.approx(makespan)
    assert validate_sim(toy, sim) == []


def test_toy_energy_from_power(toy):
    # cpu tasks at 1000 mW: A 2 + C 3 + D 1 = 6 mJ; B on fpga 1 ms at 200 mW = 0.2 mJ; no static/comm energy
    assert simulate(toy, TOY_B_HW).energy_mj == pytest.approx(6.2)


def test_bus_contention_serialises_transfers(toy):
    # both on FPGA: A->B 2-3, A->C 3-4 (bus busy), B 3-4, C 4-5, B->D 4-5, C->D 5-6, D 6-7
    assert simulate(toy, TOY_BOTH_HW, contention=True).makespan == pytest.approx(7.0)
    assert simulate(toy, TOY_BOTH_HW, contention=False).makespan == pytest.approx(6.0)


def test_capacity_violation_reported(toy):
    ev = evaluate(toy, TOY_BOTH_HW)
    assert [v.constraint for v in ev.violations] == ["capacity[fpga0].LUT"]
    assert ev.violations[0].excess == pytest.approx(200)


def test_deadline_violation_reported(toy):
    ev = evaluate(toy, TOY_ALL_SW)  # 10 ms vs deadline 8
    assert [(v.constraint, v.excess) for v in ev.violations] == [("deadline_ms", pytest.approx(2.0))]


def test_illegal_mappings_raise(toy):
    with pytest.raises(InvalidMapping):
        simulate(toy, {**TOY_ALL_SW, "A": Option("fpga0", "sw")})
    with pytest.raises(InvalidMapping, match="unmapped"):
        simulate(toy, m(A="cpu0/sw"))


def test_priority_override_changes_order(toy):
    # Run C before B on cpu0: identical makespan for all-SW, but C finishes first
    sim = simulate(toy, TOY_ALL_SW, priority=["A", "C", "B", "D"])
    assert sim.tasks["C"].finish < sim.tasks["B"].finish


def test_monte_carlo_envelope_matches_interval(toy):
    # all-SW makespan = 6 + B, with B in [3, 5]
    mc = montecarlo.run(toy, TOY_ALL_SW, n=2000, seed=1)
    assert 9.0 - 1e-9 <= mc.latency_ms.min() < 9.05
    assert 10.95 < mc.latency_ms.max() <= 11.0 + 1e-9
    # deterministic mapping: B on FPGA has no uncertainty
    assert np.ptp(montecarlo.run(toy, TOY_B_HW, n=200, seed=1).latency_ms) == 0


def test_monte_carlo_is_reproducible(collision):
    from solvers.heuristics import greedy_eft

    mp = greedy_eft(collision)
    a = montecarlo.run(collision, mp, n=50, seed=7).latency_ms
    b = montecarlo.run(collision, mp, n=50, seed=7).latency_ms
    assert np.array_equal(a, b)


@pytest.mark.parametrize("fixture", ["collision", "wifi"])
def test_golden_schedules_are_valid(fixture, request):
    from solvers.heuristics import greedy_eft

    p = request.getfixturevalue(fixture)
    sim = simulate(p, greedy_eft(p))
    assert validate_sim(p, sim) == []
