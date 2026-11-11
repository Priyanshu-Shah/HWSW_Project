"""ILP correctness: hand-checked optimum, schedule validity, and cross-checks against exhaustive search."""

from __future__ import annotations

import pytest

from conftest import GOLDEN_IDS, ROOT, TOY_B_HW, golden_problem
from datagen import random_dag
from estimate import annotate
from schemas import load_platform
from sim import evaluate
from sim.validate import validate_starts
from solvers.heuristics import exhaustive
from solvers.ilp import solve_ilp


def ilp_priority(problem, out):
    return sorted(out.starts, key=lambda n: (out.starts[n], problem.order.index(n)))


def test_toy_optimum(toy):
    out = solve_ilp(toy)
    assert out.status == "optimal"
    assert out.makespan == pytest.approx(6.0, abs=1e-6)
    assert out.mapping == TOY_B_HW


@pytest.mark.parametrize("name", GOLDEN_IDS)
def test_ilp_schedule_valid_and_reproduced_by_simulator(name):
    p = golden_problem(name)
    out = solve_ilp(p)
    assert out.status == "optimal"
    assert validate_starts(p, out.mapping, out.starts) == []
    sim = evaluate(p, out.mapping, priority=ilp_priority(p, out), contention=False).sim
    assert sim.makespan == pytest.approx(out.makespan, abs=1e-6)


@pytest.mark.parametrize("name", GOLDEN_IDS)
def test_ilp_never_worse_than_exhaustive_list_scheduling(name):
    p = golden_problem(name)
    out = solve_ilp(p)
    ilp_ev = evaluate(p, out.mapping, priority=ilp_priority(p, out), contention=False)
    _, best = exhaustive(p, contention=False)
    assert ilp_ev.feasible
    assert ilp_ev.objective <= best.objective + 1e-6


def test_chain_ilp_equals_exhaustive(wifi):
    # On a chain list scheduling is optimal for any mapping, so the two must agree exactly.
    out = solve_ilp(wifi)
    ilp_ev = evaluate(wifi, out.mapping, contention=False)
    _, best = exhaustive(wifi, contention=False)
    assert ilp_ev.objective == pytest.approx(best.objective, abs=1e-6)


@pytest.mark.parametrize("name", ["toy_forkjoin", "collision_detection"])
def test_cbc_and_highs_agree(name):
    p = golden_problem(name)
    a, b = solve_ilp(p, solver="highs"), solve_ilp(p, solver="cbc")
    assert a.status == b.status == "optimal"
    assert a.makespan == pytest.approx(b.makespan, abs=1e-5)


@pytest.mark.parametrize("seed", range(6))
def test_random_dags_against_exhaustive(seed):
    platform = load_platform(ROOT / "platforms" / "toy_cpu_fpga.synthetic.json")
    p = annotate(random_dag(7, seed), platform)
    out = solve_ilp(p)
    assert out.status == "optimal"
    assert validate_starts(p, out.mapping, out.starts) == []
    ilp_ev = evaluate(p, out.mapping, priority=ilp_priority(p, out), contention=False)
    _, best = exhaustive(p, contention=False)
    if best.feasible:
        assert ilp_ev.objective <= best.objective + 1e-6
