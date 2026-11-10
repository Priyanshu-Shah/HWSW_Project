from __future__ import annotations

import pytest

from conftest import GOLDEN_IDS, TOY_B_HW, golden_problem
from sim import evaluate
from solvers.heuristics import HEURISTICS, all_sw, greedy_eft, plan, simulated_annealing
from solvers.ilp import solve_ilp


@pytest.mark.parametrize("name", GOLDEN_IDS)
@pytest.mark.parametrize("heur", list(HEURISTICS))
def test_heuristics_return_legal_mappings(name, heur):
    p = golden_problem(name)
    mp = plan(p, heur).mapping
    assert set(mp) == set(p.order)
    assert all(mp[n] in p.options[n] for n in p.order)


@pytest.mark.parametrize("name", GOLDEN_IDS)
def test_feasible_heuristics_are_not_better_than_ilp(name):
    p = golden_problem(name)
    out = solve_ilp(p)
    prio = sorted(out.starts, key=lambda n: (out.starts[n], p.order.index(n)))
    best = evaluate(p, out.mapping, priority=prio, contention=False).objective
    for heur in HEURISTICS:
        pl = plan(p, heur)
        ev = evaluate(p, pl.mapping, priority=pl.priority, contention=False, insertion=pl.insertion)
        if ev.feasible:
            assert best <= ev.objective + 1e-6, heur


def test_greedy_respects_capacity_on_toy(toy):
    assert greedy_eft(toy) == TOY_B_HW


def test_all_sw_uses_only_cpus(collision):
    assert all(collision.pe_kind(o.pe) == "cpu" for o in all_sw(collision).values())


def test_sa_is_deterministic_per_seed(collision):
    assert simulated_annealing(collision, seed=3, n_iter=300) == simulated_annealing(collision, seed=3, n_iter=300)
