"""KL, tabu, GA and NSGA-II: optimality on hand-checked cases, determinism, constraint handling."""

from __future__ import annotations

import pytest

from conftest import GOLDEN_IDS, ROOT, TOY_ALL_SW, TOY_B_HW, golden_problem
from datagen import random_dag
from estimate import annotate
from schemas import load_platform
from sim import evaluate
from solvers.heuristics import genetic_algorithm, kernighan_lin, nsga2, tabu_search
from solvers.heuristics.local_search import kl_cost
from solvers.ilp import solve_ilp


def test_kl_from_all_sw_reaches_toy_optimum(toy):
    assert kernighan_lin(toy, start=TOY_ALL_SW) == TOY_B_HW


def test_kl_permissive_cost_forbids_large_violations(toy):
    ev = evaluate(toy, TOY_ALL_SW)  # 10 ms vs deadline 8: ratio 1.25 < q = 2 -> finite penalty
    assert kl_cost(ev) < float("inf")
    assert kl_cost(ev, q=1.2) == float("inf")


def test_tabu_from_all_sw_reaches_toy_optimum(toy):
    assert tabu_search(toy, start=TOY_ALL_SW, n_iter=30) == TOY_B_HW


@pytest.mark.parametrize("name", GOLDEN_IDS)
def test_search_heuristics_match_ilp_on_goldens(name):
    p = golden_problem(name)
    out = solve_ilp(p)
    prio = sorted(out.starts, key=lambda n: (out.starts[n], p.order.index(n)))
    opt = evaluate(p, out.mapping, priority=prio).objective
    for m in (kernighan_lin(p), tabu_search(p, n_iter=50)):
        ev = evaluate(p, m)
        assert ev.feasible
        assert ev.objective <= opt * 1.01


def test_ga_toy_and_determinism(toy):
    m1, pr1 = genetic_algorithm(toy, pop_size=16, generations=15, seed=3)
    m2, pr2 = genetic_algorithm(toy, pop_size=16, generations=15, seed=3)
    assert (m1, pr1) == (m2, pr2)
    assert m1 == TOY_B_HW


def test_ga_repair_keeps_capacity_on_random_dag():
    platform = load_platform(ROOT / "platforms" / "toy_cpu_fpga.synthetic.json")
    p = annotate(random_dag(12, seed=5), platform)
    mapping, prio = genetic_algorithm(p, pop_size=20, generations=10, seed=0)
    ev = evaluate(p, mapping, priority=prio)
    assert not [v for v in ev.violations if v.constraint.startswith("capacity")]


def test_nsga2_toy_front(toy):
    # B on HW (6 ms, 6.2 mJ) dominates C on HW (7 ms, 7.2 mJ) and all-SW (10 ms, 10 mJ); both-HW is infeasible.
    front = nsga2(toy, pop_size=16, generations=15, seed=0)
    feasible = [f for f in front if f.feasible]
    assert [f.mapping for f in feasible] == [TOY_B_HW]
    assert feasible[0].objectives == pytest.approx((6.0, 6.2))


def test_nsga2_front_is_mutually_non_dominated():
    p = golden_problem("wifi_tx")
    front = [f for f in nsga2(p, pop_size=24, generations=20, seed=1) if f.feasible]
    assert front
    for a in front:
        for b in front:
            if a is not b:
                assert not (all(x <= y for x, y in zip(a.objectives, b.objectives, strict=True))
                            and any(x < y for x, y in zip(a.objectives, b.objectives, strict=True)))


def test_magellan_reaches_toy_optimum_and_never_worsens_its_start(toy, collision):
    from solvers.heuristics import heft, magellan

    assert magellan(toy).mapping == TOY_B_HW
    for p in (toy, collision):
        start = heft(p)
        e0 = evaluate(p, start.mapping, priority=start.priority, insertion=True)
        pl = magellan(p)
        e1 = evaluate(p, pl.mapping, priority=pl.priority, insertion=pl.insertion)
        assert (not e1.feasible, e1.objective) <= (not e0.feasible, e0.objective)


def test_tabu_takes_oldest_move_when_everything_is_tabu(toy):
    # tenure longer than the run: after the first moves every node is tabu, yet the search keeps moving
    assert tabu_search(toy, start=TOY_ALL_SW, n_iter=10, tenure=100) == TOY_B_HW
