"""Tier A generator: determinism, provenance, fitted sources, and solvability; canonical CP labels."""

from __future__ import annotations

import json

import pytest

from datagen.tier_a import FPGAS, generate, load_fits
from estimate import annotate
from sim import evaluate
from solvers.cp import solve_cp


def test_fits_come_from_the_documented_sources():
    f = load_fits()
    assert f["software_time"]["n_task_types"] == 46 and f["software_time"]["n_processors"] == 17
    assert f["fpga_resources"]["n_rows"] == 1046
    assert f["implementation_fronts"]["n_kernels"] >= 20
    assert "SYNTHETIC" in f["hw_speedup_prior"]["source"]


def test_deterministic_and_labelled_synthetic():
    g1, p1 = generate(3)
    g2, p2 = generate(3)
    assert g1 == g2 and p1 == p2
    assert g1.provenance.kind == "synthetic" and "tierA" in g1.id
    params = json.loads(g1.provenance.notes)
    assert {"topology", "ccr", "fpga_pressure", "deadline_lambda", "lower_bound_ms"} <= set(params)
    assert generate(4)[0] != g1


@pytest.mark.parametrize("seed", range(6))
def test_instances_are_well_formed(seed):
    g, p = generate(seed)
    prob = annotate(g, p)
    assert not prob.structural
    assert p.pe("fpga0").capacity in list(FPGAS.values())
    params = json.loads(g.provenance.notes)
    assert params["lower_bound_ms"] <= prob.deadline_ms + 1e-6
    # every task has a software option on every CPU
    cpus = [pe.id for pe in p.pes if pe.kind == "cpu"]
    for n in prob.order:
        assert {o.pe for o in prob.options[n]} >= set(cpus)


def test_canonical_labels_are_optimal_and_reproducible():
    g, p = generate(7)
    prob = annotate(g, p)
    plain = solve_cp(prob, time_limit_s=10)
    a = solve_cp(prob, time_limit_s=10, canonical=True)
    b = solve_cp(prob, time_limit_s=10, canonical=True, seed=5)
    assert a.status == "optimal" and "canonicalised" in a.notes
    assert a.makespan == pytest.approx(plain.makespan, abs=1e-9)
    assert a.mapping == b.mapping and a.starts == b.starts  # same canonical label regardless of search seed
    prio = sorted(a.starts, key=lambda n: (a.starts[n], prob.order.index(n)))
    replay = evaluate(prob, a.mapping, priority=prio, contention=False).sim.makespan
    assert replay == pytest.approx(a.makespan, abs=1e-3 * len(prob.order))
