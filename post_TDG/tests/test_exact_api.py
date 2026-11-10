"""The exact component's entry point: method selection, certificates, and the two OR-Tools pitfalls handled."""

from __future__ import annotations

import json
import sys
from itertools import islice

import pytest

from conftest import ROOT
from datagen.tier_a import TierAConfig, generate
from estimate import annotate
from schemas.ir import TaskGraph
from schemas.platform import Platform
from sim import evaluate
from solvers.exact import certify, run, solve_exact
from solvers.heuristics import plan

LABELS = ROOT / "data" / "labels" / "tier_a_v0_bus.synthetic.jsonl"
sys.path.insert(0, str(ROOT / "tests"))
from test_chain_dp import chain  # noqa: E402


def _records(status: str, k: int):
    out = []
    with LABELS.open(encoding="utf-8") as fh:
        lines = list(islice(fh, 200))
    for line in lines:
        r = json.loads(line)
        if r["label"]["status"] == status and len(r["graph"]["nodes"]) <= 25:
            out.append((annotate(TaskGraph.model_validate(r["graph"]), Platform.model_validate(r["platform"])), r))
        if len(out) == k:
            break
    return out


@pytest.mark.skipif(not LABELS.exists(), reason="labels not generated")
def test_small_instances_reach_the_label_optimum_and_replay():
    for p, r in _records("optimal", 3):
        res = solve_exact(p, time_limit_s=30, workers=4)
        opt = r["replay"]["makespan_with_contention_ms"]
        assert res.status == "optimal"
        assert res.makespan_ms == pytest.approx(opt, rel=2e-3)
        assert res.bound_ms <= res.makespan_ms
        ev = evaluate(p, res.plan.mapping, priority=res.plan.priority, insertion=res.plan.insertion)
        assert ev.feasible and ev.sim.makespan == pytest.approx(res.makespan_ms)


@pytest.mark.skipif(not LABELS.exists(), reason="labels not generated")
def test_infeasible_instance_is_proven_infeasible():
    p, _ = _records("infeasible", 1)[0]
    res = solve_exact(p, time_limit_s=20, workers=4)
    assert res.status == "infeasible" and res.plan is None and res.certificate


def test_chain_goes_to_the_dp():
    p = chain(30, seed=3, cap=1500)
    res = solve_exact(p, time_limit_s=20, workers=4)
    assert res.method == "chain DP" and res.status == "optimal"
    assert certify(p, res.plan).status == "optimal"  # the relaxation is exact on chains


def test_microsecond_rounding_does_not_fake_infeasibility():
    # deadline = all-SW makespan to the ns, which is also the optimum: CP-SAT alone reports infeasible
    g, plat = generate(8100, TierAConfig(n_tasks=(10, 10), fpga_pressure=(1.5, 2.0), deadline_tightness=(1.0, 1.0)))
    p = annotate(g, plat)
    res = solve_exact(p, time_limit_s=20, workers=4)
    assert res.status in ("optimal", "feasible") and res.evaluation.feasible
    assert res.makespan_ms <= p.deadline_ms + 1e-9


@pytest.mark.skipif(not LABELS.exists(), reason="labels not generated")
def test_warm_start_and_result_contract():
    p, _ = _records("optimal", 1)[0]
    res = run(p, time_limit_s=20, warm_starts=[plan(p, "peft")], workers=4, n_samples=50)
    assert res.feasible and res.solver.name == "exact"
    lb = res.diagnostics.lower_bound
    assert lb.latency_ms_relaxation is not None and lb.latency_ms_relaxation >= lb.latency_ms - 1e-9
