"""Integration contract with task generation: golden inputs -> Result JSON matching stored snapshots.

Snapshots in examples/golden/expected/ are regenerated with `python scripts/make_golden_expected.py`.
They pin our own outputs on synthetic inputs (regression), not real-hardware truth.
"""

from __future__ import annotations

import json

import pytest

from conftest import GOLDEN, GOLDEN_IDS, golden_problem
from schemas.result import Result
from solvers import ilp


@pytest.mark.parametrize("name", GOLDEN_IDS)
def test_result_round_trips_and_matches_snapshot(name):
    r = ilp.solve(golden_problem(name), n_samples=200, seed=0)
    assert Result.model_validate_json(r.model_dump_json()) == r

    exp = json.loads((GOLDEN / "expected" / f"{name}.expected.json").read_text(encoding="utf-8"))
    assert r.diagnostics.loop == exp["loop"]
    assert r.feasible == exp["feasible"]
    assert {n: f"{a.pe}/{a.impl}" for n, a in r.mapping.items()} == exp["mapping"]
    assert r.diagnostics.lower_bound.latency_ms == pytest.approx(exp["lower_bound_latency_ms"], rel=1e-9)
    assert r.metrics.latency_ms.nominal == pytest.approx(exp["latency_ms_nominal"], rel=1e-9)
    assert r.metrics.energy_mj.nominal == pytest.approx(exp["energy_mj_nominal"], rel=1e-9)
    assert r.metrics.latency_ms.p95 == pytest.approx(exp["latency_ms_p95"], rel=1e-9)
