"""Mapping relaxation is a valid lower bound; LBBD reaches the CP-SAT optimum on small instances."""

from __future__ import annotations

import json
from itertools import islice

import pytest

from conftest import ROOT
from estimate import annotate
from schemas.ir import TaskGraph
from schemas.platform import Platform
from solvers.cp import solve_cp
from solvers.exact.hybrid import relax_and_schedule
from solvers.exact.lbbd import lbbd
from solvers.exact.relaxation import relaxation_bound

LABELS = ROOT / "data" / "labels" / "tier_a_v0_bus.synthetic.jsonl"


def _optimal(k: int):
    out = []
    with LABELS.open(encoding="utf-8") as fh:
        lines = list(islice(fh, 4 * k))
    for line in lines:
        r = json.loads(line)
        if r["label"]["status"] == "optimal" and len(r["graph"]["nodes"]) <= 25:
            p = annotate(TaskGraph.model_validate(r["graph"]), Platform.model_validate(r["platform"]))
            out.append((p, r["replay"]["makespan_with_contention_ms"], r["lower_bound_ms"]))
        if len(out) == k:
            break
    return out


@pytest.mark.skipif(not LABELS.exists(), reason="labels not generated")
def test_relaxation_bounds_are_valid_and_ordered():
    for p, opt, crit in _optimal(6):
        lp = relaxation_bound(p, integer=False)
        ip = relaxation_bound(p, integer=True, time_limit_s=10)
        assert ip.status == "optimal"
        assert crit - 1e-6 <= lp.bound_ms + 1e-6  # LP includes the critical path
        assert lp.bound_ms <= ip.bound_ms + 1e-6
        assert ip.bound_ms <= opt * (1 + 1e-6)  # never above the exact optimum


@pytest.mark.skipif(not LABELS.exists(), reason="labels not generated")
def test_lbbd_matches_cp_sat_optimum():
    for p, opt, _ in _optimal(3):
        out = lbbd(p, time_limit_s=60, sub_time_s=10, workers=4)
        assert out.mapping is not None and out.makespan is not None
        assert out.bound <= opt * (1 + 1e-6)
        assert out.makespan >= opt * (1 - 2e-3)  # CP-SAT µs rounding
        cp = solve_cp(p, time_limit_s=30, workers=4, contention=True)
        assert cp.status == "optimal"
        assert out.makespan <= cp.makespan * 1.05  # LBBD may stop within its gap, never far off


@pytest.mark.skipif(not LABELS.exists(), reason="labels not generated")
def test_relax_and_schedule_is_consistent():
    for p, opt, _ in _optimal(3):
        out = relax_and_schedule(p, time_limit_s=20, workers=4)
        assert out.makespan is not None and out.bound is not None
        assert out.bound <= opt * (1 + 2e-3) and out.makespan >= opt * (1 - 2e-3)
        assert out.bound <= out.makespan
