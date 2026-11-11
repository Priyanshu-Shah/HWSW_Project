"""Chain DP oracle vs CP-SAT on random multi-implementation chains, and a long-chain scale check."""

from __future__ import annotations

import time

import numpy as np
import pytest

from conftest import ROOT
from estimate import annotate
from schemas import load_platform
from schemas.ir import TaskGraph
from sim import evaluate
from solvers.cp import solve_cp
from solvers.exact.chain_dp import chain_dp


def chain(n: int, seed: int, cap: int) -> object:
    rng = np.random.default_rng(seed)
    nodes = []
    for i in range(n):
        sw = float(rng.integers(5, 50))
        impls = [{"id": "sw", "pe_type": "cpu.generic", "cost": {"time_ms": sw}}]
        for k in range(int(rng.integers(0, 4))):
            impls.append({"id": f"hw{k}", "pe_type": "fpga.generic",
                          "cost": {"time_ms": float(rng.integers(1, int(sw))),
                                   "resources": {"LUT": int(rng.integers(20, 300))}}})
        nodes.append({"id": f"t{i}", "implementations": impls})
    edges = [{"src": f"t{i}", "dst": f"t{i + 1}", "bytes": int(rng.integers(100, 5000))} for i in range(n - 1)]
    g = TaskGraph.model_validate({"id": f"chain{seed}", "provenance": {"kind": "synthetic", "source": "test"},
                                  "platform": {"id": "toy_cpu_fpga"}, "nodes": nodes, "edges": edges})
    plat = load_platform(ROOT / "platforms" / "toy_cpu_fpga.synthetic.json").model_dump(mode="json")
    plat["pes"][1]["capacity"] = {"LUT": cap}
    from schemas.platform import Platform

    return annotate(g, Platform.model_validate(plat))


@pytest.mark.parametrize("seed", range(5))
def test_chain_dp_matches_cp_sat(seed):
    p = chain(12, seed, cap=600)
    m, val = chain_dp(p)
    assert evaluate(p, m).sim.makespan == pytest.approx(val)  # the DP value is the simulated makespan
    assert evaluate(p, m).feasible
    out = solve_cp(p, contention=True)
    assert out.status == "optimal"
    assert val == pytest.approx(out.makespan, abs=1e-2)


def test_chain_dp_scales_to_long_chains():
    p = chain(1000, 7, cap=20000)
    t0 = time.perf_counter()
    m, val = chain_dp(p)
    assert time.perf_counter() - t0 < 60
    ev = evaluate(p, m)
    assert ev.feasible and ev.sim.makespan == pytest.approx(val)
