"""Tier A seed: small random layered DAGs with synthetic costs (for solver cross-checks, not training yet).

Costs are uniform draws with round-ish ranges; they are synthetic and labelled so in the graph id and
provenance. Real-distribution sampling (E3S, HLS-derived costs) comes later.
"""

from __future__ import annotations

import numpy as np

from schemas.ir import TaskGraph


def random_dag(
    n_nodes: int,
    seed: int,
    *,
    platform_id: str = "toy_cpu_fpga",
    platform_path: str | None = None,
    sw_type: str = "cpu.generic",
    hw_type: str = "fpga.generic",
    hw_prob: float = 0.6,
    max_preds: int = 2,
) -> TaskGraph:
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")
    rng = np.random.default_rng(seed)
    nodes = []
    for i in range(n_nodes):
        t_sw = round(float(rng.uniform(1.0, 10.0)), 2)
        impls = [{"id": "sw", "pe_type": sw_type, "cost": {"time_ms": t_sw, "power_mw": 1000}}]
        if rng.random() < hw_prob:
            impls.append({
                "id": "hw",
                "pe_type": hw_type,
                "cost": {
                    "time_ms": round(t_sw * float(rng.uniform(0.1, 0.6)), 2),
                    "power_mw": 300,
                    "resources": {"LUT": int(rng.integers(100, 700))},
                },
            })
        nodes.append({"id": f"t{i}", "implementations": impls})
    edges = []
    for j in range(1, n_nodes):
        k = int(rng.integers(1, min(max_preds, j) + 1))
        for i in sorted(rng.choice(j, size=k, replace=False).tolist()):
            edges.append({"src": f"t{i}", "dst": f"t{j}", "bytes": int(rng.integers(100, 2000))})
    return TaskGraph.model_validate({
        "id": f"synthetic_random_n{n_nodes}_s{seed}",
        "provenance": {"kind": "synthetic", "source": f"datagen.random_dag(n_nodes={n_nodes}, seed={seed})"},
        "platform": {"id": platform_id, "path": platform_path},
        "nodes": nodes,
        "edges": edges,
    })
