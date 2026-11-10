"""Regenerate examples/golden/expected/*.expected.json (regression snapshots of ILP results).

Usage: python scripts/make_golden_expected.py
Review the diff before committing: a change means solver/simulator behaviour changed.
"""

from __future__ import annotations

import json
from pathlib import Path

from estimate import annotate
from schemas import load_case
from solvers import ilp

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "examples" / "golden"


def main() -> None:
    out_dir = GOLDEN / "expected"
    out_dir.mkdir(exist_ok=True)
    for path in sorted(GOLDEN.glob("*.synthetic.json")):
        graph, platform = load_case(path)
        r = ilp.solve(annotate(graph, platform), n_samples=200, seed=0)
        snap = {
            "_note": "Regression snapshot of our own solver on SYNTHETIC inputs. Not hardware measurements.",
            "graph": path.name,
            "solver": r.solver.name,
            "loop": r.diagnostics.loop,
            "feasible": r.feasible,
            "mapping": {n: f"{a.pe}/{a.impl}" for n, a in r.mapping.items()},
            "lower_bound_latency_ms": r.diagnostics.lower_bound.latency_ms,
            "latency_ms_nominal": r.metrics.latency_ms.nominal,
            "latency_ms_p95": r.metrics.latency_ms.p95,
            "energy_mj_nominal": r.metrics.energy_mj.nominal,
            "feasible_fraction": r.metrics.feasible_fraction,
            "n_samples": 200,
            "seed": 0,
        }
        target = out_dir / f"{graph.id}.expected.json"
        target.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
