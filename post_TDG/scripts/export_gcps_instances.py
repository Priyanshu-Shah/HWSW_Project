"""Export GCPS-model benchmark instances (TGFF topologies, Zheng et al. 2021 Table 2 costs) as plain JSON.

So that people without a TGFF build can use them. One file per instance under data/benchmarks/gcps_tgff/:
model, AP, communication class, per-task SW time / HW time / area, edges with comm cost, area budget C, the all-SW
schedule length (AR numerator) and the LSSP schedule length of greedy-by-HGP as a reference. Costs in ns. SYNTHETIC.

Usage: python scripts/export_gcps_instances.py [--sizes 20 50 100 200 500 1000] [--graphs 10] [--ct CT2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from solvers.gcps.baselines import greedy_hgp
from solvers.gcps.instance import CT, BinInstance, fig3_example, generate
from solvers.gcps.lssp import lssp

ROOT = Path(__file__).resolve().parents[1]
MODEL = ("GCPS model (Zheng, Liang & Xiong, DAES 25, 2021): one CPU runs SW tasks sequentially; every HW task has its "
         "own unit; one shared bus carries data between tasks on different sides (same-side edges cost 0); total HW "
         "area <= C. Schedule length by LSSP (paper Alg. 1). AR = all_sw_time / schedule length. Costs in ns.")


def record(inst: BinInstance, ct: str | None, ap: float | None) -> dict:
    y = greedy_hgp(inst)
    return {
        "name": inst.name, "provenance": {"kind": "synthetic", "source": "TGFF 3.5 topology + Zheng 2021 Table 2"},
        "model": MODEL, "comm_class": ct, "comm_range_ns": list(CT[ct]) if ct else None, "ap": ap,
        "n": inst.n, "area_budget_C": inst.C, "all_sw_time": inst.t_max,
        "sw_time": inst.ts.tolist(), "hw_time": inst.th.tolist(), "area": inst.area.tolist(),
        "edges": [[i, j, c] for i, j, c in inst.edges],
        "reference": {"greedy_hgp_hw_tasks": [int(i) for i in y.nonzero()[0]], "greedy_hgp_schedule_length": lssp(inst, y)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[20, 50, 100, 200, 500, 1000])
    ap.add_argument("--graphs", type=int, default=10)
    ap.add_argument("--ct", default="CT2", choices=sorted(CT))
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "benchmarks" / "gcps_tgff")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig3 = record(fig3_example(), None, None)
    fig3["provenance"]["source"] = "Zheng 2021 Fig. 3 / Table 1 worked example (values from the paper)"
    fig3["reference"]["paper_table1_schedule_length"] = 32  # partition of the paper's Table 1
    (args.out / "gcps_fig3.synthetic.json").write_text(json.dumps(fig3, indent=1), encoding="utf-8")
    for n in args.sizes:
        for inst in generate(n, args.ct, 1 / 3, seed=0, graphs=args.graphs):
            path = args.out / f"{inst.name}.synthetic.json"
            path.write_text(json.dumps(record(inst, args.ct, 1 / 3)), encoding="utf-8")
        print(n, "done", flush=True)


if __name__ == "__main__":
    main()
