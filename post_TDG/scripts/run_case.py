"""Run the solvers on one case study and print a summary; optionally write Result JSON files.

Usage:
    python scripts/run_case.py examples/golden/collision_detection.synthetic.json
    python scripts/run_case.py <graph.json> --solvers ilp greedy_eft sa --samples 1000 --out out/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from diagnostics import lower_bound
from estimate import Option, annotate
from metrics import rank
from schemas import load_case
from schemas.result import Result
from solvers import cp, heuristics, ilp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("graph", type=Path)
    ap.add_argument("--solvers", nargs="+", default=["cp", "ilp", *heuristics.HEURISTICS])
    ap.add_argument("--time-limit", type=float, default=60.0, help="Per exact solve (CP-SAT, ILP)")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ilp-solver", default="highs", choices=["highs", "cbc"])
    ap.add_argument("--out", type=Path, default=None, help="Directory for <graph>.<solver>.result.json files")
    args = ap.parse_args()

    graph, platform = load_case(args.graph)
    problem = annotate(graph, platform)
    lb = lower_bound(problem)
    print(f"{graph.id} on {platform.id}: lower bound {lb.latency_ms:.3f} ms "
          f"(optimistic {lb.latency_ms_optimistic:.3f}), II >= {lb.ii_ms:.3f} ms")

    results: dict[str, Result] = {}
    for name in args.solvers:
        if name == "ilp":
            r = ilp.solve(problem, solver=args.ilp_solver, seed=args.seed, n_samples=args.samples,
                          time_limit_s=args.time_limit)
        elif name == "cp":
            r = cp.solve(problem, seed=args.seed, n_samples=args.samples, time_limit_s=args.time_limit)
        else:
            r = heuristics.run(problem, name, seed=args.seed, n_samples=args.samples)
        results[name] = r
        if r.metrics is None:
            print(f"  {name:11s} {r.solver.status:10s} no mapping; loop={r.diagnostics.loop}")
        else:
            lat = r.metrics.latency_ms
            print(f"  {name:11s} {r.solver.status:10s} latency {lat.nominal:8.3f} ms "
                  f"[p05 {lat.p05:.3f}, p95 {lat.p95:.3f}]  energy {r.metrics.energy_mj.nominal:8.3f} mJ  "
                  f"feasible={r.feasible} ({100 * r.metrics.feasible_fraction:.0f}% of samples)  "
                  f"loop={r.diagnostics.loop}")
        for reason in r.diagnostics.reasons:
            print(f"      - {reason}")
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{graph.id}.{name}.result.json").write_text(r.model_dump_json(indent=2), encoding="utf-8")

    mapped = {k: {n: Option(a.pe, a.impl) for n, a in r.mapping.items()} for k, r in results.items() if r.mapping}
    if len(mapped) > 1:
        rk = rank(problem, mapped, n_samples=args.samples, seed=args.seed)
        print(f"ranking (best first): {rk.candidates}; order preserved in {100 * rk.order_preserved_fraction:.1f}% "
              f"of samples; mean Kendall tau {rk.mean_kendall_tau:.3f}")


if __name__ == "__main__":
    main()
