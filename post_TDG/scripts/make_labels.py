"""Generate Tier A instances and exact CP-SAT labels; writes data/labels/tier_a_<version>.synthetic.jsonl and a
summary in docs/data_cards/tier_a_labels_summary.md.

`--semantics bus` (default) labels are optimal WITH shared-bus contention (CP-SAT transfer intervals; replayed
with insertion). `--semantics nobus` reproduces the earlier no-contention labels. Each record also stores:
- the replay of the label's start order through the simulator without contention (decoder gap, survey §7.4),
- the same replay with bus contention (what the label is worth on the full model),
- HEFT on the full model, the lower bound, deadline and all-SW makespan for context.

Usage: python scripts/make_labels.py --n 200 [--start 0] [--time-limit 10] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import date
from pathlib import Path

from datagen.tier_a import VERSION, TierAConfig, generate
from diagnostics import lower_bound
from estimate import annotate
from sim import evaluate
from solvers.cp import solve_cp
from solvers.heuristics import heft

ROOT = Path(__file__).resolve().parents[1]


def label_one(seed: int, time_limit: float, workers: int, bus: bool = True, cfg: TierAConfig | None = None) -> dict:
    g, p = generate(seed, cfg)
    prob = annotate(g, p)
    params = json.loads(g.provenance.notes)
    h = heft(prob)
    h_full = evaluate(prob, h.mapping, priority=h.priority, insertion=True)
    h_nc = evaluate(prob, h.mapping, priority=h.priority, insertion=True, contention=False)
    t0 = time.perf_counter()
    c = solve_cp(prob, time_limit_s=time_limit, workers=workers, seed=0, hint=h.mapping, canonical=True,
                 contention=bus)
    rec = {
        "id": g.id, "seed": seed, "version": VERSION, "params": params,
        "graph": g.model_dump(mode="json"), "platform": p.model_dump(mode="json"),
        "lower_bound_ms": lower_bound(prob).latency_ms, "deadline_ms": prob.deadline_ms,
        "heft": {"makespan_ms": h_full.sim.makespan, "feasible": h_full.feasible,
                 "makespan_no_contention_ms": h_nc.sim.makespan},
        "label": {"solver": "cp-sat", "status": c.status, "objective": c.objective, "bound": c.bound, "gap": c.gap,
                  "runtime_s": time.perf_counter() - t0, "notes": c.notes,
                  "semantics": "bus-contention" if bus else "no-contention",
                  "makespan_ms": c.makespan},
    }
    if c.mapping is not None:
        prio = c.priority(prob)  # tasks + transfers (unified decoder) for bus labels
        nc = evaluate(prob, c.mapping, priority=[x for x in prio if "->" not in x], contention=False)
        full = evaluate(prob, c.mapping, priority=prio)
        own = full if bus else nc  # replay under the label's own semantics
        rec["label"].update({
            "mapping": {n: [o.pe, o.impl] for n, o in c.mapping.items()},
            "starts_ms": c.starts, "priority": prio,
            "feasible_no_contention": nc.feasible, "transfer_starts_ms": c.transfer_starts,
        })
        rec["replay"] = {
            "makespan_no_contention_ms": nc.sim.makespan,
            "decoder_gap": own.sim.makespan / c.makespan - 1.0 if c.makespan else 0.0,
            "makespan_with_contention_ms": full.sim.makespan, "feasible_with_contention": full.feasible,
        }
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--semantics", choices=["bus", "nobus"], default="bus")
    ap.add_argument("--tasks", type=int, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="override the Tier A task-count range (default 10 40)")
    ap.add_argument("--tag", default="", help="extra file-name tag, e.g. _train or _large")
    args = ap.parse_args()
    bus = args.semantics == "bus"
    cfg = TierAConfig(n_tasks=tuple(args.tasks)) if args.tasks else None
    tag = ("_bus" if bus else "") + args.tag
    out = ROOT / "data" / "labels" / f"tier_a_{VERSION}{tag}.synthetic.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    t_all = time.perf_counter()
    with out.open("w", encoding="utf-8") as fh:
        for seed in range(args.start, args.start + args.n):
            r = label_one(seed, args.time_limit, args.workers, bus, cfg)
            fh.write(json.dumps(r) + "\n")
            recs.append(r)
            if (seed - args.start + 1) % 20 == 0:
                print(f"{seed - args.start + 1}/{args.n} done, {time.perf_counter() - t_all:.0f}s", flush=True)
    total = time.perf_counter() - t_all

    status = {s: sum(r["label"]["status"] == s for r in recs) for s in ("optimal", "feasible", "infeasible", "time_limit")}
    lab = [r for r in recs if "replay" in r]
    gaps = [r["replay"]["decoder_gap"] for r in lab]
    heft_key = "makespan_ms" if bus else "makespan_no_contention_ms"  # HEFT under the label's own semantics
    heft_ratio = [r["heft"][heft_key] / r["label"]["makespan_ms"] for r in lab if r["label"]["makespan_ms"]]
    cont_ratio = [r["replay"]["makespan_with_contention_ms"] / r["label"]["makespan_ms"] for r in lab
                  if r["label"]["makespan_ms"]]
    sizes = [r["params"]["n_tasks"] for r in recs]
    rt = [r["label"]["runtime_s"] for r in recs]

    def q(xs: list[float], p: float) -> float:
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] if xs else float("nan")

    lines = [
        f"# Tier A {VERSION} label run ({'WITH shared-bus contention' if bus else 'no contention'}): summary",
        "",
        f"Generated by `scripts/make_labels.py` on {date.today().isoformat()}: {args.n} instances (seeds "
        f"{args.start}–{args.start + args.n - 1}), CP-SAT {args.time_limit:g} s limit, {args.workers} workers, "
        f"canonical tie-breaking. Total {total:.0f} s. **All instances are synthetic.**",
        f"Data: `data/labels/{out.name}` (one JSON record per instance: graph, platform, label, replays, HEFT).",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| tasks per graph | {min(sizes)}–{max(sizes)} (median {statistics.median(sizes):g}) |",
        f"| label status | " + ", ".join(f"{k} {v}" for k, v in status.items() if v) + " |",
        f"| CP-SAT time per instance | median {statistics.median(rt):.2f} s, p95 {q(rt, 0.95):.2f} s, max {max(rt):.2f} s |",
        f"| decoder gap: replay of the label order under its own semantics vs label | longer by at most "
        f"{100 * max(gaps, default=0):.3f}% ({sum(x > 1e-3 for x in gaps)} of {len(gaps)} above 0.1%); shorter by up "
        f"to {-100 * min(gaps, default=0):.3f}% (CP-SAT rounds positive durations up to whole µs) |",
        f"| HEFT / optimum (both {'with' if bus else 'without'} contention) | median {statistics.median(heft_ratio):.3f}, "
        f"p95 {q(heft_ratio, 0.95):.3f}, max {max(heft_ratio):.3f} |",
        f"| label replayed WITH contention / label | median {statistics.median(cont_ratio):.3f}, "
        f"p95 {q(cont_ratio, 0.95):.3f}, max {max(cont_ratio):.3f} |",
        f"| labels feasible (deadline etc.) without / with contention | "
        f"{sum(r['label']['feasible_no_contention'] for r in lab)} / {sum(r['replay']['feasible_with_contention'] for r in lab)} of {len(lab)} |",
        "",
        "Reading: the decoder gap must be ~0 for imitation through our list-scheduling decoder to be lossless. For "
        "no-contention labels the contention ratio measures how much the optimum degrades on the full simulator "
        "model; for bus-contention labels it equals 1 + decoder gap, i.e. label and evaluator agree.",
    ]
    summary = ROOT / "docs" / "data_cards" / f"tier_a{tag}_labels_summary.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[6:]))


if __name__ == "__main__":
    main()
