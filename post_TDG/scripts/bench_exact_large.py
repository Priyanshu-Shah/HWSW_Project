"""Exact methods on the 60–100-task set at equal budgets: CP-SAT vs relax-and-schedule vs LBBD.

Instances: data/labels/tier_a_v0_bus_large.synthetic.jsonl (SYNTHETIC, 30 graphs). Every method gets the same
wall-clock budget per instance; every schedule is re-scored by the simulator (exact durations, bus contention).
Raw results are appended per (instance, method) to data/benchmarks/exact_large_runs.synthetic.jsonl, so an
interrupted run resumes. Writes:
  docs/benchmarks/exact_large.md                         per-instance table and summary
  data/benchmarks/large_best_known.synthetic.json        best schedule + strongest bound per instance, all sources

Usage: python scripts/bench_exact_large.py [--budget 20] [--seeds 5000 5003 ...] [--runs F --md F --best F]
"""

from __future__ import annotations

import argparse
import json
import math
import platform as host
import statistics
import time
from datetime import date
from pathlib import Path

from estimate import annotate
from schemas.ir import TaskGraph
from schemas.platform import Platform
from sim import evaluate
from solvers.cp import solve_cp
from solvers.exact.hybrid import relax_and_schedule
from solvers.exact.lbbd import lbbd
from solvers.heuristics.list_sched import heft

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "labels" / "tier_a_v0_bus_large.synthetic.jsonl"
RUNS = ROOT / "data" / "benchmarks" / "exact_large_runs.synthetic.jsonl"
BEST = ROOT / "data" / "benchmarks" / "large_best_known.synthetic.json"
OUT = ROOT / "docs" / "benchmarks" / "exact_large.md"
METHODS = ("cpsat", "hybrid", "lbbd")


def run_cpsat(p, budget: float) -> dict:
    t0 = time.perf_counter()
    c = solve_cp(p, time_limit_s=budget, workers=8, seed=0, hint=heft(p).mapping, contention=True)
    out = {"status": c.status, "bound": c.bound * p.norms["latency"] if c.bound is not None else None,
           "makespan": None, "mapping": None, "priority": None}
    if c.mapping is not None:
        ev = evaluate(p, c.mapping, priority=c.priority(p))
        if ev.feasible:
            out.update(makespan=ev.sim.makespan, mapping=c.mapping, priority=c.priority(p))
    out["runtime_s"] = time.perf_counter() - t0
    return out


def run_decomp(fn, p, budget: float) -> dict:
    o = fn(p, time_limit_s=budget)
    return {"status": o.status, "bound": o.bound, "makespan": o.makespan, "mapping": o.mapping,
            "priority": o.priority, "runtime_s": o.runtime_s, "iterations": o.iterations, "notes": o.notes}


def gap(ub: float | None, lb: float | None) -> float | None:
    return None if ub is None or lb is None or ub <= 0 else max(0.0, (ub - lb) / ub)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--md", type=Path, default=OUT)
    ap.add_argument("--best", type=Path, default=BEST)
    args = ap.parse_args()
    runs_path = args.runs
    records = [json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines()]
    if args.seeds:
        records = [r for r in records if r["seed"] in set(args.seeds)]
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if runs_path.exists():
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if d["budget_s"] == args.budget:
                done.add((d["seed"], d["method"]))
    for r in records:
        p = annotate(TaskGraph.model_validate(r["graph"]), Platform.model_validate(r["platform"]))
        for m in METHODS:
            if (r["seed"], m) in done:
                continue
            res = run_cpsat(p, args.budget) if m == "cpsat" else \
                run_decomp(relax_and_schedule if m == "hybrid" else lbbd, p, args.budget)
            rec = {"seed": r["seed"], "method": m, "budget_s": args.budget, **res,
                   "mapping": {t: [o.pe, o.impl] for t, o in res["mapping"].items()} if res["mapping"] else None}
            with runs_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(r["seed"], m, rec["status"], rec["makespan"] and round(rec["makespan"], 2),
                  rec["bound"] and round(rec["bound"], 2), f"{rec['runtime_s']:.1f}s", flush=True)
    report(args.budget, runs_path, args.md, args.best)


def report(budget: float, runs_path: Path, out_md: Path, best_path: Path) -> None:
    records = {r["seed"]: r for r in (json.loads(x) for x in LABELS.read_text(encoding="utf-8").splitlines())}
    runs: dict[int, dict[str, dict]] = {}
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d["budget_s"] == budget:
            runs.setdefault(d["seed"], {})[d["method"]] = d
    best_known, rows = {}, []
    for seed in sorted(runs):
        r, res = records[seed], runs[seed]
        p = annotate(TaskGraph.model_validate(r["graph"]), Platform.model_validate(r["platform"]))
        lab = r["label"]
        prior_ub = r["replay"]["makespan_with_contention_ms"] if r.get("replay") and lab.get("mapping") else None
        prior_lb = lab["bound"] * p.norms["latency"] if lab.get("bound") is not None else None
        if lab["status"] == "optimal":
            prior_lb = prior_ub
        infeasible = lab["status"] == "infeasible" or any(x["status"] == "infeasible" for x in res.values())
        # schedules from every record (all are simulator-checked); bounds only from trusted methods: superseded
        # records (e.g. hybrid_v1, start-time hints) may carry a false optimality claim (DECISIONS 2026-11-06)
        ubs = [(x["makespan"], m) for m, x in res.items() if x["makespan"] is not None]
        if prior_ub is not None:
            ubs.append((prior_ub, f"label (CP-SAT {lab['runtime_s']:.0f} s)"))
        lbs = [x["bound"] for m, x in res.items() if m in METHODS and x["bound"] is not None and x["bound"] > -1e300]
        lbs += [prior_lb] if prior_lb else []
        ub, src = min(ubs) if ubs else (None, None)
        lb = max(lbs) if lbs else None
        if lb is not None and ub is not None and lb > ub * (1 + 1e-3):
            raise RuntimeError(f"seed {seed}: bound {lb:.4f} exceeds a known schedule {ub:.4f} ({src})")
        if lb is not None and ub is not None:
            lb = min(lb, ub)  # equal within the 0.1% tolerance
        entry = {"n_tasks": len(p.order), "infeasible": infeasible, "best_makespan_ms": ub, "best_source": src,
                 "bound_ms": lb, "gap": gap(ub, lb), "optimal": ub is not None and gap(ub, lb) is not None
                 and gap(ub, lb) <= 1e-3}
        if ub is not None and src in res:
            entry["mapping"], entry["priority"] = res[src]["mapping"], res[src]["priority"]
        elif ub is not None:
            entry["mapping"], entry["priority"] = lab["mapping"], lab["priority"]
        best_known[str(seed)] = entry
        rows.append((seed, len(p.order), lab["status"], prior_ub, gap(prior_ub, prior_lb), res, ub, lb, infeasible))

    best_path.write_text(json.dumps({"provenance": {"kind": "synthetic", "source": "Tier A v0 large set; "
                                               f"bench_exact_large.py, budget {budget:g} s per method"},
                                "instances": best_known}, indent=1), encoding="utf-8")

    def cell(x: dict | None, lb: float | None) -> str:
        if x is None:
            return "—"
        if x["status"] == "infeasible":
            return "infeasible"
        if x["makespan"] is None:
            return "no schedule"
        g = gap(x["makespan"], lb)
        return f"{x['makespan']:.1f} ({g:.1%})" if g is not None else f"{x['makespan']:.1f}"

    lines = [
        "# Exact methods on 60–100 tasks: CP-SAT vs relax-and-schedule vs LBBD",
        "",
        f"Generated by `scripts/bench_exact_large.py` on {date.today().isoformat()} ({host.processor() or host.machine()}). "
        f"Budget **{budget:g} s per method per instance**, CP-SAT 8 workers. **All 30 instances are synthetic** "
        "(Tier A v0 large set, 60–100 tasks, bus contention). Every schedule is re-scored by the simulator. "
        "Cells: makespan in ms, and in brackets its gap to the **best known lower bound** for that instance (the "
        "maximum over all methods' bounds and the original label's bound; relax-and-schedule's bound includes the "
        "mapping relaxation). "
        "`label` is the original 20 s CP-SAT run that produced the data set. Superseded runs (`hybrid_v1`, which "
        "hinted CP-SAT with start times and once received a false optimum from CP-SAT 9.15) contribute schedules "
        "(all simulator-checked) but no bounds. Wall-clock budgets on a shared machine: expect a few % run-to-run "
        "variation on unproven instances.",
        "",
        "| seed | tasks | label | CP-SAT | relax-and-schedule | LBBD | best known | bound | gap |",
        "|---:|---:|---|---|---|---|---:|---:|---:|",
    ]
    for seed, n, lst, pub, pg, res, ub, lb, inf in rows:
        lab = "infeasible" if lst == "infeasible" else (f"{pub:.1f} ({gap(pub, lb):.1%})" if pub else "no schedule")
        lines.append(f"| {seed} | {n} | {lab} | {cell(res.get('cpsat'), lb)} | {cell(res.get('hybrid'), lb)} | "
                     f"{cell(res.get('lbbd'), lb)} | {'infeasible' if inf else (f'{ub:.1f}' if ub else '—')} | "
                     f"{lb:.1f} | {'—' if gap(ub, lb) is None else f'{gap(ub, lb):.1%}'} |"
                     if lb is not None else f"| {seed} | {n} | {lab} | — | — | — | — | — | — |")
    lines += ["", "## Summary (instances not proven infeasible)", "",
              "| method | schedule found | best of the three (ties count) | gap to best bound: median | p90 | max |",
              "|---|---:|---:|---:|---:|---:|"]
    feas_rows = [x for x in rows if not x[8] and x[7] is not None]
    for m in METHODS:
        gs, found, best = [], 0, 0
        for seed, *_, res, ub, lb, inf in feas_rows:
            x = res.get(m)
            if x and x["makespan"] is not None:
                found += 1
                gs.append(gap(x["makespan"], lb))
                if x["makespan"] <= min(y["makespan"] for y in res.values() if y["makespan"] is not None) * (1 + 1e-6):
                    best += 1
        gs.sort()
        stats = (f"{statistics.median(gs):.1%} | {gs[max(0, math.ceil(0.9 * len(gs)) - 1)]:.1%} | {gs[-1]:.1%}" if gs
                 else "— | — | —")
        lines.append(f"| {m} | {found}/{len(feas_rows)} | {best}/{len(feas_rows)} | {stats} |")
    n_opt = sum(1 for e in best_known.values() if e["optimal"])
    lines += ["", f"Best known after this run: **{n_opt}** of {len(best_known)} instances optimal within 0.1%; "
              f"references in `{best_path.name}`."]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-12:]))


if __name__ == "__main__":
    main()
