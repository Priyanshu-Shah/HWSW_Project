"""Fit Tier A generator distributions from real local data; writes datagen/fits/tier_a_fits.json.

Sources (all read-only; see docs/data_cards/tier_a.md):
- E3S 0.9 (local `e3s/e3s-0.9/*-cords.tgff`): EEMBC-measured task times on 17 CPUs -> software-time model
  log10 t(task, pe) = a_task + b_pe + e (two-way additive model in log space, fitted with missing entries).
- QuickEst (`third_party/quickest/.../data1.csv`): post-implementation LUT/FF/DSP/BRAM marginals, and the
  spread of post-implementation / HLS-estimate ratios -> resource sizes and resource-estimate uncertainty.
- HLSyn v20 (CC BY-SA 4.0; extract `HLSyn_data.tar.gz` from the local Kaggle zip and pass --hlsyn-dir):
  valid pragma design points per kernel -> normalised latency/resource Pareto fronts (implementation options).

Usage: python -I scripts/fit_tier_a.py --hlsyn-dir <dir containing data/designs/v20>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ["LUT", "FF", "DSP", "BRAM"]


def parse_cords(path: Path) -> dict[tuple, dict[int, float]]:
    """{processor signature: {task type: task_time_s}} for valid entries of one E3S cords file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[tuple, dict[int, float]] = {}
    for block in re.findall(r"@PROC\s+\d+\s*\{(.*?)\}", text, flags=re.S):
        rows = [ln.split() for ln in block.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        if not rows:
            continue
        sig = tuple(rows[0])  # price/power header row identifies the processor across files
        times = {}
        for r in rows[1:]:
            if len(r) >= 4 and r[2] == "1":
                times[int(r[0])] = float(r[3])
        out.setdefault(sig, {}).update(times)
    return out


def fit_sw(e3s_dir: Path) -> dict:
    procs: dict[tuple, dict[int, float]] = {}
    files = sorted(e3s_dir.glob("*-cords.tgff"))
    for f in files:
        for sig, times in parse_cords(f).items():
            procs.setdefault(sig, {}).update(times)  # type ids are global across E3S files (verified)
    sigs = sorted(procs)
    tasks = sorted({k for t in procs.values() for k in t})
    y = np.full((len(tasks), len(sigs)), np.nan)
    for j, s in enumerate(sigs):
        for i, k in enumerate(tasks):
            if k in procs[s]:
                y[i, j] = math.log10(procs[s][k] * 1e3)  # ms
    mask = ~np.isnan(y)
    a, b = np.nanmean(y, axis=1), np.zeros(len(sigs))
    for _ in range(200):  # alternating least squares for the additive model
        b = np.array([np.nanmean(y[:, j] - a) for j in range(len(sigs))])
        b -= b.mean()
        a = np.array([np.nanmean(y[i, :] - b) for i in range(len(tasks))])
    resid = (y - a[:, None] - b[None, :])[mask]
    return {
        "source": "E3S 0.9 cords files (EEMBC-measured iterations/s -> task_time), local copy e3s/e3s-0.9",
        "files": [f.name for f in files],
        "n_task_types": len(tasks), "n_processors": len(sigs), "n_valid_entries": int(mask.sum()),
        "fraction_missing": float(1 - mask.mean()),
        "units": "log10(ms)",
        "task_base_mean": float(a.mean()), "task_base_sd": float(a.std(ddof=1)),
        "pe_speed_offsets": [float(x) for x in sorted(b)], "pe_speed_sd": float(b.std(ddof=1)),
        "noise_sd": float(resid.std(ddof=1)),
        "time_ms_min": float(10 ** np.nanmin(y)), "time_ms_max": float(10 ** np.nanmax(y)),
    }


def fit_resources(quickest_csv: Path) -> dict:
    rows = list(csv.DictReader(quickest_csv.open(encoding="utf-8")))
    out: dict = {"source": "QuickEst data1.csv (cornell-zhang/quickest, BSD-3), Vivado 2017.1, 7-series",
                 "n_rows": len(rows), "per_resource": {}}
    for r in RES:
        impl = np.array([float(x[f"{r}_impl"]) for x in rows])
        hls = np.array([float(x[r]) for x in rows])
        pos = impl[impl > 0]
        both = (impl > 0) & (hls > 0)
        ratio = impl[both] / hls[both]
        q05, q50, q95 = np.percentile(ratio, [5, 50, 95])
        out["per_resource"][r] = {
            "fraction_zero": float((impl == 0).mean()),
            "log10_mean_positive": float(np.log10(pos).mean()), "log10_sd_positive": float(np.log10(pos).std(ddof=1)),
            "impl_over_hls_q05": float(q05), "impl_over_hls_q50": float(q50), "impl_over_hls_q95": float(q95),
            # uncertainty factors around a median-calibrated estimate: [q05/q50, q95/q50]
            "unc_factors": [float(min(1.0, q05 / q50)), float(max(1.0, q95 / q50))],
        }
    return out


def pareto(points: list[tuple[float, float]]) -> list[int]:
    idx = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    front, best = [], float("inf")
    for i in idx:
        if points[i][1] < best - 1e-12:
            front.append(i)
            best = points[i][1]
    return front


def fit_fronts(hlsyn_dir: Path, max_points: int = 8) -> dict:
    files = sorted(glob.glob(str(hlsyn_dir / "data" / "designs" / "v20" / "*.json")))
    fronts, speedups, n_valid = {}, [], 0
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        v = [x for x in d.values() if x.get("valid") and x.get("perf", 0) > 0]
        n_valid += len(v)
        if len(v) < 2:
            continue
        perf = [x["perf"] for x in v]
        res = [{r: max(x["res_util"].get(f"total-{r}", 0.0), 0.0) for r in RES} for x in v]
        speedups.append(max(perf) / min(perf))
        size = [res[i]["LUT"] + 50 * res[i]["DSP"] + 300 * res[i]["BRAM"] + 0.5 * res[i]["FF"] for i in range(len(v))]
        fr = pareto([(perf[i], size[i]) for i in range(len(v))])
        fr = sorted(fr, key=lambda i: perf[i])
        if len(fr) > max_points:
            fr = [fr[round(k * (len(fr) - 1) / (max_points - 1))] for k in range(max_points)]
        p0 = fr[0]
        fronts[Path(f).stem] = [
            {"latency_ratio": perf[i] / perf[p0],
             **{f"{r}_ratio": (res[i][r] / res[p0][r]) if res[p0][r] > 0 else 0.0 for r in RES}}
            for i in fr
        ]
    return {
        "source": "HLSyn v20 (UCLA-DM/HLSyn, CC BY-SA 4.0; Vitis 2020.2), valid points only",
        "licence_note": "Derived statistics from a CC BY-SA 4.0 dataset; attribute HLSyn (Bai et al., NeurIPS 2023 D&B).",
        "n_kernels": len(fronts), "n_valid_points": n_valid,
        "note": "Front over (latency, weighted size = LUT + 50 DSP + 300 BRAM + 0.5 FF); ratios relative to the fastest "
                "front point, so latency_ratio >= 1 and resources mostly <= 1 along the front.",
        "pragma_speedup_log10_mean": float(np.log10(speedups).mean()),
        "pragma_speedup_log10_sd": float(np.log10(speedups).std(ddof=1)),
        "fronts": fronts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hlsyn-dir", type=Path, required=True)
    ap.add_argument("--e3s-dir", type=Path, default=ROOT / "e3s" / "e3s-0.9")
    ap.add_argument("--quickest-csv", type=Path,
                    default=ROOT / "third_party/quickest/quickest-master/hls/data/samples/data1.csv")
    args = ap.parse_args()
    fits = {
        "generated_by": "scripts/fit_tier_a.py", "date": date.today().isoformat(),
        "software_time": fit_sw(args.e3s_dir),
        "fpga_resources": fit_resources(args.quickest_csv),
        "implementation_fronts": fit_fronts(args.hlsyn_dir),
        "hw_speedup_prior": {
            "source": "SYNTHETIC ASSUMPTION (no public paired SW/HW per-task measurements; survey §5 gap 1). "
                      "Lognormal on t_SW / t_HW of the fastest HW option, median 5x, 90% within about [0.6x, 40x]; "
                      "replace with DS3 ratios or our own board measurements.",
            "log10_median": math.log10(5.0), "log10_sd": 0.65,
        },
    }
    out = ROOT / "datagen" / "fits" / "tier_a_fits.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fits, indent=1) + "\n", encoding="utf-8")
    sw, rs, fr = fits["software_time"], fits["fpga_resources"]["per_resource"], fits["implementation_fronts"]
    print(f"SW: {sw['n_task_types']} task types x {sw['n_processors']} CPUs, {sw['n_valid_entries']} entries; "
          f"task sd {sw['task_base_sd']:.2f}, pe sd {sw['pe_speed_sd']:.2f}, noise sd {sw['noise_sd']:.2f} (log10 ms)")
    print("Resources:", {r: (round(v['fraction_zero'], 2), round(10 ** v['log10_mean_positive']), v['unc_factors'])
                         for r, v in rs.items()})
    print(f"Fronts: {fr['n_kernels']} kernels from {fr['n_valid_points']} valid points; "
          f"pragma speedup median {10 ** fr['pragma_speedup_log10_mean']:.1f}x")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
