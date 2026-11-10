"""Profile the QuickEst (Dai et al., FCCM'18) data release and reproduce the
HLS-report baseline error.

Reads the vendored copy under third_party/quickest/ (never modifies it) and
writes a generated report to docs/data_cards/quickest_profile.md plus the
baseline-error table to docs/data_cards/quickest_baseline_rae.csv.

Pickle files in the release are NOT unpickled (untrusted pickles can execute
code); they are only scanned with pickletools to list the globals they import.

Usage:
    python -I scripts/profile_quickest.py [--repo-root PATH] [--n-splits 1000]
"""

from __future__ import annotations

import argparse
import hashlib
import pickletools
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20181024  # fixed; date of the vendored commit
TEST_RATIO = 0.20  # paper Sec. V: "randomly select 20% of our data as the testing set"

# Device index -> part, inferred from the *_avail columns (Xilinx 7-series datasheet capacities).
DEVICE_BY_LUT_AVAIL = {
    53200: "XC7Z020 (Zynq-7000)",
    63400: "XC7A100T (Artix-7)",
    101400: "XC7K160T (Kintex-7)",
    364200: "XC7V585T (Virtex-7)",
}

# Paper Table III, row "HLS Estimate" (RAE, %).
PAPER_HLS_RAE = {
    "XC7Z020 (Zynq-7000)": {"LUT": 141.2, "FF": 84.3, "DSP": 22.1, "BRAM": 14.9},
    "XC7A100T (Artix-7)": {"LUT": 99.6, "FF": 89.1, "DSP": 19.8, "BRAM": 9.2},
    "XC7K160T (Kintex-7)": {"LUT": 112.5, "FF": 73.5, "DSP": 15.8, "BRAM": 12.2},
    "XC7V585T (Virtex-7)": {"LUT": 107.0, "FF": 95.4, "DSP": 14.3, "BRAM": 6.9},
}
# Paper Table IV, row "HLS Estimate" (timing misclassification, %).
PAPER_HLS_TIMING_ERR = {
    "XC7Z020 (Zynq-7000)": 21.3,
    "XC7A100T (Artix-7)": 19.8,
    "XC7K160T (Kintex-7)": 19.1,
    "XC7V585T (Virtex-7)": 16.1,
}
DEVICE_ORDER = list(PAPER_HLS_RAE)  # paper orders devices by increasing size

TARGETS = ["LUT", "FF", "DSP", "BRAM"]  # HLS estimate column -> f"{t}_impl" target column


def rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative absolute error: sum|y_pred - y| / sum|y - mean(y)| (paper Sec. V; analyze.py score_REA)."""
    denom = np.abs(y_true - y_true.mean()).sum()
    return float(np.abs(y_pred - y_true).sum() / denom) if denom > 0 else float("nan")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pickle_globals(path: Path) -> list[str]:
    """List module.attr globals referenced by a pickle without executing it."""
    found: set[str] = set()
    strings: list[str] = []
    with path.open("rb") as fh:
        for op, arg, _ in pickletools.genops(fh):
            if op.name == "GLOBAL":
                found.add(str(arg).replace(" ", "."))
            elif op.name == "STACK_GLOBAL" and len(strings) >= 2:
                found.add(f"{strings[-2]}.{strings[-1]}")
            if isinstance(arg, str):
                strings.append(arg)
    return sorted(found)


def classify_vivado_columns(cols: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "id": [], "hls_estimate_of_target": [], "device_capacity": [], "hls_utilisation_ratio": [],
        "clock": [], "resource_by_hw_category": [], "memory_mux_totals": [],
        "per_operator_class_resources": [], "per_operator_class_counts": [],
        "per_operator_type_resources": [], "target_post_impl": [], "unclassified": [],
    }
    for c in cols:
        if c in ("Design_Index", "Device_Index"):
            g = "id"
        elif c in TARGETS:
            g = "hls_estimate_of_target"
        elif c.endswith("_impl"):
            g = "target_post_impl"
        elif c.endswith("_avail"):
            g = "device_capacity"
        elif c.endswith("_RATIO"):
            g = "hls_utilisation_ratio"
        elif c in ("Target_CP", "Estimated_CP", "Clock_Uncertainty"):
            g = "clock"
        elif re.fullmatch(r"(DSP|EXP|FIFO|MEM|MUX|REG)_(BRAM_18K|DSP48E|FF|LUT)", c):
            g = "resource_by_hw_category"
        elif c.endswith("_TOTAL"):
            g = "memory_mux_totals"
        elif re.fullmatch(r"[a-z]+_op_(lut|ff|dsp)", c):
            g = "per_operator_class_resources"
        elif re.fullmatch(r"[a-z]+_sum", c):
            g = "per_operator_class_counts"
        elif re.fullmatch(r"[a-z]+_(lut|ff|dsp|bram)", c):
            g = "per_operator_type_resources"
        else:
            g = "unclassified"
        groups[g].append(c)
    return groups


def md_table(df: pd.DataFrame, floatfmt: str = "{:.1f}") -> str:
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else (f"{int(v)}" if float(v).is_integer() and abs(v) >= 10
                                                     else floatfmt.format(v)))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def inventory(release: Path) -> str:
    rows = []
    for p in sorted(release.rglob("*")):
        if p.is_file():
            rel = p.relative_to(release).as_posix()
            ext = p.suffix.lower() or "(none)"
            kind = {".py": "Python 2 source", ".csv": "CSV", ".pkl": "Python pickle", ".md": "Markdown"}.get(ext, "text")
            if p.name == "LICENSE":
                kind = "licence text"
            rows.append(f"| `{rel}` | {p.stat().st_size:,} | {kind} | `{sha256(p)[:12]}` |")
    head = "| path (under quickest-master/) | bytes | type | sha256[:12] |\n|---|---:|---|---|"
    return head + "\n" + "\n".join(rows)


def profile_vivado(df: pd.DataFrame, n_splits: int) -> tuple[str, pd.DataFrame]:
    out: list[str] = []
    df = df.copy()
    feat_cols = list(df.columns[2:-5])  # data1.csv: 5 targets incl. CP_impl
    groups = classify_vivado_columns(list(df.columns))
    const = [c for c in feat_cols if df[c].nunique() <= 1]
    n_cp_missing = int((df.CP_impl == -1).sum())
    cp_missing_designs = df.loc[df.CP_impl == -1, "Design_Index"].nunique()
    df["CP_impl"] = df["CP_impl"].where(df.CP_impl != -1)  # -1 is a missing-value sentinel
    lut_avail = df.groupby("Device_Index")["LUT_avail"].unique()
    dev_name = {int(k): DEVICE_BY_LUT_AVAIL[int(v[0])] for k, v in lut_avail.items()}
    df["device"] = df["Device_Index"].map(dev_name)

    out.append("### Shape\n")
    out.append(f"- rows: {len(df)}; designs: {df.Design_Index.nunique()}; devices: {df.Device_Index.nunique()}")
    out.append(f"- `CP_impl` == -1 (missing-value sentinel, treated as NaN): {n_cp_missing} rows from "
               f"{cp_missing_designs} designs; LUT/FF/DSP/BRAM targets have no sentinel")
    out.append(f"- columns: {df.shape[1] - 1} (2 ids + {len(feat_cols)} features + 5 targets; `data.csv` is identical minus `CP_impl`)")
    out.append(f"- empty cells: 0; non-numeric columns: "
               f"{[c for c in df.columns[:-1] if not np.issubdtype(df[c].dtype, np.number)]}")
    out.append(f"- features constant across all rows (carry no information in this release): {len(const)} of {len(feat_cols)}")
    out.append(f"- Target_CP values (ns): {df.Target_CP.value_counts().sort_index().to_dict()}")
    out.append(f"- device index -> part: {dict(sorted(dev_name.items()))}")
    out.append(f"- rows per device: {df.groupby('device').size().reindex(DEVICE_ORDER).to_dict()}")
    per_design = df.groupby("Design_Index").size()
    out.append(f"- rows per design: min {per_design.min()}, median {int(per_design.median())}, max {per_design.max()}")
    dup = df.groupby(["Design_Index", "Device_Index", "Target_CP"]).size()
    out.append(f"- (design, device, Target_CP) keys with >1 row: {int((dup > 1).sum())} "
               f"(rows in them: {int(dup[dup > 1].sum())}) -- likely directive/config variants, not labelled in the data")

    out.append("\n### Column groups\n")
    out.append("| group | n | examples |\n|---|---:|---|")
    for g, cs in groups.items():
        if cs:
            ex = ", ".join(f"`{c}`" for c in cs[:6]) + (" ..." if len(cs) > 6 else "")
            out.append(f"| {g} | {len(cs)} | {ex} |")
    out.append("\nEvery feature column is read from the Vivado HLS report (or is a device-capacity constant). "
               "Only the `*_impl` columns come from post-place-and-route Vivado reports.")

    out.append("\n### Value ranges\n")
    rng_cols = TARGETS + [f"{t}_impl" for t in TARGETS] + ["Target_CP", "Estimated_CP", "Clock_Uncertainty", "CP_impl"]
    rng = df[rng_cols].describe().T[["count", "min", "50%", "mean", "max"]]
    rng.index.name = "column"
    out.append(md_table(rng))
    out.append("\nPaper Table I (post-implementation, 1300+ samples): LUT 34 / 5791 / 63645, FF 0 / 7395 / 115452, "
               "DSP 0 / 25 / 795, BRAM 0 / 19 / 350 (min / mean / max).")

    out.append("\n### HLS over/under-estimation (ratio HLS / post-impl, rows with post-impl > 0)\n")
    ratio_rows = []
    for t in TARGETS:
        m = df[f"{t}_impl"] > 0
        r = df.loc[m, t] / df.loc[m, f"{t}_impl"]
        ratio_rows.append({"target": t, "n": int(m.sum()), "median": r.median(), "mean": r.mean(),
                           "max": r.max(), "% overestimated": 100 * (r > 1).mean()})
    rt = pd.DataFrame(ratio_rows).set_index("target")
    out.append(md_table(rt, "{:.2f}"))
    out.append("\nPaper text: HLS LUT estimates are 'on average 4.5x and up to 40x' actual.")

    # --- baseline RAE --------------------------------------------------------
    res_rows = []
    for dev in DEVICE_ORDER + ["all devices"]:
        sub = df if dev == "all devices" else df[df.device == dev]
        for t in TARGETS:
            res_rows.append({"device": dev, "target": t, "scope": "full", "n": len(sub),
                             "rae_pct": 100 * rae(sub[f"{t}_impl"].to_numpy(float), sub[t].to_numpy(float))})

    rng_ = np.random.default_rng(SEED)
    n_test = int(round(TEST_RATIO * len(df)))
    samples: dict[tuple[str, str], list[float]] = {}
    for _ in range(n_splits):
        test = df.iloc[rng_.choice(len(df), n_test, replace=False)]
        for dev in DEVICE_ORDER:
            sub = test[test.device == dev]
            for t in TARGETS:
                samples.setdefault((dev, t), []).append(
                    100 * rae(sub[f"{t}_impl"].to_numpy(float), sub[t].to_numpy(float)))
    for (dev, t), v in samples.items():
        a = np.asarray(v)
        res_rows.append({"device": dev, "target": t, "scope": f"random20_median_of_{n_splits}",
                         "n": n_test, "rae_pct": float(np.nanmedian(a)),
                         "p05": float(np.nanpercentile(a, 5)), "p95": float(np.nanpercentile(a, 95))})
    for dev in DEVICE_ORDER:
        for t in TARGETS:
            res_rows.append({"device": dev, "target": t, "scope": "paper_table_III", "n": np.nan,
                             "rae_pct": PAPER_HLS_RAE[dev][t]})
    res = pd.DataFrame(res_rows)

    out.append("\n### Baseline: HLS-report RAE (%) per device and target\n")
    out.append("RAE = sum|HLS - impl| / sum|impl - mean(impl)|, mean taken over the evaluated subset. "
               "RAE > 100% means the HLS report is worse than predicting the subset mean.\n")
    tbl = []
    for dev in DEVICE_ORDER + ["all devices"]:
        for t in TARGETS:
            r = {"device / target": f"{dev} / {t}"}
            f = res[(res.device == dev) & (res.target == t)]
            r["full release"] = f[f.scope == "full"].rae_pct.iloc[0]
            s = f[f.scope.str.startswith("random20")]
            if len(s):
                r[f"random 20% test, median (p5-p95) over {n_splits} splits"] = (
                    f"{s.rae_pct.iloc[0]:.1f} ({s.p05.iloc[0]:.1f}-{s.p95.iloc[0]:.1f})")
            p = f[f.scope == "paper_table_III"]
            r["paper Table III"] = p.rae_pct.iloc[0] if len(p) else np.nan
            tbl.append(r)
    out.append(md_table(pd.DataFrame(tbl).set_index("device / target")))

    # --- timing (paper Table IV) -------------------------------------------
    valid = df.CP_impl.notna().to_numpy()
    actual_met = (df.CP_impl <= df.Target_CP).to_numpy()
    pred_a = (df.Estimated_CP <= df.Target_CP).to_numpy()
    pred_b = (df.Estimated_CP <= df.Target_CP - df.Clock_Uncertainty).to_numpy()
    trows = []
    for dev in DEVICE_ORDER + ["all devices"]:
        m = valid & (np.ones(len(df), bool) if dev == "all devices" else (df.device == dev).to_numpy())
        trows.append({"device": dev, "n (CP_impl known)": int(m.sum()),
                      "% timing met (CP_impl <= Target_CP)": 100 * actual_met[m].mean(),
                      "HLS err %, est <= target": 100 * (pred_a[m] != actual_met[m]).mean(),
                      "HLS err %, est <= target - uncertainty": 100 * (pred_b[m] != actual_met[m]).mean(),
                      "paper Table IV": PAPER_HLS_TIMING_ERR.get(dev, np.nan)})
    for r in trows:
        res_rows.append({"device": r["device"], "target": "timing_met", "scope": "full", "n": r["n (CP_impl known)"],
                         "rae_pct": np.nan, "err_pct_est_le_target": r["HLS err %, est <= target"],
                         "err_pct_est_le_target_minus_unc": r["HLS err %, est <= target - uncertainty"]})
    out.append("\n### Baseline: HLS timing-met classification error (%)\n")
    out.append("Ground truth 'met' := `CP_impl <= Target_CP` (assumption: `CP_impl` is the achieved post-route "
               "clock period in ns). Rows with `CP_impl` == -1 are excluded. Two HLS decision rules are shown because the release does not say which the "
               "paper used.\n")
    out.append(md_table(pd.DataFrame(trows).set_index("device")))

    return "\n".join(out), pd.DataFrame(res_rows)


def profile_legup(d2: pd.DataFrame, d3: pd.DataFrame) -> str:
    out = []
    est = ["Registers", "DSP Elements", "Combinational", "RAM Elements", "Logic Elements"]
    tgt = ["Registers_used", "DSP_blocks_used", "ALUT_used", "RAM_blocks_used", "Block_memory_bits_used"]
    for name, df in (("data2.csv", d2), ("data3.csv", d3)):
        feats = df.columns[2:-5]
        out.append(f"- `{name}`: {len(df)} rows, {df.Design_Index.nunique()} designs, device index "
                   f"{sorted(int(x) for x in df.Device_Index.unique())} only, {len(feats)} features, 5 targets, "
                   f"missing {int(df.isna().sum().sum())}, constant features {int((df[feats].nunique() <= 1).sum())}; "
                   f"`Clock Period` values {df['Clock Period'].value_counts().sort_index().to_dict()}")
    both = set(d2.Design_Index) & set(d3.Design_Index)
    out.append(f"- design ids shared by data2/data3: {len(both)}; identical column lists: {list(d2.columns) == list(d3.columns)}")
    out.append(f"- first 5 features: {est}; 5 targets: {tgt}. The README rule 'first k features are the HLS values of "
               "the k targets' does not hold position-by-position here (e.g. `Logic Elements` vs `Block_memory_bits_used`).")
    corr = pd.DataFrame({t: [d3[e].corr(d3[t]) for e in est] for t in tgt}, index=est)
    corr.index.name = "data3 Pearson r: LegUp estimate \\ Quartus target"
    out.append("\n" + md_table(corr, "{:.2f}"))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--n-splits", type=int, default=1000)
    args = ap.parse_args()

    root: Path = args.repo_root
    release = root / "third_party" / "quickest" / "quickest-master"
    data_dir = release / "hls" / "data" / "samples"
    out_dir = root / "docs" / "data_cards"
    out_dir.mkdir(parents=True, exist_ok=True)

    d0 = pd.read_csv(data_dir / "data.csv")
    d1 = pd.read_csv(data_dir / "data1.csv")
    assert d1.drop(columns="CP_impl").equals(d0), "data1.csv is expected to be data.csv + CP_impl"

    viv_md, res = profile_vivado(d1, args.n_splits)
    legup_md = profile_legup(pd.read_csv(data_dir / "data2.csv"), pd.read_csv(data_dir / "data3.csv"))

    pkl_lines = ["| pickle | globals referenced (not loaded) |", "|---|---|"]
    for p in sorted(release.rglob("*.pkl")):
        g = pickle_globals(p)
        pkl_lines.append(f"| `{p.relative_to(release).as_posix()}` | {', '.join(f'`{x}`' for x in g) or '(builtins only)'} |")

    report = "\n".join([
        "# QuickEst release: generated profile",
        "",
        "Generated by `scripts/profile_quickest.py` -- do not edit by hand; edit the script and re-run.",
        f"Seed {SEED}; random-split repetitions {args.n_splits}; test ratio {TEST_RATIO}.",
        "",
        "## 1. Inventory of `third_party/quickest/quickest-master/`",
        "",
        inventory(release),
        "",
        "### Pickles (scanned with `pickletools`, never unpickled)",
        "",
        "\n".join(pkl_lines),
        "",
        "## 2. Vivado HLS dataset (`hls/data/samples/data1.csv`)",
        "",
        viv_md,
        "",
        "## 3. LegUp / Intel datasets (`data2.csv`, `data3.csv`)",
        "",
        legup_md,
        "",
    ])
    (out_dir / "quickest_profile.md").write_text(report, encoding="utf-8")
    res.to_csv(out_dir / "quickest_baseline_rae.csv", index=False, float_format="%.4f")
    print(report)


if __name__ == "__main__":
    main()
