"""Tier A generator v0: synthetic task graphs + platforms whose cost distributions are fitted to real data.

Every instance is SYNTHETIC. Its provenance lists, per input, where the distribution came from
(datagen/fits/tier_a_fits.json, produced by scripts/fit_tier_a.py; data card docs/data_cards/tier_a.md):

| input | source |
|---|---|
| topology | Kasahara STG methods (sameprob, samepred, layered) + series-parallel composition |
| SW time per CPU | E3S 0.9 (EEMBC-measured): log10 t = a_task + b_pe + e, fitted |
| HW speedup (SW/HW, fastest option) | SYNTHETIC prior (no public paired data; survey §5) |
| HW implementation options | HLSyn v20 normalised Pareto fronts (latency vs resources), resampled |
| FPGA resources of the fastest option | QuickEst post-implementation marginals (zero-inflated DSP/BRAM) |
| resource uncertainty | QuickEst impl/HLS ratio spread around the median |
| time uncertainty | SYNTHETIC ±intervals |
| edge bytes | log-uniform, rescaled to a sampled CCR (Kasahara-style 0.1-10) |
| FPGA capacities | XC7Z020 (DS190, = QuickEst *_avail); XCK26 (local Hoopa platform file) |
| interfaces | SYNTHETIC placeholders (as platforms/zynq7020_class) |
| deadline | lower bound + λ·(all-SW makespan - lower bound), λ sampled |
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from schemas.ir import TaskGraph
from schemas.platform import Platform

FITS_PATH = Path(__file__).resolve().parent / "fits" / "tier_a_fits.json"
VERSION = "v0"

FPGAS = {
    # capacity source: Xilinx DS190 for XC7Z020 (identical to QuickEst LUT_avail etc. for Device_Index 1)
    "xc7z020": {"LUT": 53200, "FF": 106400, "DSP": 220, "BRAM18K": 280},
    # capacity source: KV260 platform file in the local hoopa-master.zip (Vitis 2024.2)
    "xck26": {"LUT": 117120, "FF": 234240, "DSP": 1248, "BRAM18K": 288},
}


@dataclass
class TierAConfig:
    n_tasks: tuple[int, int] = (10, 40)
    topologies: tuple[str, ...] = ("layered", "sameprob", "samepred", "series_parallel")
    n_cpus: tuple[int, ...] = (1, 2)
    fpgas: tuple[str, ...] = ("xc7z020", "xck26")
    hw_task_fraction: float = 0.7
    impl_options: tuple[int, int] = (1, 4)
    ccr: tuple[float, float] = (0.1, 10.0)
    deadline_tightness: tuple[float, float] = (0.1, 1.0)
    fpga_pressure: tuple[float, float] = (0.5, 2.0)
    task_base_clip_sd: float = 1.5
    sw_unc: tuple[float, float] = (0.9, 1.2)
    hw_time_unc: tuple[float, float] = (0.9, 1.15)
    extra: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def load_fits() -> dict:
    return json.loads(FITS_PATH.read_text(encoding="utf-8"))


# --- topology ------------------------------------------------------------------------------------------


def _topology(kind: str, n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    if kind == "layered":
        width = max(1, round(math.sqrt(n)))
        layer = [min(i // width, n) for i in range(n)]
        for j in range(n):
            if layer[j] == 0:
                continue
            prev = [i for i in range(j) if layer[i] == layer[j] - 1] or list(range(j))
            for i in rng.choice(prev, size=min(len(prev), int(rng.integers(1, 3))), replace=False):
                edges.add((int(i), j))
    elif kind == "sameprob":
        p = float(rng.choice([0.1, 0.2])) * max(1.0, 20.0 / n)
        for j in range(1, n):
            for i in range(j):
                if rng.random() < min(p, 0.9):
                    edges.add((i, j))
    elif kind == "samepred":
        avg = int(rng.choice([1, 2, 3]))
        for j in range(1, n):
            k = int(min(j, max(1, rng.poisson(avg))))
            for i in rng.choice(j, size=k, replace=False):
                edges.add((int(i), j))
    elif kind == "series_parallel":
        def build(lo: int, hi: int) -> tuple[list[int], list[int]]:  # returns (entry nodes, exit nodes)
            if hi - lo == 1:
                return [lo], [lo]
            mid = int(rng.integers(lo + 1, hi))
            a, b = build(lo, mid), build(mid, hi)
            if rng.random() < 0.5:  # series
                for u in a[1]:
                    for v in b[0]:
                        edges.add((u, v))
                return a[0], b[1]
            return a[0] + b[0], a[1] + b[1]  # parallel
        build(0, n)
    else:
        raise ValueError(f"unknown topology {kind!r}")
    for j in range(1, n):  # Kasahara-style: connect isolated nodes to an earlier node
        if not any(d == j for _, d in edges) and not any(s == j for s, _ in edges) and rng.random() < 0.8:
            edges.add((int(rng.integers(0, j)), j))
    return sorted(edges)


# --- instance --------------------------------------------------------------------------------------------


def _platform(pid: str, n_cpus: int, fpga: str, speeds: list[float]) -> Platform:
    pes = [{"id": f"cpu{i}", "type": f"cpu.k{i}", "kind": "cpu", "concurrency": "exclusive"} for i in range(n_cpus)]
    pes.append({"id": "fpga0", "type": f"fpga.{fpga}", "kind": "fpga", "concurrency": "spatial",
                "capacity": dict(FPGAS[fpga])})
    cpus = [p["id"] for p in pes[:-1]]
    itfs = [{"id": "axi_dma", "kind": "axi_dma", "connects": cpus + ["fpga0"], "setup_us": 20, "bandwidth_mb_s": 600,
             "shared": True}]
    if n_cpus > 1:
        itfs.append({"id": "l2_shared", "kind": "shared_memory", "connects": cpus, "setup_us": 1,
                     "bandwidth_mb_s": 1000, "shared": True})
    return Platform.model_validate({
        "id": pid, "name": f"Tier A {VERSION} platform ({n_cpus} CPU + {fpga}, synthetic)",
        "provenance": {"kind": "mixed", "source": f"datagen.tier_a {VERSION}",
                       "notes": f"FPGA capacity real ({fpga}); CPU speed offsets (log10) {speeds} from E3S fit; "
                                "interfaces synthetic placeholders."},
        "pes": pes, "interfaces": itfs,
    })


def generate(seed: int, cfg: TierAConfig | None = None) -> tuple[TaskGraph, Platform]:
    """One synthetic Tier A instance (task graph + platform), deterministic in `seed`."""
    from estimate import annotate  # local import: avoid a cycle at package import
    from solvers.heuristics import all_sw

    cfg = cfg or TierAConfig()
    fits = load_fits()
    rng = np.random.default_rng([20261020, seed])
    sw, rs, fr = fits["software_time"], fits["fpga_resources"]["per_resource"], fits["implementation_fronts"]

    n = int(rng.integers(cfg.n_tasks[0], cfg.n_tasks[1] + 1))
    topo = str(rng.choice(cfg.topologies))
    n_cpus = int(rng.choice(cfg.n_cpus))
    fpga = str(rng.choice(cfg.fpgas))
    edges = _topology(topo, n, rng)
    speeds = sorted(float(x) for x in rng.choice(sw["pe_speed_offsets"], size=n_cpus, replace=False))
    pid = f"tierA_{VERSION}_s{seed}_platform"
    platform = _platform(pid, n_cpus, fpga, speeds)

    fronts = list(fr["fronts"].values())
    nodes, hw_demand = [], {r: 0.0 for r in FPGAS[fpga]}
    raw_nodes = []
    for t in range(n):
        a = float(np.clip(rng.normal(sw["task_base_mean"], sw["task_base_sd"]),
                          sw["task_base_mean"] - cfg.task_base_clip_sd * sw["task_base_sd"],
                          sw["task_base_mean"] + cfg.task_base_clip_sd * sw["task_base_sd"]))
        sw_times = [10 ** (a + b + rng.normal(0, sw["noise_sd"])) for b in speeds]
        impls = [{"id": f"sw{i}", "pe_type": f"cpu.k{i}",
                  "cost": {"time_ms": {"expr": round(tm, 6), "unc": list(cfg.sw_unc)}}}
                 for i, tm in enumerate(sw_times)]
        hw = []
        if rng.random() < cfg.hw_task_fraction:
            s = 10 ** rng.normal(fits["hw_speedup_prior"]["log10_median"], fits["hw_speedup_prior"]["log10_sd"])
            t_fast = min(sw_times) / s
            front = fronts[int(rng.integers(len(fronts)))]
            k = int(min(len(front), rng.integers(cfg.impl_options[0], cfg.impl_options[1] + 1)))
            pick = sorted({round(i * (len(front) - 1) / max(k - 1, 1)) for i in range(k)})
            base = {}
            for r_ir, r_q in (("LUT", "LUT"), ("FF", "FF"), ("DSP", "DSP"), ("BRAM18K", "BRAM")):
                q = rs[r_q]
                base[r_ir] = 0.0 if rng.random() < q["fraction_zero"] else \
                    10 ** rng.normal(q["log10_mean_positive"], q["log10_sd_positive"])
            for j, i in enumerate(pick):
                pt = front[i]
                res = {r_ir: base[r_ir] * pt[f"{r_q}_ratio"] if base[r_ir] > 0 else 0.0
                       for r_ir, r_q in (("LUT", "LUT"), ("FF", "FF"), ("DSP", "DSP"), ("BRAM18K", "BRAM"))}
                hw.append((t_fast * pt["latency_ratio"], res))
            for r in hw_demand:
                hw_demand[r] += hw[0][1][r]
        raw_nodes.append((impls, hw))

    # scale HW resources so that the fastest options would demand `pressure` × the binding capacity
    pressure = float(rng.uniform(*cfg.fpga_pressure))
    cap = FPGAS[fpga]
    worst = max((hw_demand[r] / cap[r] for r in cap if hw_demand[r] > 0), default=0.0)
    scale = pressure / worst if worst > 0 else 1.0
    for t, (impls, hw) in enumerate(raw_nodes):
        for j, (tm, res) in enumerate(hw):
            impls.append({
                "id": f"hw{j}", "pe_type": f"fpga.{fpga}",
                "cost": {"time_ms": {"expr": round(tm, 6), "unc": list(cfg.hw_time_unc)},
                         "resources": {r: {"expr": round(v * scale, 3),
                                           "unc": rs["BRAM" if r == "BRAM18K" else r]["unc_factors"]}
                                       for r, v in res.items() if v > 0}},
            })
        nodes.append({"id": f"t{t}", "implementations": impls})

    # edge bytes: log-uniform, rescaled to the sampled CCR over the AXI-DMA link (600 MB/s)
    ccr = float(math.exp(rng.uniform(math.log(cfg.ccr[0]), math.log(cfg.ccr[1]))))
    raw = [math.exp(rng.uniform(math.log(64), math.log(4e6))) for _ in edges]
    mean_comp = float(np.mean([min(i["cost"]["time_ms"]["expr"] for i in nd["implementations"]) for nd in nodes]))
    if raw:
        mean_comm = float(np.mean(raw)) / 600e3
        raw = [b * ccr * mean_comp / mean_comm for b in raw]
    edge_json = [{"src": f"t{u}", "dst": f"t{v}", "bytes": round(max(b, 1.0), 1)} for (u, v), b in zip(edges, raw,
                                                                                                         strict=True)]
    params = {"version": VERSION, "seed": seed, "n_tasks": n, "topology": topo, "n_cpus": n_cpus, "fpga": fpga,
              "cpu_speed_offsets_log10": speeds, "ccr": ccr, "fpga_pressure": pressure, "config": asdict(cfg)}
    gid = f"tierA_{VERSION}_s{seed}"
    base_graph = {
        "id": gid, "name": f"Tier A {VERSION} seed {seed} (synthetic)",
        "provenance": {"kind": "synthetic", "source": f"datagen.tier_a {VERSION}; fits {FITS_PATH.name}",
                       "notes": json.dumps(params)},
        "workload": {}, "arrival": {"kind": "one_shot"}, "objective": {"weights": {"latency": 1.0}},
        "platform": {"id": pid}, "nodes": nodes, "edges": edge_json,
    }
    # deadline between the lower bound and the all-SW makespan
    from diagnostics import lower_bound
    from sim import evaluate

    prob = annotate(TaskGraph.model_validate(base_graph), platform)
    lb = lower_bound(prob).latency_ms
    allsw = evaluate(prob, all_sw(prob)).sim.makespan
    lam = float(rng.uniform(*cfg.deadline_tightness))
    base_graph["constraints"] = {"deadline_ms": round(lb + lam * max(allsw - lb, 0.0), 6)}
    params.update({"deadline_lambda": lam, "lower_bound_ms": lb, "all_sw_makespan_ms": allsw})
    base_graph["provenance"]["notes"] = json.dumps(params)
    return TaskGraph.model_validate(base_graph), platform
