"""Featurisation of a `Problem` for the GNN partitioner, and imitation targets from CP-SAT labels.

All features are index-free and normalised per graph (survey §7.4: Placeto/REGAL lessons), so one model applies to
graphs of any size and platform with the same feature schema.

Task features (per task): min / mean CPU time, min FPGA time (0 if none), has-HW flag, #options, upward and downward
rank (mean costs), in/out degree, total in/out transfer time, plus graph-level context: deadline / lower bound,
FPGA pressure (Σ smallest HW footprints / capacity, max over resources), #CPUs, #tasks / 100.
Option features (per legal (PE, implementation)): time / task's min time, time / graph time scale, is_cpu, is_fpga,
is_accelerator, CPU speed rank, resource use / capacity for LUT, FF, DSP, BRAM18K, mem_bytes.
Edge features (per same-frame edge): mean cross-PE transfer time / time scale, bytes (log-scaled).
Unbounded ratios are compressed with log1p, and option times are compared with the task's fastest option floored at
1e-3 of the graph's time scale: E3S-derived tasks with near-zero options had produced ratios of 1e10 (DECISIONS
2026-10-24).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from estimate.annotate import Option, Problem
from solvers.heuristics.list_sched import upward_ranks

RES = ["LUT", "FF", "DSP", "BRAM18K", "mem_bytes"]
N_TASK_FEAT = 15
N_OPT_FEAT = 11
N_EDGE_FEAT = 2


@dataclass
class GraphData:
    """Tensors for one problem. Options are flattened; `opt_task[k]` is the task index of option k."""

    task_x: torch.Tensor  # (n, N_TASK_FEAT)
    opt_x: torch.Tensor  # (K, N_OPT_FEAT)
    opt_task: torch.Tensor  # (K,)
    edge_index: torch.Tensor  # (2, E) same-frame edges, src -> dst (task indices)
    edge_x: torch.Tensor  # (E, N_EDGE_FEAT)
    options: list[tuple[str, Option]]  # (task id, option) for each flattened option
    order: list[str]  # task ids, index = task index
    # imitation targets (None when unlabelled)
    y_opt: torch.Tensor | None = None  # (n,) index into the flattened options of the labelled choice
    y_items: list[tuple[str, int]] = field(default_factory=list)  # label order: ("t", task) or ("e", edge index)


def featurize(problem: Problem) -> GraphData:
    v = problem.nominal()
    order = problem.order
    idx = {n: i for i, n in enumerate(order)}
    n = len(order)
    cpus = [pe for pe in problem.pes if problem.pe_kind(pe) == "cpu"]
    rank = upward_ranks(problem)
    # downward rank (t-level) with mean costs
    mean_t = {t: float(np.mean([v.time[(t, o.impl)] for o in problem.options[t]])) for t in order}
    down: dict[str, float] = {}
    for t in order:
        down[t] = max((down[p] + mean_t[p] for p in problem.preds[t]), default=0.0)
    scale = max(float(np.mean(list(mean_t.values()))), 1e-12)

    def mean_comm(a: str, b: str) -> float:
        nb = v.bytes[(a, b)]
        ts = [c.time_ms for p in {o.pe for o in problem.options[a]} for q in {o.pe for o in problem.options[b]}
              if p != q and (c := problem.comm(p, q, nb)) is not None]
        return float(np.mean(ts)) if ts else 0.0

    ecomm = {e: mean_comm(*e) for e in problem.edges}
    lb = max(problem.norms["latency"], 1e-12)
    pressure = 0.0
    for pe in problem.pes:
        for r, cap in problem.platform.pe(pe).capacity.items():
            need = sum(min((v.res[(t, o.impl)].get(r, 0.0) for o in problem.options[t] if o.pe == pe), default=0.0)
                       for t in order)
            pressure = max(pressure, need / cap if cap else 0.0)
    ctx = [math.log1p((problem.deadline_ms or 0.0) / lb), pressure, len(cpus), n / 100.0]
    rmax = max(rank.values()) or 1.0
    dmax = max(down.values()) or 1.0

    lg = math.log1p
    task_x = []
    for t in order:
        sw = [v.time[(t, o.impl)] for o in problem.options[t] if problem.pe_kind(o.pe) == "cpu"]
        hw = [v.time[(t, o.impl)] for o in problem.options[t] if problem.pe_kind(o.pe) != "cpu"]
        task_x.append([
            lg((min(sw) if sw else 0.0) / scale), lg((float(np.mean(sw)) if sw else 0.0) / scale),
            lg((min(hw) if hw else 0.0) / scale), 1.0 if hw else 0.0, len(problem.options[t]) / 4.0,
            rank[t] / rmax, down[t] / dmax, len(problem.preds[t]) / 4.0, len(problem.succs[t]) / 4.0,
            lg(sum(ecomm[(p, t)] for p in problem.preds[t]) / scale),
            lg(sum(ecomm[(t, s)] for s in problem.succs[t]) / scale), *ctx,
        ])
    cpu_rank = {pe: i / max(len(cpus) - 1, 1) for i, pe in
                enumerate(sorted(cpus, key=lambda pe: np.mean([v.time[(t, o.impl)] for t in order
                                                               for o in problem.options[t] if o.pe == pe] or [0])))}
    opt_x, opt_task, options = [], [], []
    for t in order:
        tmin = max(min(v.time[(t, o.impl)] for o in problem.options[t]), 1e-3 * scale)
        for o in problem.options[t]:
            kind = problem.pe_kind(o.pe)
            cap = problem.platform.pe(o.pe).capacity
            res = v.res[(t, o.impl)]
            opt_x.append([
                lg(v.time[(t, o.impl)] / tmin), lg(v.time[(t, o.impl)] / scale), float(kind == "cpu"),
                float(kind == "fpga"),
                float(kind not in ("cpu", "fpga")), cpu_rank.get(o.pe, 0.0),
                *[res.get(r, 0.0) / cap[r] if cap.get(r) else 0.0 for r in RES],
            ])
            opt_task.append(idx[t])
            options.append((t, o))
    ei = [[idx[a] for a, _ in problem.edges], [idx[b] for _, b in problem.edges]]
    ex = [[lg(ecomm[e] / scale), math.log10(1.0 + v.bytes[e]) / 7.0] for e in problem.edges]
    f = torch.float32
    return GraphData(
        torch.tensor(task_x, dtype=f), torch.tensor(opt_x, dtype=f), torch.tensor(opt_task, dtype=torch.long),
        torch.tensor(ei, dtype=torch.long).reshape(2, -1), torch.tensor(ex, dtype=f).reshape(-1, N_EDGE_FEAT),
        options, list(order),
    )


def attach_label(data: GraphData, problem: Problem, mapping: dict[str, list[str]], priority: list[str]) -> GraphData:
    """Targets from a CP-SAT label: chosen option per task, and the unified task + transfer order."""
    flat = {(t, o.pe, o.impl): k for k, (t, o) in enumerate(data.options)}
    data.y_opt = torch.tensor([flat[(t, *mapping[t])] for t in data.order], dtype=torch.long)
    eidx = {e: i for i, e in enumerate(problem.edges)}
    tidx = {t: i for i, t in enumerate(data.order)}
    items = []
    for it in priority:
        if "->" in it:
            a, b = it.split("->")
            items.append(("e", eidx[(a, b)]))
        else:
            items.append(("t", tidx[it]))
    data.y_items = items
    return data
