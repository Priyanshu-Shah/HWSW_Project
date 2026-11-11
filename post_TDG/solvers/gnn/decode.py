"""Decode GNN outputs into a schedule: option per task (+ capacity repair), unified task + transfer priority list,
evaluated by the simulator. Greedy decoding, or best-of-K sampling (Gumbel noise on logits and scores)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from estimate.annotate import Option, Problem
from sim.evaluate import Evaluation, evaluate
from sim.scheduler import transfer_id
from solvers.gnn.data import GraphData
from solvers.gnn.model import GnnPartitioner, segment_log_softmax


@dataclass
class Decoded:
    mapping: dict[str, Option]
    priority: list[str]
    ev: Evaluation
    runtime_s: float
    samples: int


def _repair(problem: Problem, mapping: dict[str, Option], probs: dict[str, dict[Option, float]]) -> None:
    """While a PE is over capacity, move the task whose best alternative loses the least probability off it."""
    v = problem.nominal()
    for _ in range(len(mapping)):
        usage: dict[str, dict[str, float]] = {}
        for t, o in mapping.items():
            u = usage.setdefault(o.pe, {})
            for r, val in v.res[(t, o.impl)].items():
                u[r] = u.get(r, 0.0) + val
        over = [pe for pe, u in usage.items()
                if any(u.get(r, 0.0) > c * (1 + 1e-9) for r, c in problem.platform.pe(pe).capacity.items())]
        if not over:
            return
        pe = over[0]
        best = None
        for t, o in mapping.items():
            if o.pe != pe:
                continue
            for alt in problem.options[t]:
                if alt.pe == pe:
                    continue
                loss = probs[t][o] - probs[t][alt]
                if best is None or loss < best[0]:
                    best = (loss, t, alt)
        if best is None:
            return
        mapping[best[1]] = best[2]


def decode(model: GnnPartitioner, problem: Problem, data: GraphData, samples: int = 0, tau: float = 1.0,
           seed: int = 0) -> Decoded:
    """Greedy decode (samples = 0) or best of greedy + `samples` Gumbel-perturbed decodes (feasible first)."""
    t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        logit, tp, ep = model(data)
        lsm = segment_log_softmax(logit, data.opt_task, len(data.order)).numpy()
    tp, ep, logit_np = tp.numpy(), ep.numpy(), logit.numpy()
    rng = np.random.default_rng(seed)
    by_task: dict[str, list[int]] = {}
    for k, (t, _) in enumerate(data.options):
        by_task.setdefault(t, []).append(k)
    probs = {t: {data.options[k][1]: float(np.exp(lsm[k])) for k in ks} for t, ks in by_task.items()}
    eidx = {e: i for i, e in enumerate(problem.edges)}

    def one(noise: bool) -> tuple[dict[str, Option], list[str], Evaluation]:
        g = (lambda shape: rng.gumbel(size=shape) * tau) if noise else (lambda shape: np.zeros(shape))
        lg = logit_np + g(logit_np.shape)
        mapping = {t: data.options[max(ks, key=lambda k: lg[k])][1] for t, ks in by_task.items()}
        _repair(problem, mapping, probs)
        tpn, epn = tp + g(tp.shape), ep + g(ep.shape)
        items = [(tpn[i], t) for i, t in enumerate(data.order)]
        for (a, b), i in eidx.items():
            c = problem.comm(mapping[a].pe, mapping[b].pe, 1.0)
            if c is not None and c.interface is not None:
                items.append((epn[i], transfer_id(a, b)))
        priority = [x for _, x in sorted(items, key=lambda z: -z[0])]
        return mapping, priority, evaluate(problem, mapping, priority=priority)

    best = one(False)
    for _ in range(samples):
        cand = one(True)
        if (not cand[2].feasible, cand[2].objective) < (not best[2].feasible, best[2].objective):
            best = cand
    return Decoded(best[0], best[1], best[2], time.perf_counter() - t0, samples)
