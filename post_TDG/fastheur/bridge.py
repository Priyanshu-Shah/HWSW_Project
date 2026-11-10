"""Bridge between the handover partitioner (Problem, sim.evaluate, exact solver, GNN) and the fast heuristics
in this package (Instance / Fast and the anytime SA, tabu, GA).

Both are built from the same graph + platform JSON, so task names and "<pe>/<impl>" option names match one to
one. This module converts mappings between them and loads the GNN. (From the Thor run 2 `combo/bridge.py`.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- mapping conversion
def to_opt(inst, mapping) -> np.ndarray:
    """handover mapping {task: Option(pe, impl)} -> Thakar's global option id per task."""
    name_to_k = {(inst.o_task[k], inst.o_name[k]): k for k in range(len(inst.o_name))}
    opt = np.empty(inst.n, dtype=np.int64)
    for i, t in enumerate(inst.tnames):
        o = mapping[t]
        opt[i] = name_to_k[(i, f"{o.pe}/{o.impl}")]
    return opt


def from_opt(inst, prob, opt) -> dict:
    """Thakar's option ids -> handover mapping {task: Option}."""
    out = {}
    for i, t in enumerate(inst.tnames):
        pe, impl = inst.o_name[int(opt[i])].split("/", 1)
        out[t] = next(o for o in prob.options[t] if o.pe == pe and o.impl == impl)
    return out


def handover_priority(inst, order) -> list[str] | None:
    """Thakar's order (task ids, transfers as n + edge id) -> handover priority names; None stays None (rank)."""
    if order is None:
        return None
    from sim.scheduler import transfer_id
    n = inst.n
    return [inst.tnames[x] if x < n else transfer_id(inst.tnames[int(inst.e_src[x - n])], inst.tnames[int(inst.e_dst[x - n])])
            for x in map(int, order)]


def keys_from_task_priority(inst, task_order: list[str]) -> np.ndarray:
    """GA genome keys (smaller = earlier) from a task priority list (best first)."""
    pos = {t: j for j, t in enumerate(task_order)}
    n = inst.n
    return np.array([pos.get(t, n + i) / max(n, 1) for i, t in enumerate(inst.tnames)], dtype=float)


# ---------------------------------------------------------------- GNN
@dataclass
class Gnn:
    model: object

    @staticmethod
    def load(path: Path) -> "Gnn":
        import torch
        from solvers.gnn import GnnPartitioner
        torch.set_num_threads(1)
        ck = torch.load(path, weights_only=False, map_location="cpu")
        m = GnnPartitioner(hidden=ck["hidden"], layers=ck["layers"])
        m.load_state_dict(ck["state_dict"])
        m.eval()
        return Gnn(m)

    def featurize(self, prob):
        from solvers.gnn import featurize
        return featurize(prob)

    def decode(self, prob, data, samples: int = 0, seed: int = 0):
        """Decoded(mapping, priority, ev, runtime_s); priority holds tasks and transfers ("a->b")."""
        from solvers.gnn import decode
        return decode(self.model, prob, data, samples=samples, seed=seed)


def task_order(priority: list[str]) -> list[str]:
    return [x for x in priority if "->" not in x]


# ---------------------------------------------------------------- fast GNN population sampler (B1)
class GnnSampler:
    """One GNN forward pass, then mappings sampled from the option distribution in numpy and scored by Thakar's
    compiled evaluator (microseconds each), instead of the handover's Python simulator (milliseconds each).

    Everything in `__init__` and `population` is meant to run inside the method's clock."""

    def __init__(self, gnn: "Gnn", prob, data, inst):
        import torch
        with torch.no_grad():
            logit, tp, _ = gnn.model(data)
        logit = logit.numpy().astype(float)
        name_to_k = {(inst.o_task[k], inst.o_name[k]): k for k in range(len(inst.o_name))}
        tindex = {t: i for i, t in enumerate(inst.tnames)}
        per = [[] for _ in range(inst.n)]  # per task: [(thakar option id, logit)]
        for k, (t, o) in enumerate(data.options):
            i = tindex[t]
            per[i].append((name_to_k[(i, f"{o.pe}/{o.impl}")], logit[k]))
        width = max(len(x) for x in per)
        self.ids = np.full((inst.n, width), -1, dtype=np.int64)
        self.lg = np.full((inst.n, width), -np.inf)
        for i, x in enumerate(per):
            for j, (k, v) in enumerate(x):
                self.ids[i, j], self.lg[i, j] = k, v
        tpn = tp.numpy()
        order = [data.order[i] for i in np.argsort(-tpn)]
        self.keys = keys_from_task_priority(inst, order)

    def greedy(self) -> np.ndarray:
        return self.ids[np.arange(len(self.ids)), np.argmax(self.lg, axis=1)]

    def sample(self, m: int, tau: float, rng) -> np.ndarray:
        g = rng.gumbel(size=(m, *self.lg.shape)) * tau
        j = np.argmax(self.lg[None] + g, axis=2)
        return self.ids[np.arange(self.ids.shape[0])[None, :], j]

    def population(self, fast, m: int = 256, keep: int = 8, tau: float = 1.0, seed: int = 0,
                   time_cap_s: float | None = None) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
        """Greedy + up to m samples (stops early at time_cap_s); returns the `keep` best as
        (opt, ga keys, makespan, excess), best first. Excess > 0 means a capacity / deadline violation."""
        import time as _t
        t0 = _t.perf_counter()
        rng = np.random.default_rng(seed)
        cand = [self.greedy()]
        batch = 32
        while len(cand) < m + 1:
            cand.extend(self.sample(min(batch, m + 1 - len(cand)), tau, rng))
            if time_cap_s is not None and _t.perf_counter() - t0 > time_cap_s:
                break
        seen, scored = set(), []
        for c in cand:
            key = c.tobytes()
            if key in seen:
                continue
            seen.add(key)
            mk, ex = fast.evaluate(c)
            scored.append((ex > 0, mk, ex, c))
            if time_cap_s is not None and _t.perf_counter() - t0 > time_cap_s:
                break
        scored.sort(key=lambda z: (z[0], z[2], z[1]))
        out = []
        for _, mk, ex, c in scored[:keep]:
            k = self.keys + rng.normal(0, 0.02, size=self.keys.shape) * (len(out) > 0)
            out.append((c.copy(), k, mk, ex))
        return out
