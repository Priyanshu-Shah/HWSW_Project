"""GCPS (Zheng, Liang & Xiong, DAES 25, 2021): a per-instance, unsupervised 2-layer GCN partitioner.

Implemented from docs/survey/slices/07_gcps_reproduction.md §3. Decisions for what the paper leaves open (its §8):
- cosine edge weights mapped to [0, 1] as (cos + 1) / 2, recomputed at every layer from that layer's input H^l,
  inside autograd, on existing edges only; the diagonal keeps Â's renormalised value (Eq. 5);
- dropout 0.5 after each graph convolution, Glorot-initialised layers with bias, Adam (lr 1e-3, Table 3);
- loss T_cost = Σ(t_h x + t_s (1 − x)) on raw times (Adam is scale-invariant);
- 100 pre-training steps; each epoch = s = 5 gradient steps, then one evaluation (no dropout) + greedy + LSSP;
- F1 in X' = (1 − σ) X + σ F1 is the HGP feature min-max scaled to [0, 1];
- greedy = stop-at-first-overflow cut line; only a strict improvement resets the stopping counter (with
  ties resetting it, 20-node runs never stopped, contradicting the paper's ~0.5 s); stop when
  epoch − best_epoch ≥ k or 800 epochs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from solvers.gcps.instance import BinInstance
from solvers.gcps.lssp import greedy_area, lssp


@dataclass
class GcpsResult:
    y: np.ndarray
    t_sl: float
    ar: float
    epochs: int
    best_epoch: int
    runtime_s: float


class _Net(nn.Module):
    def __init__(self, src: torch.Tensor, dst: torch.Tensor, diag: torch.Tensor, n: int, drop: float):
        super().__init__()
        self.src, self.dst, self.diag, self.n = src, dst, diag, n
        self.l0, self.l1, self.fc = nn.Linear(2, 10), nn.Linear(10, 5), nn.Linear(5, 2)
        for m in (self.l0, self.l1, self.fc):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        self.drop = nn.Dropout(drop)

    def _conv(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """A^l z with A^l = diag(Â) + cosine weights (from h) on the undirected edges; O(E) message passing."""
        hn = nn.functional.normalize(h, dim=1, eps=1e-12)
        w = (((hn[self.src] * hn[self.dst]).sum(1) + 1.0) / 2.0).unsqueeze(1)
        out = self.diag.unsqueeze(1) * z
        out = out.index_add(0, self.dst, w * z[self.src])
        return out.index_add(0, self.src, w * z[self.dst])

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        h1 = self.drop(torch.relu(self._conv(h0, self.l0(h0))))
        h2 = self.drop(torch.relu(self._conv(h1, self.l1(h1))))
        return torch.softmax(self.fc(h2), dim=1)[:, 1]  # probability of HW


def gcps(
    inst: BinInstance,
    seed: int = 0,
    lr: float = 1e-3,
    sigma: float = 0.3,
    alpha: int = 5,
    s_steps: int = 5,
    pre_steps: int = 100,
    max_epoch: int = 800,
    drop: float = 0.5,
    threads: int | None = 1,
) -> GcpsResult:
    """`threads=1` by default: with PyTorch's default thread count the tiny per-step ops ran 11x slower."""
    t0 = time.perf_counter()
    if threads is not None:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    n = inst.n
    k = n // alpha if n <= 200 else n // (2 * alpha)
    und = sorted({(min(i, j), max(i, j)) for i, j, _ in inst.edges if i != j})
    src = torch.tensor([a for a, _ in und], dtype=torch.long)
    dst = torch.tensor([b for _, b in und], dtype=torch.long)
    deg = torch.ones(n, dtype=torch.float64)  # Ã = A + I
    for a, b in und:
        deg[a] += 1
        deg[b] += 1
    diag = 1.0 / deg  # D̃^{-1/2} Ĩ D̃^{-1/2} on the diagonal
    hg = inst.ts - inst.th
    hgp = hg / inst.area
    f = np.stack([hg, hgp], axis=1)
    h0 = torch.tensor((f - f.mean(0)) / (f.std(0) + 1e-12), dtype=torch.float64)  # population std
    f1 = (hgp - hgp.min()) / (np.ptp(hgp) or 1.0)
    ts, th = torch.tensor(inst.ts), torch.tensor(inst.th)
    net = _Net(src, dst, diag, n, drop).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def step() -> None:
        net.train()
        x = net(h0)
        loss = (th * x + ts * (1 - x)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    for _ in range(pre_steps):
        step()
    best_t, best_y, best_epoch, epoch = float("inf"), np.zeros(n, dtype=np.int8), 0, 0
    for epoch in range(max_epoch):
        for _ in range(s_steps):
            step()
        net.eval()
        with torch.no_grad():
            x = net(h0).numpy()
        y = greedy_area((1 - sigma) * x + sigma * f1, inst.area, inst.C)
        t = lssp(inst, y)
        if t < best_t - 1e-9:  # strict: ties do not reset the stopping counter (else plateaus never stop)
            best_t, best_y, best_epoch = t, y, epoch
        if epoch - best_epoch >= k:
            break
    return GcpsResult(best_y, best_t, inst.t_max / best_t, epoch + 1, best_epoch, time.perf_counter() - t0)
