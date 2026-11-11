"""GNN partitioner: directed message passing over the task DAG; heads for option choice, task priority and transfer
priority (survey §7.4; transfer priorities are required to reproduce bus-contention optima, see DECISIONS 2026-10-20).
"""

from __future__ import annotations

import torch
from torch import nn

from solvers.gnn.data import N_EDGE_FEAT, N_OPT_FEAT, N_TASK_FEAT, GraphData


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class _Layer(nn.Module):
    """h' = h + MLP([h, mean of predecessor messages, mean of successor messages])."""

    def __init__(self, h: int):
        super().__init__()
        self.fwd = nn.Linear(h + N_EDGE_FEAT, h)
        self.bwd = nn.Linear(h + N_EDGE_FEAT, h)
        self.upd = _mlp(3 * h, h, h)
        self.norm = nn.LayerNorm(h)

    def forward(self, h: torch.Tensor, ei: torch.Tensor, ex: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        src, dst = ei
        agg_in = torch.zeros_like(h).index_add(0, dst, torch.relu(self.fwd(torch.cat([h[src], ex], 1))))
        agg_out = torch.zeros_like(h).index_add(0, src, torch.relu(self.bwd(torch.cat([h[dst], ex], 1))))
        din = torch.zeros(n, 1).index_add(0, dst, torch.ones(len(dst), 1)).clamp(min=1)
        dout = torch.zeros(n, 1).index_add(0, src, torch.ones(len(src), 1)).clamp(min=1)
        return self.norm(h + self.upd(torch.cat([h, agg_in / din, agg_out / dout], 1)))


class GnnPartitioner(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 4):
        super().__init__()
        self.enc = _mlp(N_TASK_FEAT, hidden, hidden)
        self.layers = nn.ModuleList(_Layer(hidden) for _ in range(layers))
        self.opt_head = _mlp(hidden + N_OPT_FEAT, hidden, 1)
        self.task_prio = _mlp(hidden, hidden, 1)
        self.edge_prio = _mlp(2 * hidden + N_EDGE_FEAT, hidden, 1)

    def embed(self, d: GraphData) -> torch.Tensor:
        h = self.enc(d.task_x)
        for layer in self.layers:
            h = layer(h, d.edge_index, d.edge_x)
        return h

    def forward(self, d: GraphData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(option logits (K,), task priority scores (n,), edge priority scores (E,))."""
        h = self.embed(d)
        opt_logit = self.opt_head(torch.cat([h[d.opt_task], d.opt_x], 1)).squeeze(1)
        tp = self.task_prio(h).squeeze(1)
        src, dst = d.edge_index
        ep = self.edge_prio(torch.cat([h[src], h[dst], d.edge_x], 1)).squeeze(1)
        return opt_logit, tp, ep


def segment_log_softmax(logits: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    """log-softmax of option logits within each task."""
    mx = torch.full((n,), -1e30).scatter_reduce(0, seg, logits, reduce="amax")
    z = torch.exp(logits - mx[seg])
    s = torch.zeros(n).index_add(0, seg, z)
    return logits - mx[seg] - torch.log(s[seg])


def imitation_loss(model: GnnPartitioner, d: GraphData, w_rank: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Cross-entropy on the labelled option per task + pairwise rank loss on the labelled unified order
    (NeuroSchedule-style: earlier item should score higher; log(1 + exp(-(s_i - s_j))) over all ordered pairs)."""
    n = len(d.order)
    logit, tp, ep = model(d)
    lsm = segment_log_softmax(logit, d.opt_task, n)
    ce = -lsm[d.y_opt].mean()
    scores = torch.stack([tp[i] if kind == "t" else ep[i] for kind, i in d.y_items])
    m = len(scores)
    if m > 1:
        diff = scores.unsqueeze(1) - scores.unsqueeze(0)  # s_i - s_j
        earlier = torch.triu(torch.ones(m, m, dtype=torch.bool), diagonal=1)  # i before j in the label
        rank = nn.functional.softplus(-diff[earlier]).mean()
    else:
        rank = torch.zeros(())
    with torch.no_grad():  # labelled option has the highest probability of its task
        best = torch.full((n,), -1e30).scatter_reduce(0, d.opt_task, lsm, reduce="amax")
        acc = (lsm[d.y_opt] >= best - 1e-9).float().mean()
    return ce + w_rank * rank, {"ce": ce.item(), "rank": rank.item(), "acc": acc.item()}
