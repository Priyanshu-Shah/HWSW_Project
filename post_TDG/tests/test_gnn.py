"""GNN partitioner plumbing: features, label targets, training signal, decode legality. Skipped without torch."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from conftest import ROOT  # noqa: E402
from estimate import annotate  # noqa: E402
from schemas.ir import TaskGraph  # noqa: E402
from schemas.platform import Platform  # noqa: E402
from solvers.gnn import GnnPartitioner, attach_label, decode, featurize, imitation_loss  # noqa: E402
from solvers.gnn.data import N_EDGE_FEAT, N_OPT_FEAT, N_TASK_FEAT  # noqa: E402

LABELS = ROOT / "data" / "labels" / "tier_a_v0_bus.synthetic.jsonl"


def _items(k: int):
    out = []
    with LABELS.open(encoding="utf-8") as fh:
        lines = fh.readlines()
    for line in lines:
        r = json.loads(line)
        if r["label"]["status"] != "optimal":
            continue
        p = annotate(TaskGraph.model_validate(r["graph"]), Platform.model_validate(r["platform"]))
        out.append((p, attach_label(featurize(p), p, r["label"]["mapping"], r["label"]["priority"])))
        if len(out) == k:
            return out
    return out


@pytest.mark.skipif(not LABELS.exists(), reason="labels not generated")
def test_features_and_targets_are_consistent():
    (p, d), = _items(1)
    n = len(p.order)
    assert d.task_x.shape == (n, N_TASK_FEAT) and d.opt_x.shape[1] == N_OPT_FEAT
    assert d.edge_x.shape == (len(p.edges), N_EDGE_FEAT)
    assert d.opt_x.shape[0] == sum(len(p.options[t]) for t in p.order)
    assert torch.isfinite(d.task_x).all() and torch.isfinite(d.opt_x).all()
    assert all(int(d.opt_task[int(d.y_opt[i])]) == i for i in range(n))  # each target option belongs to its task
    assert sum(1 for kind, _ in d.y_items if kind == "t") == n


@pytest.mark.skipif(not LABELS.exists(), reason="labels not generated")
def test_imitation_loss_decreases_and_decode_is_legal():
    torch.manual_seed(0)
    items = _items(3)
    m = GnnPartitioner(hidden=32, layers=2)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    first = sum(imitation_loss(m, d)[0].item() for _, d in items)
    for _ in range(40):
        for _, d in items:
            loss, _ = imitation_loss(m, d)
            opt.zero_grad()
            loss.backward()
            opt.step()
    assert sum(imitation_loss(m, d)[0].item() for _, d in items) < first
    for p, d in items:
        res = decode(m, p, d, samples=4)
        assert all(res.mapping[t] in p.options[t] for t in p.order)
        assert not [v for v in res.ev.violations if v.constraint.startswith("capacity")]
