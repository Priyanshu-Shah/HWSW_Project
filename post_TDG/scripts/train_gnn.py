"""Train the GNN partitioner by imitation of CP-SAT bus-contention labels and evaluate it against the optimum.

Data (all SYNTHETIC, Tier A v0):
- train/val: data/labels/tier_a_v0_bus_train.synthetic.jsonl (last `--val` optimal records are validation);
- test:      data/labels/tier_a_v0_bus.synthetic.jsonl (seeds 0–199, never used for training);
- large:     data/labels/tier_a_v0_bus_large.synthetic.jsonl (60–100 tasks, if present: size generalisation).
Writes models/gnn_v0.pt and docs/benchmarks/gnn_imitation.md.

Usage: python scripts/train_gnn.py [--epochs 40] [--lr 1e-3] [--batch 8] [--samples 16]
One thread by default: these graphs are small, and 4 threads made a training step 2x slower.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from datetime import date
from pathlib import Path

import torch

from estimate import annotate
from schemas.ir import TaskGraph
from schemas.platform import Platform
from sim import evaluate
from solvers.gnn import GnnPartitioner, attach_label, decode, featurize, imitation_loss
from solvers.heuristics import plan

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "data" / "labels"


def load(path: Path, only_optimal: bool = True) -> list[tuple]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        lines = fh.readlines()
    for line in lines:
        r = json.loads(line)
        if "mapping" not in r["label"] or (only_optimal and r["label"]["status"] != "optimal"):
            continue
        p = annotate(TaskGraph.model_validate(r["graph"]), Platform.model_validate(r["platform"]))
        d = attach_label(featurize(p), p, r["label"]["mapping"], r["label"]["priority"])
        out.append((p, d, r["label"]["makespan_ms"], r["label"]["status"], r["id"]))
    return out


def evaluate_set(model: GnnPartitioner, items: list[tuple], samples: int,
                 baselines: tuple[str, ...] = ("heft", "peft", "magellan")) -> dict[str, list[tuple[float, bool, float]]]:
    res: dict[str, list[tuple[float, bool, float]]] = {"gnn": [], f"gnn+{samples}": [], **{h: [] for h in baselines}}
    for p, d, ref, _, _ in items:
        g = decode(model, p, d)
        res["gnn"].append((g.ev.sim.makespan / ref, g.ev.feasible, g.runtime_s))
        s = decode(model, p, d, samples=samples)
        res[f"gnn+{samples}"].append((s.ev.sim.makespan / ref, s.ev.feasible, s.runtime_s))
        for h in baselines:
            t0 = time.perf_counter()
            pl = plan(p, h)
            ev = evaluate(p, pl.mapping, priority=pl.priority, insertion=pl.insertion)
            res[h].append((ev.sim.makespan / ref, ev.feasible, time.perf_counter() - t0))
    return res


def table(title: str, res: dict, ref_note: str) -> list[str]:
    def q(xs: list[float], p: float) -> float:
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]

    n = len(next(iter(res.values())))
    lines = [f"### {title} ({n} instances; reference = {ref_note})", "",
             "| method | makespan / ref: median | mean | p90 | max | within 0.1% | deadline met | time median (ms) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, xs in res.items():
        r = [x[0] for x in xs]
        lines.append(f"| {k} | {statistics.median(r):.3f} | {statistics.mean(r):.3f} | {q(r, 0.9):.3f} | {max(r):.3f} | "
                     f"{sum(x <= 1.001 for x in r)}/{n} | {sum(x[1] for x in xs)}/{n} | "
                     f"{1000 * statistics.median(x[2] for x in xs):.1f} |")
    return lines + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8, help="graphs per optimiser step (gradient accumulation)")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    random.seed(0)

    train = load(LAB / "tier_a_v0_bus_train.synthetic.jsonl")
    val, train = train[-args.val:], train[: -args.val]
    test = load(LAB / "tier_a_v0_bus.synthetic.jsonl")
    large = load(LAB / "tier_a_v0_bus_large.synthetic.jsonl", only_optimal=False)
    print(f"train {len(train)}  val {len(val)}  test {len(test)}  large {len(large)}", flush=True)

    model = GnnPartitioner()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, foreach=True)
    best_val, best_state, log = float("inf"), None, []
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        random.shuffle(train)
        model.train()
        tot = {"ce": 0.0, "rank": 0.0, "acc": 0.0}
        for b0 in range(0, len(train), args.batch):
            batch = train[b0: b0 + args.batch]
            opt.zero_grad()
            for _, d, *_ in batch:
                loss, info = imitation_loss(model, d)
                (loss / len(batch)).backward()
                for k in tot:
                    tot[k] += info[k] / len(train)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = statistics.mean(imitation_loss(model, d)[0].item() for _, d, *_ in val)
        if ep % 5 == 4 or ep == args.epochs - 1:
            ratios = [decode(model, p, d).ev.sim.makespan / ref for p, d, ref, *_ in val]
            vr = statistics.mean(ratios)
        else:
            vr = float("nan")
        log.append((ep, tot["ce"], tot["rank"], tot["acc"], vl, vr))
        print(f"ep {ep:3d} ce {tot['ce']:.3f} rank {tot['rank']:.3f} acc {tot['acc']:.3f} | val loss {vl:.3f} "
              f"val ratio {vr:.3f} | {time.perf_counter() - t0:.0f}s", flush=True)
        if vl < best_val:
            best_val, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    train_s = time.perf_counter() - t0
    model.load_state_dict(best_state)
    (ROOT / "models").mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "hidden": 64, "layers": 4, "features": "solvers/gnn/data.py v0"},
               ROOT / "models" / "gnn_v0.pt")

    lines = [
        "# GNN partitioner v0: imitation of CP-SAT labels (Tier A v0, bus contention)",
        "",
        f"Generated by `scripts/train_gnn.py` on {date.today().isoformat()}. **All instances synthetic.** "
        f"Trained on {len(train)} optimal labels (10–40 tasks), validated on {len(val)}, best epoch by validation "
        f"loss; {args.epochs} epochs, Adam lr {args.lr:g}, {args.batch} graphs per step, {train_s:.0f} s on CPU. Every method's mapping is "
        "evaluated by the same simulator (bus contention). `gnn` = greedy decode; "
        f"`gnn+{args.samples}` = best of greedy + {args.samples} Gumbel samples.",
        "",
    ]
    lines += table("Test: Tier A v0, seeds 0–199 (unseen)", evaluate_set(model, test, args.samples),
                   "CP-SAT optimum")
    if large:
        lines += table("Size generalisation: 60–100 tasks (unseen sizes)",
                       evaluate_set(model, large, args.samples, baselines=("heft", "peft")),
                       "best CP-SAT solution within its time limit (optimal or not)")
        lines += ["MAGELLAN is omitted on this set: O(n²·K) simulator calls, 34 s on one 90-task graph.", ""]
    lines += ["### Training log (every epoch: train CE / rank / option accuracy, validation loss; "
              "validation makespan ratio every 5 epochs)", "",
              "| epoch | train CE | train rank | train acc | val loss | val ratio |", "|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {e} | {a:.3f} | {b:.3f} | {c:.3f} | {d_:.3f} | {r:.3f} |" for e, a, b, c, d_, r in log]
    out = ROOT / "docs" / "benchmarks" / "gnn_imitation.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
