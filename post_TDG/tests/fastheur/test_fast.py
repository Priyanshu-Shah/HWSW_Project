import json, sys, time
from pathlib import Path
import numpy as np
import pytest
from fastheur.instance import from_record
from fastheur.fast import Fast
from fastheur.sim import simulate

DATA = Path(__file__).resolve().parents[2] / "data"


def _recs(k=25):
    out = []
    with open(DATA / "labels/tier_a_v0_bus.synthetic.jsonl") as f:
        for line in f:
            out.append(json.loads(line))
            if len(out) >= k:
                break
    return out


def test_fast_matches_reference_rank_and_key():
    rng = np.random.default_rng(0)
    for r in _recs():
        inst = from_record(r)
        fa = Fast(inst)
        for _ in range(8):
            opt = np.array([rng.choice(inst.task_opts[i]) for i in range(inst.n)])
            ref = simulate(inst, opt)
            mk, ex = fa.evaluate(opt)
            assert mk == pytest.approx(ref.makespan, rel=1e-12)
            assert ex == pytest.approx(ref.cap_excess, rel=1e-9, abs=1e-12)
            key = rng.random(inst.n)
            order = list(np.argsort(key, kind="stable"))
            ref = simulate(inst, opt, order)
            mk, ex = fa.evaluate(opt, key)
            assert mk == pytest.approx(ref.makespan, rel=1e-12)
