import json
from pathlib import Path
import numpy as np
import pytest
from fastheur.instance import from_record, from_gcps
from fastheur.fast import Fast
from fastheur.algos.heft import heft, peft
from fastheur.algos.common import Plan

DATA = Path(__file__).resolve().parents[2] / "data"


def _recs(k):
    out = []
    with open(DATA / "labels/tier_a_v0_bus.synthetic.jsonl") as f:
        for line in f:
            out.append(json.loads(line))
            if len(out) >= k:
                break
    return out


@pytest.mark.parametrize("which", ["heft", "peft"])
def test_compiled_equals_reference(which):
    for r in _recs(60):
        inst = from_record(r)
        fa = Fast(inst)
        ref = (heft if which == "heft" else peft)(inst)
        plan, mk = (fa.heft() if which == "heft" else fa.peft())
        s_ref = ref.schedule(inst)
        s_new = plan.schedule(inst)
        assert s_new.makespan == pytest.approx(s_ref.makespan, rel=1e-12), r["id"]
        assert mk == pytest.approx(s_ref.makespan, rel=1e-12)
        assert list(plan.opt) == list(ref.opt)


def test_fixed_completion_equals_reference():
    rng = np.random.default_rng(1)
    for r in _recs(25):
        inst = from_record(r)
        fa = Fast(inst)
        fixed = {int(i): int(rng.choice(inst.task_opts[i])) for i in rng.choice(inst.n, size=inst.n // 3, replace=False)}
        ref = heft(inst, fixed=fixed)
        plan, mk = fa.heft(fixed)
        assert plan.schedule(inst).makespan == pytest.approx(ref.schedule(inst).makespan, rel=1e-12)
