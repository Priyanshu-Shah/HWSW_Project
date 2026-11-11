"""Simulator vs the data package: label replay, HEFT on the paper example, HEFT vs the stored HEFT, golden optima."""
import json
from pathlib import Path

import numpy as np
import pytest

from fastheur.algos.common import mean_edge_comm, mean_task_time, rank_u
from fastheur.algos.heft import heft, oct_table, peft
from fastheur.fixtures import HEFT_PAPER_RANK_U, heft_paper_example
from fastheur.instance import from_golden, from_gcps, from_record
from fastheur.sim import simulate

DATA = Path(__file__).resolve().parents[2] / "data"


def _recs(path, k=None):
    out = []
    with open(DATA / path) as f:
        for line in f:
            out.append(json.loads(line))
            if k and len(out) >= k:
                break
    return out


def test_unified_replay_reproduces_every_optimal_label():
    n = 0
    for r in _recs("labels/tier_a_v0_bus.synthetic.jsonl"):
        L = r["label"]
        if L["status"] != "optimal":
            continue
        inst = from_record(r)
        idx = {t: i for i, t in enumerate(inst.tnames)}
        eid = {(int(inst.e_src[e]), int(inst.e_dst[e])): e for e in range(inst.E)}
        order = []
        for it in L["priority"]:
            if "->" in it:
                a, b = it.split("->")
                order.append(inst.n + eid[(idx[a], idx[b])])
            else:
                order.append(idx[it])
        s = simulate(inst, inst.mapping_from_names(L["mapping"]), order)
        ref = r["replay"]["makespan_with_contention_ms"]
        assert s.makespan == pytest.approx(ref, rel=1e-9), r["id"]
        assert s.capacity_ok
        n += 1
    assert n == 176


def test_heft_paper_example():
    inst = heft_paper_example()
    rk = rank_u(inst, mean_task_time(inst), mean_edge_comm(inst))
    for i, name in enumerate(inst.tnames):
        assert rk[i] == pytest.approx(HEFT_PAPER_RANK_U[int(name[1:])], abs=1e-3)
    assert heft(inst).schedule(inst).makespan == pytest.approx(80.0)


def test_peft_oct_hand_computed():
    # last tasks of the HEFT example: OCT(n10)=0; OCT(n9, p) = min_w (w(n10, pw) + 13 if pw != p)
    inst = heft_paper_example()
    o = oct_table(inst, mean_edge_comm(inst))
    t9 = inst.tnames.index("n9")
    vals = [o[t9][k] for k in inst.task_opts[t9]]
    assert vals == [20.0, 7.0, 16.0]
    assert all(v == 0.0 for v in o[inst.tnames.index("n10")].values())


def test_heft_agrees_with_stored_heft():
    ok = tot = 0
    for r in _recs("labels/tier_a_v0_bus.synthetic.jsonl"):
        inst = from_record(r)
        s = heft(inst).schedule(inst)
        ok += abs(s.makespan - r["heft"]["makespan_ms"]) < 1e-6 * r["heft"]["makespan_ms"]
        tot += 1
    assert tot == 200 and ok >= 190          # 192 / 200 exact; the rest differ by tie-breaking in the stored one


def test_golden_optima():
    from fastheur.fast import Fast
    from fastheur.algos import METHODS
    G = DATA.parent / "examples/golden"
    for name, opt in (("toy_forkjoin", 6.0), ("collision_detection", 27.023306666666667), ("wifi_tx", 7.467875)):
        inst = from_golden(G / f"{name}.synthetic.json")
        res = METHODS["magellan"](inst, Fast(inst), 0, 2.0)
        assert res.makespan == pytest.approx(opt, rel=1e-6)


def test_peft_known_weakness_on_collision_detection():
    # documented in format/model.md: PEFT puts optical flow on the FPGA and the CNN no longer fits (369 ms vs 27 ms)
    inst = from_golden(DATA.parent / "examples/golden/collision_detection.synthetic.json")
    assert peft(inst).schedule(inst).makespan == pytest.approx(369.5, abs=0.1)


def test_gcps_loader_shape():
    inst = from_gcps(DATA / "benchmarks/gcps_tgff/gcps_tgff_n20_CT2_s0_0.synthetic.json")
    assert inst.n == 20 and inst.m == 2 and inst.K == 40
    assert inst.pe_cap[1, 0] == pytest.approx(1983.950195384572)


def test_gcps_conversion_reproduces_reference_schedule_length():
    # the GCPS files store the schedule length (LSSP) of the greedy-HGP partition; our list scheduler must agree
    for n, tol in ((20, 1e-9), (100, 1e-9), (1000, 1e-3)):
        f = DATA / f"benchmarks/gcps_tgff/gcps_tgff_n{n}_CT2_s0_0.synthetic.json"
        d = json.loads(f.read_text())
        inst = from_gcps(f)
        hw = set(d["reference"]["greedy_hgp_hw_tasks"])
        opt = np.array([inst.task_opts[i][1 if int(inst.tnames[i][1:]) in hw else 0] for i in range(inst.n)])
        s = simulate(inst, opt)
        assert s.capacity_ok
        assert s.makespan * 1e6 == pytest.approx(d["reference"]["greedy_hgp_schedule_length"], rel=tol)
