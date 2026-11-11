"""HEFT/PEFT against the worked example of Topcuoglu, Hariri & Wu (IEEE TPDS 2002, Fig. 2/3):
10 tasks, 3 processors, HEFT schedule length 80. Communication cost c_ij is processor-independent and zero on
the same processor; we encode it as bytes on a non-shared link with 1 MB/s (1000 bytes = 1 time unit)."""

from __future__ import annotations

import pytest

from estimate import annotate
from schemas.ir import TaskGraph
from schemas.platform import Platform
from sim import evaluate
from sim.validate import validate_sim
from solvers.heuristics import heft, optimistic_cost_table, peft, upward_ranks

W = {  # computation cost on P1, P2, P3
    1: (14, 16, 9), 2: (13, 19, 18), 3: (11, 13, 19), 4: (13, 8, 17), 5: (12, 13, 10),
    6: (13, 16, 9), 7: (7, 15, 11), 8: (5, 11, 14), 9: (18, 12, 20), 10: (21, 7, 16),
}
C = {(1, 2): 18, (1, 3): 12, (1, 4): 9, (1, 5): 11, (1, 6): 14, (2, 8): 19, (2, 9): 16, (3, 7): 23,
     (4, 8): 27, (4, 9): 23, (5, 9): 13, (6, 8): 15, (7, 10): 17, (8, 10): 11, (9, 10): 13}


@pytest.fixture(scope="module")
def topcuoglu():
    platform = Platform.model_validate({
        "id": "topcuoglu3", "provenance": {"kind": "synthetic", "source": "Topcuoglu et al. 2002 example"},
        "pes": [{"id": f"P{i}", "type": f"p{i}", "kind": "cpu", "concurrency": "exclusive"} for i in (1, 2, 3)],
        "interfaces": [{"id": "net", "kind": "other", "connects": ["P1", "P2", "P3"], "setup_us": 0,
                        "bandwidth_mb_s": 1.0, "shared": False}],
    })
    graph = TaskGraph.model_validate({
        "id": "topcuoglu10", "provenance": {"kind": "synthetic", "source": "Topcuoglu et al. 2002 example"},
        "platform": {"id": "topcuoglu3"},
        "nodes": [{"id": f"n{t}", "implementations": [
            {"id": f"on{i}", "pe_type": f"p{i}", "cost": {"time_ms": W[t][i - 1]}} for i in (1, 2, 3)]}
            for t in W],
        "edges": [{"src": f"n{a}", "dst": f"n{b}", "bytes": c * 1000} for (a, b), c in C.items()],
    })
    return annotate(graph, platform)


def test_upward_ranks_match_paper(topcuoglu):
    r = upward_ranks(topcuoglu)
    expected = {1: 108.0, 2: 77.0, 3: 80.0, 4: 80.0, 5: 69.0, 6: 63.33, 7: 42.67, 8: 35.67, 9: 44.33, 10: 14.67}
    for t, v in expected.items():
        assert r[f"n{t}"] == pytest.approx(v, abs=0.01)


def test_heft_schedule_length_80(topcuoglu):
    plan = heft(topcuoglu)
    ev = evaluate(topcuoglu, plan.mapping, priority=plan.priority, insertion=plan.insertion)
    assert ev.sim.makespan == pytest.approx(80.0)
    assert plan.priority[:4] == ["n1", "n3", "n4", "n2"]
    assert validate_sim(topcuoglu, ev.sim, contention=False) == []


def test_insertion_never_hurts_heft_order(topcuoglu):
    plan = heft(topcuoglu)
    ins = evaluate(topcuoglu, plan.mapping, priority=plan.priority, insertion=True).sim.makespan
    no = evaluate(topcuoglu, plan.mapping, priority=plan.priority, insertion=False).sim.makespan
    assert ins <= no + 1e-9


def test_peft_valid_and_oct_is_zero_at_exit(topcuoglu):
    oct_ = optimistic_cost_table(topcuoglu)
    assert all(oct_[("n10", p)] == 0 for p in ("P1", "P2", "P3"))
    plan = peft(topcuoglu)
    ev = evaluate(topcuoglu, plan.mapping, priority=plan.priority, insertion=True)
    assert validate_sim(topcuoglu, ev.sim, contention=False) == []
    assert set(plan.mapping) == {f"n{t}" for t in W}


def test_heft_respects_fpga_capacity(toy):
    # B and C both want the FPGA (600 LUT each, capacity 1000): HEFT must not place both there.
    plan = heft(toy)
    on_fpga = [n for n, o in plan.mapping.items() if o.pe == "fpga0"]
    assert len(on_fpga) <= 1
    assert evaluate(toy, plan.mapping, priority=plan.priority, insertion=True).feasible
