"""Exact throughput (max cycle ratio) and steady-state latency, checked by hand.

Two-stage pipeline x (CPU, 2 ms) -> y (FPGA, 3 ms) on the toy platform; the transfer takes 1 ms on the shared bus.
Cycles: x self (2/1), y self (3/1), bus self (1/1), and the buffer cycle x->bus->y->x with weight 6 and B tokens.
"""

from __future__ import annotations

import pytest

from conftest import ROOT, TOY_B_HW, golden_variant, m
from estimate import annotate
from schemas import load_platform
from schemas.ir import TaskGraph
from sim.throughput import max_cycle_ratio, throughput

PIPE = m(x="cpu0/sw", y="fpga0/hw")


def pipeline(state_edge: bool = False):
    edges = [{"src": "x", "dst": "y", "bytes": 1000}]
    if state_edge:
        edges.append({"src": "y", "dst": "x", "bytes": 1000, "distance": 1})
    g = TaskGraph.model_validate({
        "id": "pipe", "provenance": {"kind": "synthetic", "source": "test"},
        "platform": {"id": "toy_cpu_fpga"},
        "arrival": {"kind": "streaming", "period_ms": 10},
        "nodes": [
            {"id": "x", "implementations": [{"id": "sw", "pe_type": "cpu.generic", "cost": {"time_ms": 2}}]},
            {"id": "y", "implementations": [{"id": "hw", "pe_type": "fpga.generic",
                                             "cost": {"time_ms": {"expr": 3, "unc": [0.5, 2.0]}}}]},
        ],
        "edges": edges,
    })
    return annotate(g, load_platform(ROOT / "platforms" / "toy_cpu_fpga.synthetic.json"))


def test_max_cycle_ratio_basic():
    lam, cyc = max_cycle_ratio(["a", "b"], [("a", "b", 2.0, 0), ("b", "a", 4.0, 2), ("a", "a", 1.0, 1)])
    assert lam == pytest.approx(3.0) and set(cyc) == {"a", "b"}
    assert max_cycle_ratio(["a", "b"], [("a", "b", 1.0, 0), ("b", "a", 1.0, 0)])[0] == float("inf")
    assert max_cycle_ratio(["a", "b"], [("a", "b", 1.0, 0)])[0] == 0.0


def test_double_buffering_reaches_the_bottleneck():
    r = throughput(pipeline(), PIPE, buffers=2)
    assert r.period_ms == pytest.approx(3.0)  # y self-loop dominates; buffer cycle 6/2 = 3
    assert r.load_bound_ms == pytest.approx(3.0)


def test_single_buffer_serialises_the_pipeline():
    r = throughput(pipeline(), PIPE, buffers=1)
    assert r.period_ms == pytest.approx(6.0)  # buffer cycle 2 + 1 + 3 with one token
    assert r.load_bound_ms == pytest.approx(3.0)  # the bottleneck bound misses it
    assert set(r.critical_cycle) == {"x", "x->y", "y"}


def test_cross_frame_state_creates_a_recurrence():
    r = throughput(pipeline(state_edge=True), PIPE, buffers=2)
    assert r.period_ms == pytest.approx(7.0)  # x -> bus -> y -> transfer back -> x: 2 + 1 + 3 + 1, one token


def test_steady_latency_and_overload():
    p = pipeline()
    assert throughput(p, PIPE, input_period_ms=3.0).steady_latency_ms == pytest.approx(6.0)
    assert throughput(p, PIPE, input_period_ms=2.0).steady_latency_ms == float("inf")


def test_corner_bounds_are_monotone():
    p = pipeline()
    lo = throughput(p, PIPE, costs=p.bound_view("lo")).period_ms
    hi = throughput(p, PIPE, costs=p.bound_view("hi")).period_ms
    nom = throughput(p, PIPE).period_ms
    assert lo <= nom <= hi
    # y ranges 1.5..6: low corner = buffer cycle (2 + 1 + 1.5) / 2 = 2.25; high corner = y self-loop 6
    assert (lo, hi) == pytest.approx((2.25, 6.0))


def test_period_never_below_load_bound_on_goldens(collision, wifi):
    from solvers.heuristics import heft

    for p in (collision, wifi):
        plan = heft(p)
        r = throughput(p, plan.mapping, priority=plan.priority, insertion=True)
        assert r.period_ms >= r.load_bound_ms - 1e-9


def test_toy_streaming_matches_cpu_cycle():
    p = golden_variant("toy_forkjoin", lambda d: d.update(arrival={"kind": "streaming", "period_ms": 8}))
    r = throughput(p, TOY_B_HW)
    assert r.period_ms == pytest.approx(6.0)  # cpu0 runs A, C, D = 2 + 3 + 1 per frame
    assert r.steady_latency_ms == pytest.approx(6.0)


def test_result_reports_throughput_for_streaming(collision):
    from solvers import heuristics

    assert collision.back_edges == [("track", "track", 1)]
    r = heuristics.run(collision, "heft", n_samples=100)
    tm = r.metrics.throughput
    assert tm is not None and tm.buffers == 2
    lo, hi = tm.period_bounds_ms
    assert lo <= tm.period_ms.lo and tm.period_ms.hi <= hi + 1e-9  # samples stay inside the sound corners
    assert tm.period_ms.nominal >= tm.load_bound_ms - 1e-9
    assert tm.steady_latency_ms is not None and tm.steady_latency_ms.nominal >= r.metrics.latency_ms.nominal - 1e-9


def test_exact_period_violation_when_bottleneck_passes():
    from solvers import heuristics
    from solvers.common import build_result
    from schemas.result import SolverInfo

    p = pipeline()
    p.buffers, p.period_ms = 1, 4.0  # single buffering: exact period 6 > 4 although the load bound is 3
    r = build_result(p, PIPE, SolverInfo(name="manual", status="heuristic", runtime_s=0.0), n_samples=20)
    assert [v.constraint for v in r.diagnostics.violations] == ["period_ms.exact"]
    assert not r.feasible and r.diagnostics.loop == "fb2"
    assert r.metrics.feasible_fraction < 1.0
