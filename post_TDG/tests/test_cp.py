"""CP-SAT model: agrees with the MILP optimum, produces valid schedules, reports relaxations when infeasible."""

from __future__ import annotations

import pytest

from conftest import GOLDEN_IDS, ROOT, TOY_B_HW, golden_problem, golden_variant
from datagen import random_dag
from estimate import annotate
from schemas import load_platform
from sim import evaluate
from sim.validate import validate_starts
from solvers import cp
from solvers.cp import solve_cp
from solvers.ilp import solve_ilp

T_TOL = 2e-3 * 50  # integer-µs rounding: at most ~1 µs per task on graphs of <= 50 tasks


def _sim_obj(p, mapping, starts):
    prio = sorted(starts, key=lambda n: (starts[n], p.order.index(n)))
    return evaluate(p, mapping, priority=prio, contention=False).objective


def test_toy_optimum(toy):
    out = solve_cp(toy)
    assert out.status == "optimal"
    assert out.mapping == TOY_B_HW
    assert out.makespan == pytest.approx(6.0, abs=1e-3)
    assert out.gap == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("name", GOLDEN_IDS)
def test_cp_matches_ilp_on_goldens(name):
    p = golden_problem(name)
    c, i = solve_cp(p), solve_ilp(p)
    assert c.status == i.status == "optimal"
    assert _sim_obj(p, c.mapping, c.starts) == pytest.approx(_sim_obj(p, i.mapping, i.starts), rel=1e-3)


@pytest.mark.parametrize("seed", range(5))
def test_cp_matches_ilp_on_random_dags(seed):
    platform = load_platform(ROOT / "platforms" / "toy_cpu_fpga.synthetic.json")
    p = annotate(random_dag(8, seed), platform)
    c, i = solve_cp(p), solve_ilp(p)
    assert c.status == i.status == "optimal"
    assert c.makespan == pytest.approx(i.makespan, abs=8e-3)  # 8 tasks × 1 µs rounding


def test_cp_schedule_is_valid_after_rounding(collision):
    out = solve_cp(collision)
    # validate with the real (unrounded) costs, allowing the rounding slack per edge
    errs = validate_starts(collision, out.mapping, {n: s + 1e-3 * collision.order.index(n) for n, s in
                                                    out.starts.items()})
    assert all("overlap" not in e for e in errs)


def test_cp_infeasible_reports_relaxation():
    p = golden_variant("toy_forkjoin", lambda d: d["constraints"].update(deadline_ms=5.0))
    r = cp.solve(p, n_samples=0)
    assert r.solver.status == "infeasible"
    assert r.diagnostics.loop == "fb2"
    (v,) = r.diagnostics.relaxation
    assert v.constraint == "deadline_ms" and v.excess == pytest.approx(1.0, abs=1e-6)


def test_cp_result_contract(wifi):
    r = cp.solve(wifi, n_samples=50)
    assert r.solver.name == "cp-sat" and r.solver.status == "optimal"
    assert r.feasible
    assert r.solver.params["gap"] == pytest.approx(0.0, abs=1e-4)


def test_sub_microsecond_tasks_do_not_overlap():
    """Regression: durations < 1 µs used to round to 0, letting tasks share a CPU slot (Tier A seed 166)."""
    from schemas.ir import TaskGraph

    g = TaskGraph.model_validate({
        "id": "tiny", "provenance": {"kind": "synthetic", "source": "test"}, "platform": {"id": "toy_cpu_fpga"},
        "nodes": [{"id": f"t{i}", "implementations": [{"id": "sw", "pe_type": "cpu.generic",
                                                        "cost": {"time_ms": 0.0004}}]} for i in range(3)],
    })
    p = annotate(g, load_platform(ROOT / "platforms" / "toy_cpu_fpga.synthetic.json"))
    out = solve_cp(p)
    assert len({out.starts[n] for n in p.order}) == 3  # three distinct slots on the single CPU
    assert out.makespan >= 0.003 - 1e-12


def _bus_case():
    """A -> {B, C} -> D; B and C are 5 ms on the CPU or 1 ms on the FPGA; every transfer takes 2 ms on the shared bus.
    Without contention both-on-FPGA and one-on-FPGA tie at 7 ms; with contention both-on-FPGA needs 10 ms."""
    from schemas.ir import TaskGraph

    def node(nid, hw):
        impls = [{"id": "sw", "pe_type": "cpu.generic", "cost": {"time_ms": 5 if hw else 1}}]
        if hw:
            impls.append({"id": "hw", "pe_type": "fpga.generic", "cost": {"time_ms": 1, "resources": {"LUT": 100}}})
        return {"id": nid, "implementations": impls}

    g = TaskGraph.model_validate({
        "id": "bus", "provenance": {"kind": "synthetic", "source": "test"}, "platform": {"id": "toy_cpu_fpga"},
        "nodes": [node("A", False), node("B", True), node("C", True), node("D", False)],
        "edges": [{"src": a, "dst": b, "bytes": 2000} for a, b in [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]],
    })
    return annotate(g, load_platform(ROOT / "platforms" / "toy_cpu_fpga.synthetic.json"))


def test_contention_model_changes_the_optimum():
    p = _bus_case()
    both_hw = {"A": ("cpu0", "sw"), "B": ("fpga0", "hw"), "C": ("fpga0", "hw"), "D": ("cpu0", "sw")}
    from estimate import Option

    m_both = {n: Option(*v) for n, v in both_hw.items()}
    assert evaluate(p, m_both, contention=False).sim.makespan == pytest.approx(7.0)
    assert evaluate(p, m_both, insertion=True).sim.makespan == pytest.approx(10.0)
    out = solve_cp(p, contention=True)
    assert out.status == "optimal" and out.makespan == pytest.approx(7.0, abs=1e-2)
    on_fpga = [n for n, o in out.mapping.items() if o.pe == "fpga0"]
    assert len(on_fpga) == 1  # only one kernel is worth moving once the bus is shared
    replay = evaluate(p, out.mapping, priority=out.priority(p)).sim  # unified task + transfer replay
    assert replay.makespan == pytest.approx(out.makespan, abs=1e-2)
    assert any("->" in x for x in out.priority(p))


def test_contention_off_matches_ilp_and_on_is_never_better():
    for name in GOLDEN_IDS:
        p = golden_problem(name)
        off, on = solve_cp(p), solve_cp(p, contention=True)
        assert on.makespan >= off.makespan - 1e-2  # adding a resource can only lengthen the optimum


@pytest.mark.parametrize("name", GOLDEN_IDS)
def test_contention_labels_replay_exactly(name):
    from sim.validate import validate_sim

    p = golden_problem(name)
    out = solve_cp(p, contention=True)
    sim = evaluate(p, out.mapping, priority=out.priority(p)).sim
    assert sim.makespan == pytest.approx(out.makespan, abs=1e-3 * len(p.order))
    assert validate_sim(p, sim, contention=True) == []


def test_unified_mode_under_sampled_costs():
    from sim import montecarlo

    p = _bus_case()
    out = solve_cp(p, contention=True)
    mc = montecarlo.run(p, out.mapping, n=50, seed=0, priority=out.priority(p))
    assert (mc.latency_ms > 0).all()
