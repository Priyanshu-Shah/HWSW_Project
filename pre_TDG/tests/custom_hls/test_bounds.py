import pytest

from hwsw.custom_hls import compute_latency_bounds


def estimate(value: float, lo_factor: float = 0.5) -> dict:
    return {"value": value, "lo": value * lo_factor, "hi": value * 2, "source": "roofline"}


def node(node_id: str, phase: str = "iteration", affinity: str = "any") -> dict:
    return {
        "id": node_id,
        "name": node_id,
        "phase": phase,
        "affinity": affinity,
        "kind": "opaque",
        "opaque": {"sw_time_s": 1, "source": "library_default"},
    }


def edge(src: str, dst: str) -> dict:
    return {"id": f"{src}_{dst}", "src": src, "dst": dst, "kind": "data", "bytes": 0}


def costed_tdg(nodes, edges, costs, deadline_s=None) -> dict:
    tdg = {"nodes": nodes, "edges": edges, "costs": {"nodes": costs, "edges": []}}
    if deadline_s is not None:
        tdg["constraints"] = {"deadline_s": deadline_s}
    return tdg


def sw_and_hw(node_id: str, sw_s: float, hw_s: float | None = None) -> dict:
    cost = {"id": node_id, "sw": {"time_s": estimate(sw_s)}}
    if hw_s is not None:
        cost["hw"] = [{"point": "p1", "time_s": estimate(hw_s), "resources": {"lut": 1}}]
    return cost


@pytest.fixture
def diamond() -> dict:
    """a -> (b | c) -> d: the path through the slower branch decides the latency."""
    return costed_tdg(
        [node("a"), node("b"), node("c"), node("d")],
        [edge("a", "b"), edge("a", "c"), edge("b", "d"), edge("c", "d")],
        [
            sw_and_hw("a", 1.0),
            sw_and_hw("b", 6.0, 2.0),
            sw_and_hw("c", 3.0, 1.0),
            sw_and_hw("d", 2.0),
        ],
    )


def test_all_software_follows_the_slowest_branch(diamond):
    assert compute_latency_bounds(diamond).all_software_s == pytest.approx(1 + 6 + 2)


def test_best_case_uses_each_nodes_fastest_option(diamond):
    bounds = compute_latency_bounds(diamond)
    assert bounds.best_case_s == pytest.approx(1 + 2 + 2)  # b on hardware (2), c on hardware (1)
    assert bounds.best_case_optimistic_s == pytest.approx((1 + 2 + 2) / 2)  # lo = value / 2


def test_setup_nodes_do_not_count():
    tdg = costed_tdg(
        [node("load", phase="setup"), node("run")],
        [edge("load", "run")],
        [sw_and_hw("load", 100.0), sw_and_hw("run", 1.0)],
    )
    assert compute_latency_bounds(tdg).all_software_s == pytest.approx(1.0)


def test_hw_only_node_means_no_all_software_bound():
    tdg = costed_tdg(
        [node("a", affinity="hw_only")], [], [{"id": "a", "hw": sw_and_hw("a", 1, 0.5)["hw"]}]
    )
    bounds = compute_latency_bounds(tdg)
    assert bounds.all_software_s is None
    assert bounds.best_case_s == pytest.approx(0.5)


def test_unreachable_deadline_is_flagged(diamond):
    diamond["constraints"] = {"deadline_s": 2.0}  # even the optimistic best case is 2.5
    (warning,) = compute_latency_bounds(diamond).warnings
    assert "no partition can meet the deadline" in warning


def test_reachable_deadline_is_not_flagged(diamond):
    diamond["constraints"] = {"deadline_s": 4.0}
    assert compute_latency_bounds(diamond).warnings == ()


def test_hardware_that_barely_helps_is_flagged():
    tdg = costed_tdg([node("a")], [], [sw_and_hw("a", 1.0, 0.99)])
    (warning,) = compute_latency_bounds(tdg).warnings
    assert "cannot improve latency" in warning


def test_node_without_any_cost_is_an_error():
    with pytest.raises(ValueError, match="node a has no software or hardware cost"):
        compute_latency_bounds(costed_tdg([node("a")], [], [{"id": "a"}]))
