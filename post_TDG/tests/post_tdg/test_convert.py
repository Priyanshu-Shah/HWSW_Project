"""Costed TDG + board -> IR + Platform (D7), on the costed benchmark graphs and the TDG example."""

import copy
import json
from pathlib import Path

import pytest

from estimate import annotate
from post_tdg.convert import FPGA_PE, ConversionError, convert, convert_board

FIXTURES = Path(__file__).parent / "fixtures"
TDGS = sorted((FIXTURES / "tdgs").glob("*.costed.json"))


def load(path: Path) -> tuple[dict, dict]:
    tdg = json.loads(path.read_text(encoding="utf-8"))
    board = json.loads((FIXTURES / "boards" / f"{tdg['platform']}.json").read_text(encoding="utf-8"))
    return tdg, board


def tdg_named(name: str) -> tuple[dict, dict]:
    return load(FIXTURES / "tdgs" / f"{name}.costed.json")


@pytest.mark.parametrize("path", TDGS, ids=lambda p: p.stem)
def test_every_costed_graph_converts_and_annotates(path):
    tdg, board = load(path)
    conversion = convert(tdg, board)
    problem = annotate(conversion.graph, conversion.platform)
    iteration = [n["id"] for n in tdg["nodes"] if n["phase"] == "iteration"]
    assert problem.order and sorted(problem.order) == sorted(iteration)
    assert all(problem.options[t] for t in problem.order)


def test_units_and_ranges():
    tdg, board = tdg_named("t02_sobel_edges")
    graph = convert(tdg, board).graph
    cost = next(c for c in tdg["costs"]["nodes"] if c["id"] == "n3")
    node = next(n for n in graph.nodes if n.id == "n3")
    sw = next(i for i in node.implementations if i.id == "sw")
    assert sw.cost.time_ms.expr == pytest.approx(cost["sw"]["time_s"]["value"] * 1e3)
    lo, hi = sw.cost.time_ms.unc
    assert lo * sw.cost.time_ms.expr == pytest.approx(cost["sw"]["time_s"]["lo"] * 1e3)
    assert hi * sw.cost.time_ms.expr == pytest.approx(cost["sw"]["time_s"]["hi"] * 1e3)
    points = [i for i in node.implementations if i.id != "sw"]
    assert [p.id for p in points] == [p["point"] for p in cost["hw"]]
    assert points[0].cost.resources["dsp"].expr == cost["hw"][0]["resources"]["dsp"]
    assert graph.constraints.deadline_ms == pytest.approx(tdg["constraints"]["deadline_s"] * 1e3)
    assert graph.arrival.kind == "one_shot"  # D3: one iteration, latency only


def test_setup_nodes_and_their_edges_are_dropped():
    tdg, board = tdg_named("t01_object_detection")
    conversion = convert(tdg, board)
    assert conversion.setup_nodes == ["n4"]
    assert "n4" not in {n.id for n in conversion.graph.nodes}
    assert all("n4" not in (e.src, e.dst) for e in conversion.graph.edges)


def test_affinity_keeps_only_allowed_implementations():
    tdg, board = tdg_named("t02_sobel_edges")
    graph = convert(tdg, board).graph
    affinity = {n["id"]: n["affinity"] for n in tdg["nodes"]}
    for node in graph.nodes:
        kinds = {"sw" if i.id == "sw" else "hw" for i in node.implementations}
        if affinity[node.id] == "sw_only":
            assert kinds == {"sw"}
        if affinity[node.id] == "hw_only":
            assert kinds == {"hw"}


def test_board_becomes_cores_fpga_and_interfaces():
    _, board = tdg_named("t02_sobel_edges")
    platform = convert_board(board)
    cpus = [pe for pe in platform.pes if pe.kind == "cpu"]
    assert len(cpus) == board["cpu"]["cores"]
    assert all(pe.concurrency == "exclusive" for pe in cpus)
    fpga = platform.pe(FPGA_PE)
    assert fpga.concurrency == "spatial" and fpga.capacity["lut"] == board["fpga"]["capacity"]["lut"]
    link = board["interfaces"][board["interfaces"]["default"]]
    axi = platform.interfaces_between("cpu0", FPGA_PE)[0]
    assert axi.transfer_ms(1e6) == pytest.approx((link["setup_s"] + 1e6 / link["bandwidth_Bps"]) * 1e3)
    assert {i.id for i in platform.interfaces_between("cpu0", "cpu1")} == {axi.id, "dram"}


def test_a_graph_without_costs_is_refused():
    tdg, board = tdg_named("t02_sobel_edges")
    del tdg["costs"]
    with pytest.raises(ConversionError, match="custom_hls"):
        convert(tdg, board)


def test_a_graph_for_another_board_is_refused():
    tdg, _ = tdg_named("t02_sobel_edges")
    _, other = tdg_named("t04_fir_audio")
    with pytest.raises(ConversionError, match="board"):
        convert(tdg, other)


def test_a_node_with_no_allowed_cost_is_refused():
    tdg, board = tdg_named("t02_sobel_edges")
    tdg = copy.deepcopy(tdg)
    node = next(n for n in tdg["nodes"] if n["id"] == "n3")
    node["affinity"] = "hw_only"
    next(c for c in tdg["costs"]["nodes"] if c["id"] == "n3")["hw"] = []
    with pytest.raises(ConversionError, match="n3"):
        convert(tdg, board)


@pytest.mark.parametrize("path", TDGS, ids=lambda p: p.stem)
def test_lower_bound_agrees_with_the_tdg_best_case(path):
    """The partitioner's bound adds transfers to pre_TDG's best case (fastest option, no transfers), so it is
    never below it; its all-software latency on the cores is never below the software critical path."""
    tdg, board = load(path)
    conversion = convert(tdg, board)
    problem = annotate(conversion.graph, conversion.platform)
    from diagnostics import lower_bound

    best_case_ms = _critical_path_ms(tdg, fastest=True)
    assert lower_bound(problem).latency_ms >= best_case_ms * (1 - 1e-6)


def _critical_path_ms(tdg: dict, fastest: bool) -> float:
    costs = {c["id"]: c for c in tdg["costs"]["nodes"]}
    nodes = [n["id"] for n in tdg["nodes"] if n["phase"] == "iteration"]

    def time_ms(node_id):
        cost = costs[node_id]
        options = [cost["sw"]["time_s"]["value"]] if cost.get("sw") else []
        if fastest:
            options += [p["time_s"]["value"] for p in cost.get("hw", [])]
        return min(options) * 1e3

    finish: dict[str, float] = {}
    remaining = list(nodes)
    while remaining:
        for node_id in list(remaining):
            preds = [e["src"] for e in tdg["edges"] if e["dst"] == node_id and e["src"] in nodes]
            if all(p in finish for p in preds):
                finish[node_id] = max((finish[p] for p in preds), default=0.0) + time_ms(node_id)
                remaining.remove(node_id)
    return max(finish.values())


def test_parallel_edges_are_merged():
    tdg, board = tdg_named("t02_sobel_edges")
    tdg = copy.deepcopy(tdg)
    first = tdg["edges"][0]
    tdg["edges"].append({**first, "id": "extra", "bytes": 100})
    tdg["edges"].append({**first, "id": "ctrl", "kind": "control", "bytes": 0})
    edges = convert(tdg, board).graph.edges
    pair = [e for e in edges if (e.src, e.dst) == (first["src"], first["dst"])]
    assert len(pair) == 1
    assert pair[0].bytes.expr == first["bytes"] + 100
