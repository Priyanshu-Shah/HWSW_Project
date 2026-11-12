"""partition(): portfolio with CP-SAT (<= 40 nodes), GNN + GA path (> 40), the best-effort answer, and the CLI."""

import copy
import json

import pytest
from test_convert import TDGS, load, tdg_named

from post_tdg import partition as module
from post_tdg.cli import main
from post_tdg.partition import partition

pytest.importorskip("torch")


@pytest.mark.parametrize("path", TDGS, ids=lambda p: p.stem)
def test_every_graph_gets_a_checked_partition(path):
    tdg, board = load(path)
    result = partition(tdg, board, time_limit_s=10, samples=200).result
    iteration = {n["id"] for n in tdg["nodes"] if n["phase"] == "iteration"}
    assert {n["id"] for n in result["nodes"]} == iteration
    assert result["latency_s"]["nominal"] > 0
    for key, used in result["fpga_resources"]["used"].items():
        assert used <= result["fpga_resources"]["capacity"][key]
    if result["all_software_s"] is not None:
        assert result["latency_s"]["nominal"] <= result["all_software_s"] * (1 + 1e-9)
    if result["feasible"] and result["deadline_s"]:
        assert result["latency_s"]["nominal"] <= result["deadline_s"] * (1 + 1e-9)
    affinity = {n["id"]: n["affinity"] for n in tdg["nodes"]}
    for node in result["nodes"]:
        assert not (affinity[node["id"]] == "sw_only" and node["target"] == "hw")


def test_small_graphs_are_solved_to_a_proven_optimum():
    result = partition(*tdg_named("t10_video_stabilization"), samples=100).result
    assert result["solver"]["method"] == "exact"
    assert result["solver"]["status"] == "optimal"
    assert result["feasible"] and result["feedback"]["loop"] == "none"
    assert result["speedup"] > 1


def test_an_impossible_deadline_gives_fb1_and_the_fastest_partition():
    result = partition(*tdg_named("t01_object_detection"), samples=100).result
    assert result["feedback"]["loop"] == "fb1"
    assert not result["feasible"] and result["best_effort"]
    assert result["nodes"]
    assert result["latency_s"]["nominal"] > result["deadline_s"]
    assert result["feedback"]["min_latency_s"] > result["deadline_s"]


def test_without_the_gnn():
    result = partition(*tdg_named("t02_sobel_edges"), gnn_path=None, samples=100).result
    assert result["solver"]["status"] == "optimal"


def test_a_missing_gnn_model_is_noted_not_fatal(tmp_path):
    result = partition(*tdg_named("t02_sobel_edges"), gnn_path=tmp_path / "none.pt", samples=100).result
    assert "GNN not used" in result["solver"]["notes"]


def test_large_graphs_use_the_gnn_seeded_ga(monkeypatch):
    tdg, board = chained_copies(*tdg_named("t10_video_stabilization"), copies=6)  # 54 nodes
    assert len(tdg["nodes"]) > module.EXACT_MAX_TASKS
    result = partition(tdg, board, time_limit_s=3, samples=100).result
    assert result["solver"]["method"] == "ga+gnn"
    assert len(result["nodes"]) == 54
    assert result["solver"]["lower_bound_s"] <= result["latency_s"]["nominal"] * (1 + 1e-9)


def test_cli_writes_the_result_and_sets_the_exit_code(tmp_path, capsys):
    tdg, board = tdg_named("t02_sobel_edges")
    tdg_file, board_file = tmp_path / "sobel.costed.json", tmp_path / "board.json"
    tdg_file.write_text(json.dumps(tdg))
    board_file.write_text(json.dumps(board))
    assert main([str(tdg_file), "--board", str(board_file)]) == 0
    written = json.loads((tmp_path / "sobel.partition.json").read_text())
    assert written["graph_id"] == tdg["graph"]["id"]
    assert "speed-up" in capsys.readouterr().out

    impossible, impossible_board = tdg_named("t01_object_detection")
    tdg_file.write_text(json.dumps(impossible))
    board_file.write_text(json.dumps(impossible_board))
    assert main([str(tdg_file), "--board", str(board_file), "--out", str(tmp_path / "t01.json")]) == 1
    assert main([str(tmp_path / "missing.json"), "--board", str(board_file)]) == 2


def chained_copies(tdg: dict, board: dict, copies: int) -> tuple[dict, dict]:
    """`copies` renamed copies of a graph, each copy's last iteration node feeding the next copy's first."""
    out = copy.deepcopy(tdg)
    out["nodes"], out["edges"], out["costs"] = [], [], {"nodes": [], "edges": []}
    first, last = tdg["nodes"][0]["id"], tdg["nodes"][-1]["id"]
    for c in range(copies):
        rename = {n["id"]: f"c{c}_{n['id']}" for n in tdg["nodes"]}
        for node in tdg["nodes"]:
            out["nodes"].append({**node, "id": rename[node["id"]]})
        for cost in tdg["costs"]["nodes"]:
            out["costs"]["nodes"].append({**cost, "id": rename[cost["id"]]})
        for edge in tdg["edges"]:
            out["edges"].append(
                {**edge, "id": f"c{c}_{edge['id']}", "src": rename[edge["src"]], "dst": rename[edge["dst"]]}
            )
        if c:
            out["edges"].append(
                {"id": f"link{c}", "src": f"c{c - 1}_{last}", "dst": rename[first], "kind": "data", "bytes": 1024}
            )
    out["constraints"] = {}
    return out, board


def test_the_portfolio_records_its_steps_and_candidates():
    result = partition(*tdg_named("t10_video_stabilization"), samples=100)
    params = result.raw.solver.params
    assert any(step.startswith("CP-SAT") for step in params["steps"])
    assert {"peft", "exact"} <= set(params["candidates"])
    assert result.result["latency_s"]["nominal"] * 1e3 <= min(params["candidates"].values()) * (1 + 1e-9)


def test_a_tiny_budget_skips_the_gnn_and_cp_sat():
    result = partition(*tdg_named("t10_video_stabilization"), time_limit_s=0.02, samples=100)
    steps = result.raw.solver.params["steps"]
    assert steps and steps[0].startswith("SA from HEFT")
    assert not any(step.startswith("CP-SAT") for step in steps)
    assert result.result["nodes"]
