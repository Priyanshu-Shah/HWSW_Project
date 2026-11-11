import copy
import importlib.util
import json

import pytest

from fake_ollama import FakeOllama, call, text
from hwsw.benchmarks import (
    answer_question,
    list_task_ids,
    load_reference,
    load_task,
    match_nodes,
    ollama_failed,
    rescore,
    results_table,
    run_task,
    score_spec,
    score_tdg,
)
from hwsw.conversation import Question
from hwsw.custom_hls import estimate_costs
from hwsw.platforms import load_board
from hwsw.repo_paths import BENCHMARKS_DIR
from hwsw.validation import validate_tdg

TASK_IDS = list_task_ids()


def test_there_are_ten_tasks():
    assert len(TASK_IDS) == 10


def test_task_and_reference_files_are_up_to_date():
    spec = importlib.util.spec_from_file_location(
        "make_benchmarks", BENCHMARKS_DIR / "make_benchmarks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for make in module.BENCHMARKS:
        task_file, tdg = make()
        task = load_task(task_file["id"])
        assert json.loads(json.dumps(task_file)) == task.model_dump(), "run make_benchmarks.py"
        assert json.loads(json.dumps(tdg)) == load_reference(task), "run make_benchmarks.py"


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_reference_is_valid_and_costs(task_id, boards_dir):
    task = load_task(task_id)
    reference = load_reference(task)
    board = load_board(reference["platform"], boards_dir)
    assert validate_tdg(reference, board).is_valid
    assert validate_tdg(estimate_costs(reference, board), board).is_valid
    assert set(task.roles) == {node["id"] for node in reference["nodes"]}
    assert task.expected.platform == reference["platform"]
    assert task.expected.arrival_kind == reference["arrival"]["kind"]
    assert task.expected.deadline_s == reference.get("constraints", {}).get("deadline_s")


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_a_reference_scores_perfectly_against_itself(task_id, boards_dir):
    task = load_task(task_id)
    reference = load_reference(task)
    scores = score_tdg(reference, reference, task, boards_dir)
    assert scores["matches"] == {node["id"]: node["id"] for node in reference["nodes"]}
    assert scores["node_f1"] == scores["edge_f1"] == 1.0
    assert scores["edge_bytes_within_2x"] == scores["node_ops_within_2x"] == 1.0
    assert scores["total_ops_ratio"] == scores["best_case_ratio"] == 1.0


def test_a_different_graph_scores_lower(boards_dir):
    task = load_task("t02")
    reference = load_reference(task)
    graph = copy.deepcopy(reference)
    graph["nodes"][2]["name"] = "mystery_step"  # unmatched: costs one node
    graph["edges"][0]["bytes"] *= 3  # still matched, but too big
    scores = score_tdg(graph, reference, task, boards_dir)
    assert scores["node_f1"] == 0.8
    assert scores["edge_f1"] == 1.0  # n2 still reaches n4 through the unmatched node
    assert scores["edge_bytes_within_2x"] == 0.5


def test_extra_intermediate_nodes_keep_dependencies_but_missing_edges_cost(boards_dir):
    task = load_task("t02")
    reference = load_reference(task)
    with_buffer = copy.deepcopy(reference)
    with_buffer["nodes"].insert(1, {**reference["nodes"][4], "id": "nb", "name": "frame_buffer"})
    with_buffer["edges"][0] = {**reference["edges"][0], "dst": "nb"}
    with_buffer["edges"].append({**reference["edges"][0], "id": "eb", "src": "nb"})
    assert score_tdg(with_buffer, reference, task, boards_dir)["edge_f1"] == 1.0

    cut = copy.deepcopy(reference)
    cut["edges"] = [edge for edge in cut["edges"] if edge["id"] != "e2"]  # n2 -/-> n3
    assert score_tdg(cut, reference, task, boards_dir)["edge_f1"] == 0.857


def test_invalid_or_unknown_platform_graphs_score_as_invalid(boards_dir):
    task = load_task("t02")
    reference = load_reference(task)
    assert not score_tdg({**reference, "edges": []} | {"nodes": []}, reference, task, boards_dir)[
        "valid"
    ]
    assert (
        "unknown platform"
        in score_tdg({**reference, "platform": "zcu999"}, reference, task, boards_dir)["error"]
    )


def test_matching_prefers_more_role_words_then_position():
    reference = {"nodes": [{"id": "r1", "kind": "opaque"}, {"id": "r2", "kind": "loop_nest"}]}
    roles = {"r1": ["capture", "read"], "r2": ["sobel", "gradient", "read"]}
    graph = {
        "nodes": [
            {"id": "g1", "name": "read_camera", "kind": "opaque"},
            {"id": "g2", "name": "sobel_gradients", "kind": "loop_nest"},
            {"id": "g3", "name": "reader", "kind": "loop_nest"},
        ]
    }
    assert match_nodes(graph, reference, roles) == {"g2": "r2", "g1": "r1"}


def test_spec_scores(boards_dir):
    task = load_task("t09")
    spec = {
        "platform": "pynq_z2",
        "arrival": {"kind": "periodic", "period_s": 0.001},
        "timing": {"hardness": "hard", "deadline_s": 0.00102, "derived_from": "1 kHz"},
        "assumptions": [{"source": "web"}, {"source": "default"}],
    }
    scores = score_spec(spec, task.expected)
    assert all(scores[key] for key in ("platform_ok", "arrival_ok", "hardness_ok", "deadline_ok"))
    assert (scores["assumptions"], scores["web_sources"]) == (2, 1)
    soft = {**spec, "timing": {"hardness": "none", "derived_from": "?"}}
    assert not score_spec(soft, task.expected)["deadline_ok"]


@pytest.mark.parametrize(
    ("text", "choices", "expected"),
    [
        ("Which data type should the network use?", [], "int8"),
        ("Which board?", ["kv260", "pynq_z2", "zcu104"], "zcu104"),
        ("Is the timing hard or soft?", ["hard", "soft", "none"], "hard"),
        ("What hardware accelerator library?", [], "suggested"),  # not a timing question
        ("Anything else?", [], "suggested"),
        ("Which precision?", ["fp32", "fp16"], "int8"),  # fits no choice: typed in
        ("Which detection model?", ["YOLOv3", "SSD"], "YOLOv8n"),
    ],
)
def test_scripted_user(text, choices, expected):
    question = Question(id="q", text=text, suggested_answer="suggested", choices=choices)
    assert answer_question(load_task("t01"), question) == expected


def test_scripted_user_matches_the_question_id_first():
    question = Question(
        id="hardness", text="Must each 1 MiB chunk finish in time?", suggested_answer="soft"
    )
    assert answer_question(load_task("t06"), question) == "hard"


def rebuild_replies(spec: dict, reference: dict) -> list[dict]:
    """Scripted model replies that make the pipeline produce `reference` exactly."""
    fields = ("id", "name", "phase", "affinity", "kind")
    nodes = [{**{key: node[key] for key in fields}, "purpose": "-"} for node in reference["nodes"]]
    payloads = [
        {key: value for key, value in node.items() if key not in fields}
        for node in reference["nodes"]
    ]
    return [
        call("submit_spec", spec=spec),
        call("submit_notes", notes=[]),
        text(json.dumps({"graph_id": reference["graph"]["id"], "nodes": nodes})),
        text(json.dumps({"edges": reference["edges"]})),
        *(text(json.dumps(payload)) for payload in payloads),
    ]


def test_run_task_end_to_end_with_a_fake_model(boards_dir, tmp_path):
    task = load_task("t02")
    reference = load_reference(task)
    spec = {
        "task_summary": task.task,
        "platform": "kv260",
        "arrival": reference["arrival"],
        "timing": {"hardness": "hard", "deadline_s": 1 / 60, "derived_from": "60 fps (D3)"},
        "assumptions": [],
        "open_questions": [],
    }
    fake = FakeOllama(*rebuild_replies(spec, reference))
    result = run_task(task, boards_dir, client=fake.client(), runs_dir=tmp_path)
    assert result["error"] is None
    assert all(result["spec"][key] for key in ("platform_ok", "hardness_ok", "deadline_ok"))
    assert result["tdg"]["node_f1"] == 1.0 and result["tdg"]["total_ops_ratio"] == 1.0
    assert result["llm_calls"] == 4 + len(reference["nodes"])
    assert "| t02_sobel_edges |" in results_table([result])


def test_rescore_reads_the_saved_files(boards_dir, tmp_path):
    task = load_task("t02")
    reference = load_reference(task)
    (tmp_path / "tdg.json").write_text(json.dumps(reference))
    old = {"task": task.id, "run_dir": str(tmp_path), "spec": None, "tdg": {"edge_f1": 0.0}}
    (new,) = rescore([old], boards_dir)
    assert new["tdg"]["edge_f1"] == 1.0 and new["spec"] is None


def test_run_task_reports_a_failed_conversation(boards_dir, tmp_path):
    result = run_task(load_task("t02"), boards_dir, client=FakeOllama().client(), runs_dir=tmp_path)
    assert result["error"] and result["spec"] is None and result["tdg"] is None
    assert ollama_failed(result)
    assert "Valid graphs: 0 of 1" in results_table([result])
