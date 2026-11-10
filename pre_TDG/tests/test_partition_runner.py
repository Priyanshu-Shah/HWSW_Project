"""Hand-over to the post_TDG partitioner: a fake `partition` program, and the real one."""

import json
import stat

import pytest

from hwsw.custom_hls import estimate_costs
from hwsw.json_files import read_json, write_json
from hwsw.partition_runner import partition_costed_tdg, summary_lines
from hwsw.platforms import load_board
from hwsw.repo_paths import FIXTURE_BOARDS_DIR, FIXTURE_TDGS_DIR, PARTITION_COMMAND

RESULT = {
    "graph_id": "sobel_edge",
    "feasible": True,
    "deadline_s": 0.016667,
    "latency_s": {"nominal": 0.0061, "p05": 0.0053, "p95": 0.0069},
    "speedup": 1.5,
    "nodes": [
        {"id": "n2", "name": "sobel_gradient", "target": "hw", "point": "p3"},
        {"id": "n1", "name": "read_frame", "target": "sw", "point": None},
    ],
    "solver": {"method": "exact", "status": "optimal", "runtime_s": 0.05},
    "feedback": {"loop": "none", "reasons": ["best mapping meets all constraints"]},
}


@pytest.fixture
def costed_file(tmp_path):
    tdg = read_json(FIXTURE_TDGS_DIR / "sobel_edge.tdg.json")
    path = tmp_path / "tdg.costed.json"
    write_json(path, estimate_costs(tdg, load_board(tdg["platform"], FIXTURE_BOARDS_DIR)))
    return path


def fake_partitioner(tmp_path, exit_code: int, result: dict | None = RESULT):
    """A stand-in `partition` program: writes `result` to --out and exits with `exit_code`."""
    script = tmp_path / "partition"
    payload = json.dumps(result) if result is not None else ""
    script.write_text(
        "#!/bin/sh\n"
        'while [ "$1" != "--out" ]; do shift; done\n'
        + (
            f"cat > \"$2\" <<'EOF'\n{payload}\nEOF\n"
            if result is not None
            else "echo 'error: broken' >&2\n"
        )
        + f"exit {exit_code}\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_the_result_is_read_back(tmp_path, costed_file):
    outcome = partition_costed_tdg(
        costed_file, boards_dir=FIXTURE_BOARDS_DIR, command=fake_partitioner(tmp_path, 0)
    )
    assert outcome.error is None and outcome.result == RESULT
    assert (costed_file.parent / "partition.json").is_file()


def test_an_infeasible_answer_is_still_a_result(tmp_path, costed_file):
    infeasible = {**RESULT, "feasible": False}
    fake = fake_partitioner(tmp_path, 1, infeasible)
    outcome = partition_costed_tdg(costed_file, boards_dir=FIXTURE_BOARDS_DIR, command=fake)
    assert outcome.result["feasible"] is False


def test_a_crash_is_an_error(tmp_path, costed_file):
    outcome = partition_costed_tdg(
        costed_file, boards_dir=FIXTURE_BOARDS_DIR, command=fake_partitioner(tmp_path, 2, None)
    )
    assert outcome.result is None and "broken" in outcome.error


def test_missing_partitioner_or_board_is_an_error(tmp_path, costed_file):
    missing = partition_costed_tdg(
        costed_file, boards_dir=FIXTURE_BOARDS_DIR, command=tmp_path / "none"
    )
    assert missing.result is None and "not installed" in missing.error
    fake = fake_partitioner(tmp_path, 0)
    no_board = partition_costed_tdg(costed_file, boards_dir=tmp_path, command=fake)
    assert no_board.result is None and "board file" in no_board.error


def test_summary_names_the_hardware_nodes():
    lines = summary_lines(RESULT)
    assert "hardware: sobel_gradient (p3)" in lines
    assert any("1.50x faster" in line for line in lines)


@pytest.mark.skipif(not PARTITION_COMMAND.is_file(), reason="post_TDG partitioner not installed")
def test_the_real_partitioner(costed_file):
    outcome = partition_costed_tdg(costed_file, boards_dir=FIXTURE_BOARDS_DIR, time_limit_s=5)
    assert outcome.error is None
    assert outcome.result["solver"]["status"] == "optimal"
    tdg = read_json(costed_file)
    assert {n["id"] for n in outcome.result["nodes"]} == {n["id"] for n in tdg["nodes"]}
