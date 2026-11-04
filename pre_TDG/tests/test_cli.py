import pytest
from typer.testing import CliRunner

from conftest import FIXTURES_DIR
from hwsw.cli import app
from hwsw.json_files import read_json, write_json

runner = CliRunner()
SOBEL = FIXTURES_DIR / "tdgs" / "sobel_edge.tdg.json"


@pytest.fixture
def boards_option(boards_dir) -> list[str]:
    return ["--boards-dir", str(boards_dir)]


def test_cost_writes_a_valid_costed_tdg(tmp_path, boards_option):
    out = tmp_path / "sobel.costed.json"
    result = runner.invoke(app, ["cost", str(SOBEL), "--out", str(out), *boards_option])
    assert result.exit_code == 0, result.output
    assert "costs" in read_json(out)

    result = runner.invoke(app, ["validate", str(out), *boards_option])
    assert result.exit_code == 0, result.output
    assert "valid" in result.output


def test_cost_defaults_to_a_costed_file_next_to_the_input(tmp_path, boards_option):
    tdg_path = tmp_path / "sobel.tdg.json"
    write_json(tdg_path, read_json(SOBEL))
    assert runner.invoke(app, ["cost", str(tdg_path), *boards_option]).exit_code == 0
    assert (tmp_path / "sobel.tdg.costed.json").is_file()


def test_validate_reports_problems_and_fails(tmp_path, boards_option):
    broken = read_json(SOBEL)
    broken["edges"][0]["dst"] = "n9"
    path = tmp_path / "broken.tdg.json"
    write_json(path, broken)
    result = runner.invoke(app, ["validate", str(path), *boards_option])
    assert result.exit_code == 1
    assert "dst 'n9' does not exist" in result.output


def test_cost_refuses_an_invalid_tdg(tmp_path, boards_option):
    broken = read_json(SOBEL)
    broken["objective"] = {"energy": 1.0}
    path = tmp_path / "broken.tdg.json"
    write_json(path, broken)
    result = runner.invoke(app, ["cost", str(path), *boards_option])
    assert result.exit_code == 1
    assert not (tmp_path / "broken.tdg.costed.json").exists()


def test_unknown_platform_lists_the_available_boards(tmp_path, boards_option):
    other = read_json(SOBEL)
    other["platform"] = "no_such_board"
    path = tmp_path / "other.tdg.json"
    write_json(path, other)
    result = runner.invoke(app, ["cost", str(path), *boards_option])
    assert result.exit_code == 1
    assert "unknown platform 'no_such_board'" in result.output
    assert "zcu104" in result.output


def test_bounds_prints_both_bounds(tmp_path, boards_option):
    out = tmp_path / "sobel.costed.json"
    runner.invoke(app, ["cost", str(SOBEL), "--out", str(out), *boards_option])
    result = runner.invoke(app, ["bounds", str(out)])
    assert result.exit_code == 0, result.output
    assert "all software" in result.output
    assert "best case" in result.output
    assert "deadline" in result.output
