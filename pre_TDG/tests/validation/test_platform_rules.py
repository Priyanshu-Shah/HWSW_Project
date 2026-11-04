from helpers import find_node_cost
from hwsw.json_files import write_json
from hwsw.platforms import load_board
from hwsw.validation import validate_tdg, validate_tdg_file


def test_point_exceeding_board_capacity(costed_example, zcu104):
    find_node_cost(costed_example, "n2")["hw"][1]["resources"]["dsp"] = 5000
    (issue,) = validate_tdg(costed_example, zcu104).errors
    assert issue.rule == "R5"
    assert issue.location == "costs node n2 hw point p2"
    assert "needs 5000 dsp but board 'zcu104' has 1728" in issue.message


def test_capacity_is_not_checked_without_a_board(costed_example):
    find_node_cost(costed_example, "n2")["hw"][1]["resources"]["dsp"] = 5000
    assert validate_tdg(costed_example).is_valid


def test_tdg_for_a_different_board(costed_example, boards_dir):
    (issue,) = validate_tdg(costed_example, load_board("kv260", boards_dir)).errors
    assert issue.rule == "R5"
    assert "targets 'zcu104'" in issue.message


def test_edge_override_must_name_a_real_interface(costed_example, zcu104):
    costed_example["costs"]["edges"].append({"id": "e1", "interface": "axi_hp9"})
    (issue,) = validate_tdg(costed_example, zcu104).errors
    assert issue.rule == "R5"
    assert "available: axi_hp0" in issue.message


def test_file_validation_loads_the_board_named_in_the_tdg(costed_example, boards_dir, tmp_path):
    path = tmp_path / "tdg.json"
    write_json(path, costed_example)
    assert validate_tdg_file(path, boards_dir).is_valid


def test_file_validation_reports_unknown_platform(costed_example, tmp_path):
    costed_example["platform"] = "no_such_board"
    path = tmp_path / "tdg.json"
    write_json(path, costed_example)
    report = validate_tdg_file(path, tmp_path)
    assert [issue.rule for issue in report.errors] == ["R5"]
    assert "unknown platform 'no_such_board'" in report.errors[0].message
