from helpers import rules_of
from hwsw.custom_hls import estimate_costs
from hwsw.platforms import load_board
from hwsw.validation import validate_tdg


def test_contract_example_is_valid(costed_example, zcu104):
    report = validate_tdg(costed_example, zcu104)
    assert report.is_valid, report.format()
    assert not report.warnings


def test_graph_without_costs_is_valid(example_graph, zcu104):
    assert validate_tdg(example_graph, zcu104).is_valid


def test_fixture_graphs_are_valid(fixture_graph, boards_dir):
    board = load_board(fixture_graph["platform"], boards_dir)
    report = validate_tdg(fixture_graph, board)
    assert report.is_valid, report.format()


def test_costed_fixture_graphs_are_valid(fixture_graph, boards_dir):
    board = load_board(fixture_graph["platform"], boards_dir)
    report = validate_tdg(estimate_costs(fixture_graph, board), board)
    assert report.is_valid, report.format()


def test_board_is_optional(costed_example):
    assert validate_tdg(costed_example).is_valid


def test_schema_failure_stops_before_semantic_rules(example_graph):
    example_graph["edges"][0]["dst"] = "n9"  # would be R2...
    del example_graph["arrival"]["kind"]  # ...but the document is malformed first
    assert rules_of(validate_tdg(example_graph)) == {"SCHEMA"}
