"""Validate a TDG (graph only, or graph + cost map): JSON Schema first, then semantic rules."""

from pathlib import Path

from hwsw.json_files import read_json
from hwsw.platforms import Board, BoardNotFoundError, load_board
from hwsw.repo_paths import LIBRARY_BOARDS_DIR
from hwsw.validation.constraint_rules import check_objective_and_deadline
from hwsw.validation.cost_map_rules import check_cost_map
from hwsw.validation.graph_rules import check_graph_structure
from hwsw.validation.issue import ValidationReport, error
from hwsw.validation.json_schema_rules import check_against_schema, load_tdg_schema
from hwsw.validation.loop_nest_rules import check_loop_nests
from hwsw.validation.platform_rules import check_platform


def validate_tdg(tdg: dict, board: Board | None = None) -> ValidationReport:
    """Check `tdg` against the schema and rules R1-R12.

    Without `board`, the checks that need board specs (R5) are skipped.
    """
    schema_issues = check_against_schema(tdg, load_tdg_schema())
    if schema_issues:
        return ValidationReport(schema_issues)  # the semantic rules assume a well-formed document

    issues = [
        *check_graph_structure(tdg),
        *check_loop_nests(tdg),
        *check_objective_and_deadline(tdg),
    ]
    if "costs" in tdg:
        issues += check_cost_map(tdg)
    if board is not None:
        issues += check_platform(tdg, board)
    return ValidationReport(issues)


def validate_tdg_file(path: Path, boards_dir: Path = LIBRARY_BOARDS_DIR) -> ValidationReport:
    """Like `validate_tdg`, loading the board named by the TDG's `platform` field."""
    tdg = read_json(path)
    board = None
    if isinstance(tdg, dict) and isinstance(tdg.get("platform"), str):
        try:
            board = load_board(tdg["platform"], boards_dir)
        except BoardNotFoundError as not_found:
            report = validate_tdg(tdg)
            report.issues.append(error("R5", "platform", str(not_found)))
            return report
    return validate_tdg(tdg, board)
