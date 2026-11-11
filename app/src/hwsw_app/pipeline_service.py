"""The pipeline operations the API offers. Thin: every step is a call into the `hwsw` package."""

import tempfile
from dataclasses import asdict
from pathlib import Path

from hwsw.custom_hls import (
    CostEstimationError,
    compute_latency_bounds,
    estimate_costs,
)
from hwsw.json_files import write_json
from hwsw.partition_runner import partition_costed_tdg, summary_lines
from hwsw.platforms import list_board_ids, load_board
from hwsw.validation import Issue, validate_tdg, validate_tdg_against_boards
from hwsw.validation.issue import error

from hwsw_app.schemas import (
    BoardOut,
    BoardsResponse,
    BoundsOut,
    CostResponse,
    IssueOut,
    PartitionResponse,
    ValidateResponse,
)
from hwsw_app.settings import AppSettings


class PipelineService:
    def __init__(self, settings: AppSettings):
        self._settings = settings

    def boards(self) -> BoardsResponse:
        boards_dir = self._settings.boards_dir
        boards = [load_board(board_id, boards_dir) for board_id in list_board_ids(boards_dir)]
        return BoardsResponse(
            source=self._settings.boards_source,
            boards=[BoardOut(id=b.id, name=b.name, summary=b.summary) for b in boards],
        )

    def board_ids(self) -> list[str]:
        return list_board_ids(self._settings.boards_dir)

    def validate(self, tdg: dict) -> ValidateResponse:
        report = validate_tdg_against_boards(tdg, self._settings.boards_dir)
        return ValidateResponse(valid=report.is_valid, issues=_issues_out(report.issues))

    def cost(self, tdg: dict) -> CostResponse:
        """Validate a graph-only TDG, add its cost map, validate again, and bound the latency."""
        before = validate_tdg_against_boards(tdg, self._settings.boards_dir)
        if not before.is_valid:
            return CostResponse(
                valid=False, costed_tdg=None, issues=_issues_out(before.issues), bounds=None
            )

        board = load_board(tdg["platform"], self._settings.boards_dir)
        try:
            costed = estimate_costs(tdg, board)
        except CostEstimationError as failure:
            problems = [error("COST", "costing", problem) for problem in failure.problems]
            return CostResponse(
                valid=False, costed_tdg=None, issues=_issues_out(problems), bounds=None
            )

        after = validate_tdg(costed, board)
        return CostResponse(
            valid=after.is_valid,
            costed_tdg=costed,
            issues=_issues_out(after.issues),
            bounds=self._bounds_or_none(costed),
        )

    def bounds(self, costed_tdg: dict) -> BoundsOut:
        """Raises KeyError or ValueError if `costed_tdg` has no usable cost map."""
        return BoundsOut(**asdict(compute_latency_bounds(costed_tdg)))

    def partition(self, costed_tdg: dict) -> PartitionResponse:
        """Hand a costed TDG to the post_TDG partitioner, through the JSON files as `hwsw` does."""
        if "costs" not in costed_tdg:
            return PartitionResponse(result=None, error="cost the graph first", summary=[])
        with tempfile.TemporaryDirectory() as folder:
            costed_file = Path(folder) / "tdg.costed.json"
            write_json(costed_file, costed_tdg)
            outcome = partition_costed_tdg(costed_file, boards_dir=self._settings.boards_dir)
        if outcome.result is None:
            return PartitionResponse(result=None, error=outcome.error, summary=[])
        return PartitionResponse(
            result=outcome.result, error=None, summary=summary_lines(outcome.result)
        )

    def _bounds_or_none(self, costed_tdg: dict) -> BoundsOut | None:
        try:
            return self.bounds(costed_tdg)
        except (KeyError, ValueError):
            return None


def _issues_out(issues: list[Issue]) -> list[IssueOut]:
    return [
        IssueOut(
            rule=issue.rule,
            severity=issue.severity.value,
            location=issue.location,
            message=issue.message,
        )
        for issue in issues
    ]
