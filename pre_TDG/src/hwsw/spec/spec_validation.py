"""Validate a task spec (stage A output / stage B input)."""

from functools import cache
from pathlib import Path

from hwsw.json_files import read_json
from hwsw.validation.issue import ValidationReport, error
from hwsw.validation.json_schema_rules import check_against_schema

SPEC_SCHEMA_PATH = Path(__file__).parent / "spec.schema.json"


@cache
def load_spec_schema() -> dict:
    return read_json(SPEC_SCHEMA_PATH)


def validate_spec(spec: dict, *, ready_for_tdg_builder: bool = False) -> ValidationReport:
    """Check `spec` against the schema. With `ready_for_tdg_builder`, unanswered questions fail."""
    issues = check_against_schema(spec, load_spec_schema())
    if ready_for_tdg_builder and not issues:
        issues = [
            error("SPEC", "open_questions", f"unanswered question: {question}")
            for question in spec["open_questions"]
        ]
    return ValidationReport(issues)
