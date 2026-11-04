"""Check a document against a JSON Schema and explain each failure in one readable line."""

from collections.abc import Iterator
from functools import cache
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError, best_match

from hwsw.json_files import read_json
from hwsw.repo_paths import TDG_SCHEMA_PATH
from hwsw.validation.issue import Issue, error

SCHEMA_RULE = "SCHEMA"


@cache
def load_tdg_schema() -> dict:
    return read_json(TDG_SCHEMA_PATH)


def check_against_schema(document: Any, schema: dict) -> list[Issue]:
    validator = Draft202012Validator(schema)
    failures = sorted(validator.iter_errors(document), key=lambda failure: _location(failure))
    return [
        error(SCHEMA_RULE, _location(specific) or "document", specific.message)
        for specific in map(_most_specific, failures)
    ]


def _location(failure: ValidationError) -> str:
    """`nodes[1].loop_nest.arrays.img.dtype` for a failure at that place in the document."""
    location = ""
    for step in failure.absolute_path:
        location += f"[{step}]" if isinstance(step, int) else f".{step}"
    return location.lstrip(".")


def _most_specific(failure: ValidationError) -> ValidationError:
    """For `oneOf` failures, report the branch the document was *trying* to match.

    A node with kind "loop_nest" fails the "opaque" and "template" branches only because it is
    not one of them; those failures are noise. Branches that fail on a `const` (the `kind` or
    `op` discriminator) are dropped, and the best remaining failure is reported.
    """
    if not failure.context:
        return failure
    remaining = [
        branch_failures
        for branch_failures in _group_by_branch(failure.context)
        if not any(f.validator == "const" for f in branch_failures)
    ]
    candidates = [f for branch_failures in remaining for f in branch_failures] or failure.context
    return _most_specific(best_match(candidates))


def _group_by_branch(context: list[ValidationError]) -> Iterator[list[ValidationError]]:
    branches: dict[Any, list[ValidationError]] = {}
    for failure in context:
        branches.setdefault(failure.relative_schema_path[0], []).append(failure)
    return iter(branches.values())
