"""What the validator reports: issues, and the report that collects them."""

from dataclasses import dataclass, field
from enum import StrEnum


class Severity(StrEnum):
    ERROR = "error"  # the document must not be passed on
    WARNING = "warning"  # suspicious; flag it, but it can be passed on


@dataclass(frozen=True)
class Issue:
    rule: str  # "SCHEMA" or a rule id from the TDG schema header (R1..R12)
    severity: Severity
    location: str  # where in the document, e.g. "edge e4" or "nodes[2].loop_nest.loops"
    message: str

    def __str__(self) -> str:
        return f"{self.severity.value} [{self.rule}] {self.location}: {self.message}"


def error(rule: str, location: str, message: str) -> Issue:
    return Issue(rule, Severity.ERROR, location, message)


def warning(rule: str, location: str, message: str) -> Issue:
    return Issue(rule, Severity.WARNING, location, message)


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity is Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def format(self) -> str:
        """One issue per line. Written to be read by an LLM in a repair loop."""
        return "\n".join(str(issue) for issue in self.issues)
