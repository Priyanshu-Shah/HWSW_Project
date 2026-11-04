"""Validation of TDGs against ../TDG/schemas/tdg.schema.json and rules R1-R12."""

from hwsw.validation.issue import Issue, Severity, ValidationReport
from hwsw.validation.tdg_validator import validate_tdg, validate_tdg_file

__all__ = ["Issue", "Severity", "ValidationReport", "validate_tdg", "validate_tdg_file"]
