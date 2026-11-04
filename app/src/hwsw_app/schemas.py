"""Request and response bodies of the HTTP API."""

from pydantic import BaseModel


class IssueOut(BaseModel):
    rule: str
    severity: str  # "error" | "warning"
    location: str
    message: str


class BoundsOut(BaseModel):
    all_software_s: float | None
    best_case_s: float
    best_case_optimistic_s: float
    deadline_s: float | None
    warnings: list[str]


class TdgRequest(BaseModel):
    tdg: dict


class ValidateResponse(BaseModel):
    valid: bool
    issues: list[IssueOut]


class CostResponse(BaseModel):
    valid: bool
    costed_tdg: dict | None  # present whenever costing itself succeeded, even if issues remain
    issues: list[IssueOut]
    bounds: BoundsOut | None


class BoardOut(BaseModel):
    id: str
    name: str
    summary: str


class BoardsResponse(BaseModel):
    source: str  # where the board files came from; see settings.py
    boards: list[BoardOut]


class ExamplesResponse(BaseModel):
    names: list[str]


class SessionCreated(BaseModel):
    session_id: str
