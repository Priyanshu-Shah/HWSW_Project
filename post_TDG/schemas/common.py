"""Shared building blocks for the IR, platform and result schemas."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema, model_validator

SCHEMA_VERSION = "0.1.0"

ID_PATTERN = r"^[A-Za-z_][A-Za-z0-9_.\-]*$"
Ident = Annotated[str, Field(pattern=ID_PATTERN, min_length=1, max_length=128)]


class StrictModel(BaseModel):
    """Base model: unknown keys are errors, so typos in hand-written JSON fail loudly."""

    # Infinite values (e.g. an unsustainable stream's latency) serialise as the string "Infinity" and read back as
    # inf; pydantic's default would write null and break round-trips.
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")


class Provenance(StrictModel):
    """Where a number (or a whole file) comes from. `synthetic` must also appear in the file name."""

    kind: Literal["synthetic", "measured", "datasheet", "estimated", "derived", "mixed"]
    source: str = Field(description="Citation, URL or short description of the origin.")
    notes: str | None = None


_EXPR_JSON = {"anyOf": [{"type": "number"}, {"type": "string", "minLength": 1}]}


class Param(StrictModel):
    """A cost value: a number or an expression over workload parameters, with bounded uncertainty.

    `unc` = (lo, hi) multiplicative factors on the nominal value; the true value is assumed to lie in
    [lo * nominal, hi * nominal]. Monte Carlo samples the factor uniformly in that interval.
    """

    expr: Annotated[float | str, WithJsonSchema(_EXPR_JSON)]
    unc: tuple[float, float] = (1.0, 1.0)

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, data: Any) -> Any:
        if isinstance(data, (int, float, str)) and not isinstance(data, bool):
            return {"expr": data}
        return data

    @model_validator(mode="after")
    def _check_unc(self) -> "Param":
        lo, hi = self.unc
        if not (0 < lo <= 1.0 <= hi):
            raise ValueError(f"unc must satisfy 0 < lo <= 1 <= hi, got {self.unc}")
        return self


_PARAM_JSON = {
    "description": "Number, expression string over workload parameters, or {expr, unc}.",
    "anyOf": [
        {"type": "number"},
        {"type": "string", "minLength": 1},
        {
            "type": "object",
            "properties": {
                "expr": _EXPR_JSON,
                "unc": {
                    "type": "array",
                    "prefixItems": [{"type": "number"}, {"type": "number"}],
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": ["expr"],
            "additionalProperties": False,
        },
    ],
}

CostParam = Annotated[Param, WithJsonSchema(_PARAM_JSON)]
"""A Param that may be written in JSON as a bare number or expression string."""

NumOrExpr = Annotated[float | str, WithJsonSchema(_EXPR_JSON)]
"""A number or an expression over workload parameters, without uncertainty (used for limits)."""


class Interval(StrictModel):
    """A metric under cost uncertainty: nominal value plus Monte Carlo summary."""

    nominal: float
    lo: float = Field(description="Minimum over samples (envelope).")
    hi: float = Field(description="Maximum over samples (envelope).")
    p05: float
    p50: float
    p95: float
    n_samples: int = Field(ge=0)
