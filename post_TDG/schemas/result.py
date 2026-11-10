"""Result schema: mapping, schedule, metrics and diagnostics produced by the partitioner."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from schemas.common import SCHEMA_VERSION, Ident, Interval, StrictModel


class Assignment(StrictModel):
    pe: Ident
    impl: Ident


class ScheduledTask(StrictModel):
    node: Ident
    pe: Ident
    impl: Ident
    start_ms: float
    finish_ms: float


class ScheduledComm(StrictModel):
    src: Ident
    dst: Ident
    interface: Ident | None = Field(None, description="None when both ends are on the same PE.")
    bytes: float
    start_ms: float
    finish_ms: float


class Violation(StrictModel):
    """A constraint that is not met. `constraint` names it, e.g. deadline_ms, deadline_ms[track],
    capacity[fpga0].DSP, period_ms, energy_mj_max."""

    constraint: str
    value: float
    limit: float
    excess: float = Field(description="value - limit, in the constraint's unit.")
    relative: float = Field(description="excess / limit.")


class ThroughputMetrics(StrictModel):
    """Streaming throughput from max-cycle-ratio analysis of the mapped graph (sim/throughput.py)."""

    period_ms: Interval = Field(description="Minimum sustainable frame period (Monte Carlo, order frozen).")
    period_bounds_ms: tuple[float, float] = Field(
        description="Period at the low/high corner of every cost interval: sound bounds for the frozen order."
    )
    load_bound_ms: float = Field(description="Per-resource busy-time lower bound (nominal); period >= this.")
    steady_latency_ms: Interval | None = Field(
        None, description="Frame latency in steady state at the input period (inf if input is too fast)."
    )
    input_period_ms: float | None = None
    buffers: int
    critical_cycle: list[str] = Field(description="Actors on the cycle that sets the nominal period.")


class Metrics(StrictModel):
    latency_ms: Interval
    energy_mj: Interval
    ii_ms: Interval = Field(description="Initiation interval (throughput bottleneck) estimate.")
    avg_power_mw: Interval
    usd: float
    resources: dict[str, dict[str, float]] = Field(description="Nominal use per PE per resource.")
    feasible_fraction: float = Field(ge=0, le=1, description="Fraction of Monte Carlo samples meeting all constraints.")
    throughput: ThroughputMetrics | None = Field(None, description="Present for streaming arrival.")
    seed: int


class LowerBound(StrictModel):
    """Optimistic bounds: each task on its fastest option, zero communication, unlimited parallelism."""

    latency_ms: float = Field(description="Critical path at nominal costs.")
    latency_ms_optimistic: float = Field(description="Critical path at the low end of every cost interval.")
    ii_ms: float
    energy_mj: float
    violations: list[Violation] = Field(description="Constraints the bound violates at nominal costs.")
    certain: list[str] = Field(
        default_factory=list, description="Constraints violated even at the low end of every cost interval."
    )
    structural: list[str] = Field(default_factory=list, description="Problems no mapping can fix, e.g. no compatible PE.")
    latency_ms_relaxation: float | None = Field(
        None, description="Makespan bound from the mapping relaxation (PE/bus load, transfers, capacity); set when "
        "the strong bound was requested. When it exceeds the deadline, no mapping can meet it.")


class Diagnostics(StrictModel):
    loop: Literal["none", "fb1", "fb2"] = Field(
        description="fb1: back to task generation (spec/graph problem); fb2: back to Bind/Partition."
    )
    reasons: list[str]
    lower_bound: LowerBound | None = None
    violations: list[Violation] = Field(default_factory=list, description="Violations of the returned mapping.")
    relaxation: list[Violation] | None = Field(
        None, description="Minimal relaxation that makes the ILP feasible (when it was infeasible)."
    )


class SolverInfo(StrictModel):
    name: str
    status: Literal["optimal", "feasible", "infeasible", "time_limit", "heuristic", "error"]
    runtime_s: float
    seed: int | None = None
    objective: float | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class Result(StrictModel):
    schema_version: Literal["0.1.0"] = SCHEMA_VERSION
    graph_id: Ident
    platform_id: Ident
    workload: dict[str, float]
    solver: SolverInfo
    mapping: dict[str, Assignment]
    schedule: list[ScheduledTask]
    comms: list[ScheduledComm]
    feasible: bool = Field(description="All constraints met at nominal costs.")
    metrics: Metrics | None = None
    diagnostics: Diagnostics


class Ranking(StrictModel):
    """Relative metric: candidates ordered by nominal objective, with order stability under uncertainty."""

    candidates: list[str] = Field(description="Labels, best first (nominal objective).")
    objective_nominal: list[float]
    pairwise_confidence: list[list[float]] = Field(
        description="[i][j] = fraction of samples where candidate i is strictly better than j (i before j)."
    )
    order_preserved_fraction: float = Field(description="Fraction of samples whose full ordering equals nominal.")
    mean_kendall_tau: float = Field(description="Mean Kendall tau-b between each sample's ordering and nominal.")
    n_samples: int
    seed: int
