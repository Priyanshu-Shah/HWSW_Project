"""IR schema: hierarchical task graph with parameterised costs and constraints."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import Field, model_validator

from schemas.common import SCHEMA_VERSION, CostParam, Ident, NumOrExpr, Provenance, StrictModel


class Cost(StrictModel):
    """Cost of running one implementation of a node once (one graph iteration)."""

    time_ms: CostParam
    energy_mj: CostParam | None = Field(None, description="Dynamic energy per run. Overrides power_mw.")
    power_mw: CostParam | None = Field(None, description="Dynamic power while running; energy = power * time.")
    ii_ms: CostParam | None = Field(
        None, description="Initiation interval when pipelined on spatial hardware. Defaults to time_ms."
    )
    resources: dict[str, CostParam] = Field(
        default_factory=dict,
        description="Resource use on the PE, e.g. LUT, FF, DSP, BRAM18K, mem_bytes, usd.",
    )


class Implementation(StrictModel):
    """One way to run a node: a PE type plus its cost. Several per node = multiple-choice partitioning."""

    id: Ident
    pe_type: str = Field(description="Matched against Platform PE.type.")
    cost: Cost
    provenance: Provenance | None = None
    notes: str | None = None


class Edge(StrictModel):
    src: Ident
    dst: Ident
    bytes: CostParam = Field(description="Data moved per iteration; communication time comes from the platform.")
    name: str | None = None
    distance: int = Field(
        0, ge=0,
        description="Iteration distance: 0 = same frame (precedence); k > 0 = dst of frame i+k uses src of frame i "
        "(state carried across frames, e.g. a tracker). Distance > 0 edges may form cycles, including self-loops; "
        "they matter for streaming throughput, not for one iteration's schedule.",
    )


class Node(StrictModel):
    id: Ident
    name: str | None = None
    level: Literal["application", "task", "function", "loop", "basic_block"] = "task"
    implementations: list[Implementation] = Field(min_length=1)
    pinned_pe: Ident | None = Field(None, description="Fixed upstream decision: node must run on this PE.")
    deadline_ms: NumOrExpr | None = Field(None, description="Finish deadline relative to iteration start.")
    subgraph: SubGraph | None = Field(
        None, description="Refinement of this node. Phase 0 solvers treat the node as atomic."
    )
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_impls(self) -> "Node":
        ids = [i.id for i in self.implementations]
        if len(ids) != len(set(ids)):
            raise ValueError(f"node {self.id}: duplicate implementation ids {ids}")
        return self


class SubGraph(StrictModel):
    nodes: list[Node] = Field(min_length=1)
    edges: list[Edge] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_dag(self) -> "SubGraph":
        _check_edges(self.nodes, self.edges)
        return self


class Arrival(StrictModel):
    """How iterations arrive. periodic: each iteration must finish within the period.
    streaming: iterations are pipelined; only the initiation interval must fit in the period."""

    kind: Literal["one_shot", "periodic", "streaming"] = "one_shot"
    period_ms: NumOrExpr | None = None
    buffers: int = Field(2, ge=1, description="Streaming: frames of buffering per same-frame edge (2 = double buffering).")

    @model_validator(mode="after")
    def _period_required(self) -> "Arrival":
        if self.kind != "one_shot" and self.period_ms is None:
            raise ValueError(f"arrival kind {self.kind!r} needs period_ms")
        return self


class Constraints(StrictModel):
    deadline_ms: NumOrExpr | None = Field(None, description="End-to-end latency bound per iteration.")
    energy_mj_max: NumOrExpr | None = Field(None, description="Energy bound per iteration.")
    power_mw_max: NumOrExpr | None = Field(None, description="Average power bound.")
    usd_max: NumOrExpr | None = None


class Objective(StrictModel):
    """Weighted sum of metrics, each normalised by its lower bound before weighting."""

    weights: dict[Literal["latency", "energy", "usd"], float] = Field(default_factory=lambda: {"latency": 1.0})


class PlatformRef(StrictModel):
    id: Ident
    path: str | None = Field(None, description="Path to the platform JSON, relative to this file.")
    allowed_pes: list[Ident] | None = Field(None, description="Subset of PEs this case study may use.")


class AlternativeInfo(StrictModel):
    """Marks this graph as one of several whole-application alternatives from task generation."""

    group: str
    variant: str


class TaskGraph(StrictModel):
    schema_version: Literal["0.1.0"] = SCHEMA_VERSION
    id: Ident
    name: str | None = None
    description: str | None = None
    provenance: Provenance
    alternative: AlternativeInfo | None = None
    workload: dict[str, float] = Field(default_factory=dict, description="Parameters that costs are functions of.")
    arrival: Arrival = Field(default_factory=Arrival)
    constraints: Constraints = Field(default_factory=Constraints)
    objective: Objective = Field(default_factory=Objective)
    platform: PlatformRef
    nodes: list[Node] = Field(min_length=1)
    edges: list[Edge] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "TaskGraph":
        _check_edges(self.nodes, self.edges)
        return self

    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)


def _check_edges(nodes: list[Node], edges: list[Edge]) -> None:
    """Same-frame edges must form a DAG; cross-frame (distance > 0) edges only need known endpoints."""
    ids = [n.id for n in nodes]
    check_dag(ids, [(e.src, e.dst) for e in edges if e.distance == 0])
    known = set(ids)
    seen: set[tuple[str, str]] = set()
    for e in edges:
        if e.distance > 0:
            if e.src not in known or e.dst not in known:
                raise ValueError(f"edge {e.src}->{e.dst} references an unknown node")
            if (e.src, e.dst) in seen:
                raise ValueError(f"duplicate cross-frame edge {e.src}->{e.dst}")
            seen.add((e.src, e.dst))


def check_dag(node_ids: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """Validate ids and acyclicity; return a deterministic topological order (Kahn, ties by input order)."""
    if len(node_ids) != len(set(node_ids)):
        dup = sorted({n for n in node_ids if node_ids.count(n) > 1})
        raise ValueError(f"duplicate node ids: {dup}")
    known = set(node_ids)
    seen_edges: set[tuple[str, str]] = set()
    succ: dict[str, list[str]] = defaultdict(list)
    indeg = {n: 0 for n in node_ids}
    for s, d in edges:
        if s not in known or d not in known:
            raise ValueError(f"edge {s}->{d} references an unknown node")
        if s == d:
            raise ValueError(f"self-loop on {s}")
        if (s, d) in seen_edges:
            raise ValueError(f"duplicate edge {s}->{d}")
        seen_edges.add((s, d))
        succ[s].append(d)
        indeg[d] += 1
    order: list[str] = []
    ready = [n for n in node_ids if indeg[n] == 0]
    while ready:
        n = ready.pop(0)
        order.append(n)
        for d in succ[n]:
            indeg[d] -= 1
            if indeg[d] == 0:
                ready.append(d)
    if len(order) != len(node_ids):
        raise ValueError(f"graph has a cycle among {sorted(set(node_ids) - set(order))}")
    return order


Node.model_rebuild()
