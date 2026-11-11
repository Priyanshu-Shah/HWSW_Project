"""Contract schemas (IR, platform, result) shared with the task-generation team."""

from __future__ import annotations

import json
from pathlib import Path

from schemas.common import SCHEMA_VERSION, Interval, Param, Provenance
from schemas.ir import TaskGraph
from schemas.platform import Platform
from schemas.result import Ranking, Result

__all__ = [
    "SCHEMA_VERSION",
    "Interval",
    "Param",
    "Platform",
    "Provenance",
    "Ranking",
    "Result",
    "TaskGraph",
    "load_case",
    "load_graph",
    "load_platform",
]


def load_graph(path: str | Path) -> TaskGraph:
    return TaskGraph.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def load_platform(path: str | Path) -> Platform:
    return Platform.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def load_case(graph_path: str | Path) -> tuple[TaskGraph, Platform]:
    """Load a task graph and the platform it references (path relative to the graph file)."""
    graph_path = Path(graph_path)
    graph = load_graph(graph_path)
    if graph.platform.path is None:
        raise ValueError(f"{graph_path}: platform.path is not set")
    platform = load_platform(graph_path.parent / graph.platform.path)
    if platform.id != graph.platform.id:
        raise ValueError(f"platform id mismatch: graph wants {graph.platform.id!r}, file has {platform.id!r}")
    return graph, platform
