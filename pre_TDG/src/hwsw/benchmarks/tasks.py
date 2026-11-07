"""Benchmark tasks (benchmarks/tasks/*.json) and their reference TDGs (benchmarks/reference/)."""

from pathlib import Path

from pydantic import BaseModel

from hwsw.json_files import read_json
from hwsw.repo_paths import BENCHMARKS_DIR


class ScriptedAnswer(BaseModel):
    keywords: list[str]  # the answer is given to a question containing any of these
    answer: str


class ExpectedSpec(BaseModel):
    platform: str
    arrival_kind: str
    hardness: str
    deadline_s: float | None


class BenchmarkTask(BaseModel):
    id: str
    title: str
    task: str  # what the user types
    answers: list[ScriptedAnswer]
    expected: ExpectedSpec
    roles: dict[str, list[str]]  # reference node id -> words a matching node's name may contain


def list_task_ids(benchmarks_dir: Path = BENCHMARKS_DIR) -> list[str]:
    return sorted(path.stem for path in (benchmarks_dir / "tasks").glob("*.json"))


def load_task(task_id: str, benchmarks_dir: Path = BENCHMARKS_DIR) -> BenchmarkTask:
    """`task_id` may be the full id or its prefix, e.g. "t02"."""
    matches = [found for found in list_task_ids(benchmarks_dir) if found.startswith(task_id)]
    if len(matches) != 1:
        known = ", ".join(list_task_ids(benchmarks_dir))
        raise KeyError(f"'{task_id}' matches {len(matches)} tasks; tasks: {known}")
    return BenchmarkTask.model_validate(read_json(benchmarks_dir / "tasks" / f"{matches[0]}.json"))


def load_reference(task: BenchmarkTask, benchmarks_dir: Path = BENCHMARKS_DIR) -> dict:
    return read_json(benchmarks_dir / "reference" / f"{task.id}.tdg.json")
