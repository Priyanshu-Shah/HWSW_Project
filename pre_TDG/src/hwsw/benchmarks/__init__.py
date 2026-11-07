"""Benchmarks (Phase 7): tasks with reference TDGs, a scripted user, scoring, and a runner."""

from hwsw.benchmarks.runner import ollama_failed, rescore, results_table, run_task
from hwsw.benchmarks.scoring import match_nodes, score_spec, score_tdg
from hwsw.benchmarks.scripted_user import answer_batch, answer_question
from hwsw.benchmarks.tasks import (
    BenchmarkTask,
    list_task_ids,
    load_reference,
    load_task,
)

__all__ = [
    "BenchmarkTask",
    "answer_batch",
    "answer_question",
    "list_task_ids",
    "load_reference",
    "load_task",
    "match_nodes",
    "ollama_failed",
    "rescore",
    "results_table",
    "run_task",
    "score_spec",
    "score_tdg",
]
