"""Run benchmark tasks through the whole LLM pipeline with a scripted user, and score them."""

import json
import time
from pathlib import Path

from hwsw.benchmarks.scoring import score_spec, score_tdg
from hwsw.benchmarks.scripted_user import answer_batch
from hwsw.benchmarks.tasks import BenchmarkTask, load_reference, load_task
from hwsw.conversation import AgentError, QuestionBatch, UserInput
from hwsw.json_files import read_json, write_json
from hwsw.llm import LlmSettings, OllamaClient, PipelineAgent, RunLog, new_run_dir
from hwsw.llm.run_log import LOG_FILE_NAME
from hwsw.platforms import list_board_ids
from hwsw.repo_paths import BENCHMARKS_DIR, RUNS_DIR

MAX_TURNS = 6  # question rounds are capped at 3 by the agent; this is a safety net
OLLAMA_ERROR_PREFIX = "Ollama at "  # how OllamaClient's errors start


def run_task(
    task: BenchmarkTask,
    boards_dir: Path,
    settings: LlmSettings | None = None,
    client: OllamaClient | None = None,  # tests pass a fake
    runs_dir: Path = RUNS_DIR,
    benchmarks_dir: Path = BENCHMARKS_DIR,
) -> dict:
    """One task end to end. The result is also written to the run folder as score.json."""
    run_dir = new_run_dir(f"bench_{task.id}", runs_dir)
    log = RunLog(run_dir / LOG_FILE_NAME)
    if client is None:
        client = OllamaClient(settings, log=log)
    else:
        client.log = log
    agent = PipelineAgent(
        list_board_ids(boards_dir), boards_dir=boards_dir, client=client, run_dir=run_dir
    )
    started = time.monotonic()
    conversation = _converse(agent, task)
    result = {
        "task": task.id,
        "model": client.settings.model,
        "run_dir": str(run_dir),
        "minutes": round((time.monotonic() - started) / 60, 2),
        **conversation,
        **_effort(run_dir / LOG_FILE_NAME),
        "spec": score_spec(agent.spec, task.expected) if agent.spec else None,
        "tdg": None,
    }
    if agent.tdg is not None:
        reference = load_reference(task, benchmarks_dir)
        result["tdg"] = score_tdg(agent.tdg, reference, task, boards_dir)
    write_json(run_dir / "score.json", result)
    return result


def rescore(
    results: list[dict], boards_dir: Path, benchmarks_dir: Path = BENCHMARKS_DIR
) -> list[dict]:
    """The same runs scored again from their saved spec.json and tdg.json (no LLM calls)."""
    rescored = []
    for result in results:
        task = load_task(result["task"], benchmarks_dir)
        run_dir = Path(result["run_dir"])
        spec, tdg = _read_optional(run_dir / "spec.json"), _read_optional(run_dir / "tdg.json")
        reference = load_reference(task, benchmarks_dir)
        rescored.append(
            {
                **result,
                "spec": score_spec(spec, task.expected) if spec else None,
                "tdg": score_tdg(tdg, reference, task, boards_dir) if tdg else None,
            }
        )
    return rescored


def _read_optional(path: Path) -> dict | None:
    return read_json(path) if path.is_file() else None


def ollama_failed(result: dict) -> bool:
    """Ollama itself failed (down, model not loadable): the next tasks would fail the same way."""
    return (result["error"] or "").startswith(OLLAMA_ERROR_PREFIX)


def _converse(agent: PipelineAgent, task: BenchmarkTask) -> dict:
    user_input = UserInput(text=task.task)
    rounds = questions = 0
    error = None
    for _ in range(MAX_TURNS):
        batch = None
        for event in agent.handle(user_input):
            if isinstance(event, QuestionBatch):
                batch = event
            elif isinstance(event, AgentError):
                error = event.message
        if agent.tdg is not None or error or batch is None:
            break
        rounds += 1
        questions += len(batch.questions)
        user_input = UserInput(answers=answer_batch(task, batch))
    if agent.tdg is None and not error:
        error = "the conversation ended without a task graph"
    return {"question_rounds": rounds, "questions": questions, "error": error}


def _effort(log_path: Path) -> dict:
    calls = tools = prompt_tokens = output_tokens = 0
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["kind"] == "llm_call":
                calls += 1
                prompt_tokens += record.get("prompt_tokens") or 0
                output_tokens += record.get("output_tokens") or 0
            elif record["kind"] == "tool_call":
                tools += 1
    return {
        "llm_calls": calls,
        "tool_calls": tools,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
    }


def results_table(results: list[dict]) -> str:
    """Markdown: one row per run, then the averages."""
    header = (
        "| task | model | spec ok | valid | node F1 | edge F1 | ops ratio | node ops 2x "
        "| bytes 2x | all-SW ratio | best ratio | rounds | questions | LLM calls | min | error |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = [_row(result) for result in results]
    return header + "\n".join(rows) + "\n\n" + _averages(results) + "\n"


def _row(result: dict) -> str:
    spec, tdg = result["spec"], result["tdg"] or {}
    spec_ok = "-" if spec is None else f"{sum(spec[key] for key in _SPEC_CHECKS)}/4"
    cells = [
        result["task"],
        result["model"],
        spec_ok,
        _cell(tdg.get("valid")),
        *(_cell(tdg.get(key)) for key in _TDG_COLUMNS),
        result["question_rounds"],
        result["questions"],
        result["llm_calls"],
        result["minutes"],
        (result["error"] or tdg.get("error") or "")[:80].replace("|", "/"),
    ]
    return "| " + " | ".join(map(str, cells)) + " |"


def _averages(results: list[dict]) -> str:
    graphs = [result["tdg"] for result in results if result["tdg"] and result["tdg"]["valid"]]
    lines = [f"Valid graphs: {len(graphs)} of {len(results)}."]
    for key in ("node_f1", "edge_f1", "node_ops_within_2x", "edge_bytes_within_2x"):
        values = [graph[key] for graph in graphs if graph.get(key) is not None]
        if values:
            lines.append(f"Mean {key}: {sum(values) / len(values):.2f}.")
    ratios = [graph["total_ops_ratio"] for graph in graphs if graph.get("total_ops_ratio")]
    within = sum(1 / 2 <= ratio <= 2 for ratio in ratios)
    lines.append(f"Total ops within 2x of the reference: {within} of {len(ratios)}.")
    rounds = [result["question_rounds"] for result in results]
    lines.append(f"Mean question rounds: {sum(rounds) / max(len(rounds), 1):.1f}.")
    return " ".join(lines)


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


_SPEC_CHECKS = ("platform_ok", "arrival_ok", "hardness_ok", "deadline_ok")
_TDG_COLUMNS = (
    "node_f1",
    "edge_f1",
    "total_ops_ratio",
    "node_ops_within_2x",
    "edge_bytes_within_2x",
    "all_software_ratio",
    "best_case_ratio",
)
