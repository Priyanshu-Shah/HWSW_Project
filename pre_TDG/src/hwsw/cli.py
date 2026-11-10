"""Command line: `hwsw validate | cost | bounds | clarify | build | run | bench | rescore`, and
`hwsw agents` (draws the agent graph)."""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.prompt import Prompt

from hwsw.benchmarks import (
    list_task_ids,
    load_task,
    ollama_failed,
    rescore,
    results_table,
    run_task,
)
from hwsw.conversation import (
    AgentError,
    AssistantMessage,
    AssumptionsNotice,
    ConversationAgent,
    QuestionBatch,
    UserInput,
)
from hwsw.custom_hls import (
    CostEstimationError,
    compute_latency_bounds,
    estimate_costs,
    load_constants,
)
from hwsw.json_files import read_json, write_json
from hwsw.llm import (
    ClarificationAgent,
    LlmError,
    LlmSettings,
    OllamaClient,
    PipelineAgent,
    RunLog,
    TdgBuilder,
    new_run_dir,
)
from hwsw.llm.pipeline_agent import agent_graph
from hwsw.llm.run_log import LOG_FILE_NAME
from hwsw.platforms import BoardNotFoundError, list_board_ids, load_board
from hwsw.repo_paths import BENCHMARKS_DIR, LIBRARY_BOARDS_DIR
from hwsw.validation import ValidationReport, validate_tdg, validate_tdg_file

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
console = Console(highlight=False)

TdgFile = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="TDG JSON file.")]
BoardsDir = Annotated[Path, typer.Option(help="Folder with the board files (<id>.json).")]


@app.command()
def validate(tdg_file: TdgFile, boards_dir: BoardsDir = LIBRARY_BOARDS_DIR) -> None:
    """Check a TDG (graph only, or graph + costs) against the schema and rules R1-R12."""
    report = validate_tdg_file(tdg_file, boards_dir)
    _print_report(report)
    if not report.is_valid:
        raise typer.Exit(code=1)
    console.print(f"[green]valid[/green] ({len(report.warnings)} warnings)")


@app.command()
def cost(
    tdg_file: TdgFile,
    out: Annotated[
        Path | None, typer.Option(help="Output file (default: <name>.costed.json).")
    ] = None,
    boards_dir: BoardsDir = LIBRARY_BOARDS_DIR,
    constants_file: Annotated[
        Path | None, typer.Option(exists=True, help="Cost-model constants.")
    ] = None,
) -> None:
    """Add the cost map to a graph-only TDG with custom_hls."""
    tdg = read_json(tdg_file)
    board = _load_board_or_exit(tdg.get("platform", ""), boards_dir)

    before = validate_tdg(tdg, board)
    _print_report(before)
    if not before.is_valid:
        raise typer.Exit(code=1)

    try:
        costed = estimate_costs(tdg, board, load_constants(constants_file))
    except CostEstimationError as failure:
        console.print(f"[red]cannot cost the TDG:[/red]\n{failure}")
        raise typer.Exit(code=1) from failure

    after = validate_tdg(costed, board)
    _print_report(after)
    if not after.is_valid:
        raise typer.Exit(code=1)

    out = out or tdg_file.with_name(tdg_file.name.removesuffix(".json") + ".costed.json")
    write_json(out, costed)
    console.print(f"[green]wrote[/green] {out}")


@app.command()
def bounds(costed_tdg_file: TdgFile) -> None:
    """Print the all-software and best-case latency of a costed TDG."""
    result = compute_latency_bounds(read_json(costed_tdg_file))
    all_software = "n/a" if result.all_software_s is None else _milliseconds(result.all_software_s)
    console.print(f"all software : {all_software}")
    console.print(
        f"best case    : {_milliseconds(result.best_case_s)} "
        f"(optimistic {_milliseconds(result.best_case_optimistic_s)})"
    )
    if result.deadline_s is not None:
        console.print(f"deadline     : {_milliseconds(result.deadline_s)}")
    for warning in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")


@app.command()
def clarify(
    task: Annotated[str, typer.Argument(help="The task in plain English.")],
    boards_dir: BoardsDir = LIBRARY_BOARDS_DIR,
) -> None:
    """Stage A: talk with the LLM until the task spec is complete; writes runs/<time>_spec/."""
    run_dir = new_run_dir("spec")
    client = OllamaClient(log=RunLog(run_dir / LOG_FILE_NAME))
    agent = ClarificationAgent(list_board_ids(boards_dir), client=client, boards_dir=boards_dir)
    console.print(f"model {client.settings.model}; log {run_dir / LOG_FILE_NAME}")
    _converse(agent, task, lambda: agent.spec is not None)
    write_json(run_dir / "spec.json", agent.spec)
    console.print(f"[green]wrote[/green] {run_dir / 'spec.json'}")


@app.command()
def build(
    spec_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="spec.json")],
    boards_dir: BoardsDir = LIBRARY_BOARDS_DIR,
) -> None:
    """Stage B: spec -> graph-only TDG, written as tdg.json next to the spec."""
    spec = read_json(spec_file)
    board = _load_board_or_exit(spec["platform"], boards_dir)
    client = OllamaClient(log=RunLog(spec_file.parent / LOG_FILE_NAME))
    try:
        with console.status("building the graph..."):
            result = TdgBuilder(client, board).build(spec)
    except LlmError as failure:
        console.print(f"[red]{failure}[/red]")
        raise typer.Exit(code=1) from failure
    for warning in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    out = spec_file.with_name("tdg.json")
    write_json(out, result.tdg)
    console.print(f"[green]wrote[/green] {out} ({len(result.tdg['nodes'])} nodes)")


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="The task in plain English.")],
    boards_dir: BoardsDir = LIBRARY_BOARDS_DIR,
) -> None:
    """The whole pipeline: questions -> spec -> TDG -> costs -> bounds, in runs/<time>_run/."""
    agent = PipelineAgent(
        list_board_ids(boards_dir), boards_dir=boards_dir, run_dir=new_run_dir("run")
    )
    console.print(f"run folder {agent.run_dir}")
    _converse(agent, task, lambda: agent.tdg is not None)
    console.print(f"[green]wrote[/green] {agent.run_dir / 'spec.json'}, tdg.json")
    tdg_file = agent.run_dir / "tdg.json"
    cost(tdg_file, out=None, boards_dir=boards_dir, constants_file=None)
    bounds(tdg_file.with_name("tdg.costed.json"))


@app.command()
def bench(
    tasks: Annotated[
        list[str] | None, typer.Argument(help="Task ids or prefixes (e.g. t02); default: all.")
    ] = None,
    boards_dir: BoardsDir = LIBRARY_BOARDS_DIR,
    model: Annotated[str | None, typer.Option(help="Ollama model (default: D2's).")] = None,
    think: Annotated[str, typer.Option(help="Reasoning effort: low | medium | high.")] = "medium",
    repeats: Annotated[int, typer.Option(min=1)] = 1,
) -> None:
    """Run benchmark tasks with a scripted user; writes benchmarks/results/<time>_<model>.md."""
    settings = LlmSettings.from_environment()
    settings = LlmSettings(host=settings.host, model=model or settings.model, think=think)
    task_list = [load_task(task_id) for task_id in tasks or list_task_ids()]
    name = f"{datetime.now():%Y%m%d_%H%M%S}_{settings.model.replace(':', '-')}_{think}"
    out = BENCHMARKS_DIR / "results" / name
    results = []
    for _ in range(repeats):
        for task in task_list:
            console.print(f"{task.id} ({settings.model}, think {think}) ...")
            result = run_task(task, boards_dir, settings)
            results.append(result)
            write_json(out.with_suffix(".json"), results)  # saved after every task
            out.with_suffix(".md").write_text(results_table(results), encoding="utf-8")
            console.print(f"  {result['minutes']} min; error: {result['error']}")
            if ollama_failed(result):
                console.print("[red]Ollama failed; stopping (see decisions.md D2).[/red]")
                raise typer.Exit(code=1)
    console.print(results_table(results))
    console.print(f"[green]wrote[/green] {out}.md and .json")


@app.command(name="rescore")
def rescore_command(
    results_file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="benchmarks/results/<run>.json")
    ],
    boards_dir: BoardsDir = LIBRARY_BOARDS_DIR,
) -> None:
    """Score a benchmark run again from its saved files (after a scoring change); no LLM."""
    results = rescore(read_json(results_file), boards_dir)
    write_json(results_file, results)
    results_file.with_suffix(".md").write_text(results_table(results), encoding="utf-8")
    console.print(results_table(results))


@app.command()
def agents(boards_dir: BoardsDir = LIBRARY_BOARDS_DIR) -> None:
    """Print the agent graph (D6) as a Mermaid diagram; paste it into https://mermaid.live."""
    graph = agent_graph(list_board_ids(boards_dir), OllamaClient(), boards_dir, tools=[])
    console.print(graph.compile().get_graph().draw_mermaid(), markup=False)


def _converse(agent: ConversationAgent, task: str, is_done: Callable[[], bool]) -> None:
    """Shows the agent's events as they come and asks its questions in the terminal."""
    user_input = UserInput(text=task)
    while not is_done():
        batch = None
        with console.status("thinking..."):
            for event in agent.handle(user_input):
                if isinstance(event, AssistantMessage):
                    console.print(event.text)
                elif isinstance(event, AssumptionsNotice):
                    for item in event.assumptions:
                        url = f" <{item.url}>" if item.url else ""
                        console.print(f"  {item.key} = {item.value} [{item.source}]{url}")
                elif isinstance(event, QuestionBatch):
                    batch = event
                elif isinstance(event, AgentError):
                    console.print(f"[red]{event.message}[/red]")
                    raise typer.Exit(code=1)
        user_input = UserInput(answers=_ask_in_terminal(batch)) if batch else UserInput()


def _ask_in_terminal(batch: QuestionBatch) -> dict[str, str]:
    return {
        question.id: Prompt.ask(
            question.text,
            console=console,
            default=question.suggested_answer,
            choices=question.choices or None,
        )
        for question in batch.questions
    }


def _load_board_or_exit(board_id: str, boards_dir: Path):
    try:
        return load_board(board_id, boards_dir)
    except BoardNotFoundError as not_found:
        console.print(f"[red]{not_found}[/red]")
        raise typer.Exit(code=1) from not_found


def _print_report(report: ValidationReport) -> None:
    for issue in report.errors:
        console.print(f"[red]{issue}[/red]", markup=True, highlight=False)
    for issue in report.warnings:
        console.print(f"[yellow]{issue}[/yellow]", markup=True, highlight=False)


def _milliseconds(seconds: float) -> str:
    return f"{seconds * 1e3:.4g} ms"
