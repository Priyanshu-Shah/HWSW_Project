"""Command line: `hwsw validate | cost | bounds` (clarify, build, run come with the LLM stages)."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from hwsw.custom_hls import (
    CostEstimationError,
    compute_latency_bounds,
    estimate_costs,
    load_constants,
)
from hwsw.json_files import read_json, write_json
from hwsw.platforms import BoardNotFoundError, load_board
from hwsw.repo_paths import LIBRARY_BOARDS_DIR
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
