"""Hands a costed TDG to the post_TDG partitioner and reads its answer back (D7).

The partitioner is the post_TDG team's program, in its own environment: we run its `partition`
command on the JSON files (the contract) and never import its code.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from hwsw.json_files import read_json
from hwsw.repo_paths import LIBRARY_BOARDS_DIR, PARTITION_COMMAND

# The partitioner's own budget (it stops on time) plus model loading and the Monte Carlo metrics.
DEFAULT_TIME_LIMIT_S = 10.0
OVERHEAD_S = 120.0


@dataclass(frozen=True)
class PartitionOutcome:
    result: dict | None  # post_TDG's partition.result.json; None when it could not run
    error: str | None = None


def partition_costed_tdg(
    costed_tdg_file: Path,
    *,
    out: Path | None = None,
    boards_dir: Path = LIBRARY_BOARDS_DIR,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    command: Path = PARTITION_COMMAND,
) -> PartitionOutcome:
    out = out or costed_tdg_file.with_name("partition.json")
    board_id = read_json(costed_tdg_file).get("platform", "")
    board_file = boards_dir / f"{board_id}.json"
    if not board_file.is_file():
        return PartitionOutcome(None, f"board file {board_file} not found")
    if not command.is_file():
        return PartitionOutcome(
            None, f"the partitioner is not installed ({command}); see post_TDG/README.md"
        )
    out.unlink(missing_ok=True)  # never read back a result from an earlier run
    arguments = [
        str(command),
        str(costed_tdg_file),
        "--board",
        str(board_file),
        "--out",
        str(out),
        "--time-limit",
        str(time_limit_s),
    ]
    try:
        timeout = time_limit_s + OVERHEAD_S
        done = subprocess.run(
            arguments, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return PartitionOutcome(None, f"the partitioner did not finish in {timeout:.0f} s")
    # Exit 0: constraints met; 1: answered, but no partition meets them (the result says why).
    if done.returncode not in (0, 1) or not out.is_file():
        detail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["no output"]
        failure = f"the partitioner failed (exit {done.returncode}): {detail[0]}"
        return PartitionOutcome(None, failure)
    return PartitionOutcome(read_json(out))


def summary_lines(result: dict) -> list[str]:
    """Plain-text summary of a partition result, for the terminal and the chat."""
    lines = []
    latency = result.get("latency_s")
    if latency:
        ms = {key: latency[key] * 1e3 for key in ("nominal", "p05", "p95")}
        line = f"latency {ms['nominal']:.3g} ms (p05 {ms['p05']:.3g}, p95 {ms['p95']:.3g})"
        if result.get("speedup"):
            line += f", {result['speedup']:.2f}x faster than all-software"
        lines.append(line)
    deadline = result.get("deadline_s")
    if deadline:
        verdict = "met" if result["feasible"] else "NOT met"
        lines.append(f"deadline {deadline * 1e3:.3g} ms: {verdict}")
    hardware = [f"{n['name']} ({n['point']})" for n in result["nodes"] if n["target"] == "hw"]
    lines.append("hardware: " + (", ".join(hardware) if hardware else "none (all software)"))
    solver = result["solver"]
    lines.append(f"solver: {solver['method']}, {solver['status']}, {solver['runtime_s']:.2f} s")
    lines.extend(result["feedback"]["reasons"] if result["feedback"]["loop"] != "none" else [])
    return lines
