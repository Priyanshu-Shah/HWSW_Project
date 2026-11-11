"""Benchmark score of the last step: partition the generated graph and the reference graph with the
post_TDG partitioner, and compare. Skipped (None) when the partitioner is not installed.

- `latency_ratio`: best latency of the generated graph / of the reference graph (1 = as good).
- `same_side`: share of matched nodes (`score_tdg`'s matches) put on the same side (HW or SW).
"""

from pathlib import Path

from hwsw.custom_hls import CostEstimationError, estimate_costs
from hwsw.json_files import write_json
from hwsw.partition_runner import partition_costed_tdg
from hwsw.platforms import load_board
from hwsw.repo_paths import PARTITION_COMMAND


def score_partition(
    tdg: dict,
    reference: dict,
    matches: dict[str, str],  # reference node id -> generated node id
    boards_dir: Path,
    run_dir: Path,
    command: Path = PARTITION_COMMAND,
) -> dict | None:
    if not command.is_file():
        return None
    results = {}
    for name, graph in (("tdg", tdg), ("reference", reference)):
        try:
            costed = estimate_costs(graph, load_board(graph["platform"], boards_dir))
        except CostEstimationError as failure:
            return {"error": f"cannot cost the {name}: {failure}"}
        costed_file = run_dir / f"{name}.costed.json"
        write_json(costed_file, costed)
        outcome = partition_costed_tdg(
            costed_file,
            out=run_dir / f"{name}.partition.json",
            boards_dir=boards_dir,
            command=command,
        )
        if outcome.result is None:
            return {"error": f"{name}: {outcome.error}"}
        results[name] = outcome.result
    return _compare(results["tdg"], results["reference"], matches)


def _compare(result: dict, reference: dict, matches: dict[str, str]) -> dict:
    side = {node["id"]: node["target"] for node in result["nodes"]}
    reference_side = {node["id"]: node["target"] for node in reference["nodes"]}
    pairs = [
        (reference_side[ref_id], side[node_id])
        for ref_id, node_id in matches.items()
        if ref_id in reference_side and node_id in side
    ]
    latency = (result.get("latency_s") or {}).get("nominal")
    reference_latency = (reference.get("latency_s") or {}).get("nominal")
    return {
        "status": result["solver"]["status"],
        "feasible": result["feasible"],
        "latency_s": latency,
        "reference_latency_s": reference_latency,
        "latency_ratio": round(latency / reference_latency, 3)
        if latency and reference_latency
        else None,
        "speedup": result.get("speedup"),
        "hw_nodes": sum(target == "hw" for target in side.values()),
        "same_side": round(sum(a == b for a, b in pairs) / len(pairs), 3) if pairs else None,
    }
