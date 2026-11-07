"""Scores of one pipeline output against its benchmark task (implementation.md, Phase 7).

- spec: platform, arrival kind, timing hardness and deadline as expected; assumption sources
- TDG: valid; nodes matched to the reference by role (F1); dependencies between matched nodes
  (F1, see `_reduced_edges`); op counts and edge bytes within 2x of the reference; latency
  bounds against the reference's
"""

import re
from pathlib import Path

from hwsw.benchmarks.tasks import BenchmarkTask, ExpectedSpec
from hwsw.custom_hls import CostEstimationError, compute_latency_bounds, estimate_costs
from hwsw.platforms import BoardNotFoundError, load_board
from hwsw.validation import validate_tdg

DEADLINE_TOLERANCE = 0.05
AGREEMENT_FACTOR = 2.0


def score_spec(spec: dict, expected: ExpectedSpec) -> dict:
    deadline = spec["timing"].get("deadline_s")
    if expected.deadline_s is None or deadline is None:
        deadline_ok = expected.deadline_s is None and deadline is None
    else:
        deadline_ok = (
            abs(deadline - expected.deadline_s) <= DEADLINE_TOLERANCE * expected.deadline_s
        )
    sources = [assumption["source"] for assumption in spec["assumptions"]]
    return {
        "platform_ok": spec["platform"] == expected.platform,
        "arrival_ok": spec["arrival"]["kind"] == expected.arrival_kind,
        "hardness_ok": spec["timing"]["hardness"] == expected.hardness,
        "deadline_ok": deadline_ok,
        "assumptions": len(sources),
        "web_sources": sources.count("web"),
    }


def score_tdg(tdg: dict, reference: dict, task: BenchmarkTask, boards_dir: Path) -> dict:
    try:
        board = load_board(tdg["platform"], boards_dir)
    except BoardNotFoundError as unknown:
        return {"valid": False, "error": str(unknown)}
    if not validate_tdg(tdg, board).is_valid:
        return {"valid": False}

    matches = match_nodes(tdg, reference, task.roles)
    edge_matches = _match_edges(tdg, reference, matches)
    scores = {
        "valid": True,
        "nodes": len(tdg["nodes"]),
        "reference_nodes": len(reference["nodes"]),
        "node_f1": _f1(len(matches), len(tdg["nodes"]), len(reference["nodes"])),
        "edge_f1": _dependency_f1(tdg, reference, matches),
        "edge_bytes_within_2x": _fraction_agreeing(
            [(edge["bytes"], ref_edge["bytes"]) for edge, ref_edge in edge_matches]
        ),
        "matches": {ref_id: node_id for node_id, ref_id in matches.items()},
    }
    reference_board = load_board(reference["platform"], boards_dir)
    try:
        costed = estimate_costs(tdg, board)
    except CostEstimationError as failure:
        return {**scores, "error": f"cannot cost: {failure}"}
    reference_costed = estimate_costs(reference, reference_board)
    return {**scores, **_cost_scores(costed, reference_costed, matches)}


def match_nodes(tdg: dict, reference: dict, roles: dict[str, list[str]]) -> dict[str, str]:
    """generated node id -> reference node id, one to one.

    A pair is possible when a word of the generated node's name starts with one of the reference
    node's role words. Pairs with more such words win; ties go to nodes at similar positions.
    """
    candidates = []
    for position, node in enumerate(tdg["nodes"]):
        words = _words(node["name"])
        for ref_position, ref_node in enumerate(reference["nodes"]):
            hits = sum(
                any(word.startswith(role) for word in words) for role in roles[ref_node["id"]]
            )
            if hits:
                distance = abs(
                    position / len(tdg["nodes"]) - ref_position / len(reference["nodes"])
                )
                kind_bonus = 0.5 if node["kind"] == ref_node["kind"] else 0.0
                candidates.append((-(hits + kind_bonus), distance, node["id"], ref_node["id"]))
    matches: dict[str, str] = {}
    for _, _, node_id, ref_id in sorted(candidates):
        if node_id not in matches and ref_id not in matches.values():
            matches[node_id] = ref_id
    return matches


def _dependency_f1(tdg: dict, reference: dict, matches: dict[str, str]) -> float:
    generated = {(matches[src], matches[dst]) for src, dst in _reduced_edges(tdg, set(matches))}
    expected = _reduced_edges(reference, set(matches.values()))
    return _f1(len(generated & expected), len(generated), len(expected))


def _reduced_edges(tdg: dict, kept: set[str]) -> set[tuple[str, str]]:
    """(a, b) for kept nodes a, b where b is reached from a directly or only through nodes that
    are not kept. An extra or missing intermediate step (a buffer node, a split stage) then does
    not count as a wrong dependency; a missing or extra dependency still does."""
    successors: dict[str, list[str]] = {node["id"]: [] for node in tdg["nodes"]}
    for edge in tdg["edges"]:
        successors[edge["src"]].append(edge["dst"])
    reduced = set()
    for start in kept:
        pending, seen = list(successors[start]), set()
        while pending:
            node_id = pending.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            if node_id in kept:
                reduced.add((start, node_id))
            else:
                pending.extend(successors[node_id])
    return reduced


def _match_edges(tdg: dict, reference: dict, matches: dict[str, str]) -> list[tuple[dict, dict]]:
    reference_edges = {(edge["src"], edge["dst"]): edge for edge in reference["edges"]}
    pairs = []
    for edge in tdg["edges"]:
        key = (matches.get(edge["src"]), matches.get(edge["dst"]))
        if key in reference_edges:
            pairs.append((edge, reference_edges.pop(key)))
    return pairs


def _cost_scores(costed: dict, reference_costed: dict, matches: dict[str, str]) -> dict:
    ops = _ops_by_node(costed)
    reference_ops = _ops_by_node(reference_costed)
    bounds = compute_latency_bounds(costed)
    reference_bounds = compute_latency_bounds(reference_costed)
    return {
        "total_ops_ratio": _ratio(sum(ops.values()), sum(reference_ops.values())),
        "node_ops_within_2x": _fraction_agreeing(
            [(ops[node_id], reference_ops[ref_id]) for node_id, ref_id in matches.items()]
        ),
        "all_software_ratio": _ratio(bounds.all_software_s, reference_bounds.all_software_s),
        "best_case_ratio": _ratio(bounds.best_case_s, reference_bounds.best_case_s),
        "all_software_s": bounds.all_software_s,
        "best_case_s": bounds.best_case_s,
        "bounds_warnings": list(bounds.warnings),
    }


def _ops_by_node(costed: dict) -> dict[str, float]:
    return {cost["id"]: cost.get("ops_estimated") or 0 for cost in costed["costs"]["nodes"]}


def _words(name: str) -> list[str]:
    return [word for word in re.split(r"[^a-z0-9]+", name.lower()) if word]


def _f1(matched: int, generated: int, reference: int) -> float:
    if not matched:
        return 0.0
    precision, recall = matched / generated, matched / reference
    return round(2 * precision * recall / (precision + recall), 3)


def _fraction_agreeing(pairs: list[tuple[float, float]]) -> float | None:
    """Share of (value, reference) pairs within 2x of each other (both zero counts as agreeing)."""
    if not pairs:
        return None
    agreeing = sum(
        (a == 0 and b == 0)
        or (a > 0 and b > 0 and 1 / AGREEMENT_FACTOR <= a / b <= AGREEMENT_FACTOR)
        for a, b in pairs
    )
    return round(agreeing / len(pairs), 3)


def _ratio(value: float | None, reference: float | None) -> float | None:
    if value is None or not reference:
        return None
    return round(value / reference, 3)
