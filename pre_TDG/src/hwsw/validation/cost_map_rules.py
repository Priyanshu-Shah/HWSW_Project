"""Rules for the `costs` part: references (R2), ranges (R4), required costs (R6, R8), ops (R9)."""

from collections import Counter

from hwsw.validation.issue import Issue, error, warning

# D1: ops_claimed (LLM) and ops_estimated (custom_hls) must agree within this factor.
OPS_AGREEMENT_FACTOR = 2.0


def check_cost_map(tdg: dict) -> list[Issue]:
    nodes_by_id = {node["id"]: node for node in tdg["nodes"]}
    node_costs = {cost["id"]: cost for cost in tdg["costs"]["nodes"]}
    return [
        *_duplicate_cost_entries(tdg["costs"]["nodes"]),
        *_costs_reference_existing_items(tdg, nodes_by_id),
        *_estimates_are_ordered(tdg["costs"]),
        *_nodes_have_the_costs_they_need(nodes_by_id, node_costs),
        *_ops_agree(nodes_by_id, node_costs),
    ]


def _duplicate_cost_entries(node_costs: list[dict]) -> list[Issue]:
    ids = Counter(cost["id"] for cost in node_costs)
    issues = [
        error("R1", f"costs node {node_id}", f"node {node_id} has {count} cost entries; expected 1")
        for node_id, count in ids.items()
        if count > 1
    ]
    for cost in node_costs:
        point_ids = Counter(point["point"] for point in cost.get("hw", []))
        issues += [
            error(
                "R1",
                f"costs node {cost['id']}",
                f"hardware point '{point}' is listed {count} times",
            )
            for point, count in point_ids.items()
            if count > 1
        ]
    return issues


def _costs_reference_existing_items(tdg: dict, nodes_by_id: dict[str, dict]) -> list[Issue]:
    edge_ids = {edge["id"] for edge in tdg["edges"]}
    issues = [
        error(
            "R2",
            f"costs node {cost['id']}",
            f"cost entry for node '{cost['id']}', which does not exist",
        )
        for cost in tdg["costs"]["nodes"]
        if cost["id"] not in nodes_by_id
    ]
    issues += [
        error(
            "R2",
            f"costs edge {cost['id']}",
            f"cost entry for edge '{cost['id']}', which does not exist",
        )
        for cost in tdg["costs"].get("edges", [])
        if cost["id"] not in edge_ids
    ]
    return issues


def _estimates_are_ordered(costs: dict) -> list[Issue]:
    """R4: every estimate must satisfy lo <= value <= hi."""
    issues = []
    for cost in costs["nodes"]:
        where = f"costs node {cost['id']}"
        if "sw" in cost:
            issues += _check_range(f"{where} sw time_s", cost["sw"]["time_s"])
        for point in cost.get("hw", []):
            issues += _check_range(f"{where} hw point {point['point']} time_s", point["time_s"])
    for cost in costs.get("edges", []):
        if "time_s" in cost:
            issues += _check_range(f"costs edge {cost['id']} time_s", cost["time_s"])
    return issues


def _check_range(location: str, estimate: dict) -> list[Issue]:
    if estimate["lo"] <= estimate["value"] <= estimate["hi"]:
        return []
    return [
        error(
            "R4",
            location,
            f"expected lo <= value <= hi, got lo={estimate['lo']}, value={estimate['value']}, "
            f"hi={estimate['hi']}",
        )
    ]


def _nodes_have_the_costs_they_need(
    nodes_by_id: dict[str, dict], node_costs: dict[str, dict]
) -> list[Issue]:
    issues = []
    for node_id, node in nodes_by_id.items():
        cost = node_costs.get(node_id, {})
        if node["affinity"] == "hw_only" and not cost.get("hw"):
            issues.append(
                error(
                    "R6",
                    f"node {node_id}",
                    "affinity is hw_only but it has no hardware design point",
                )
            )
        if node["phase"] == "iteration" and node["affinity"] != "hw_only" and "sw" not in cost:
            issues.append(
                error("R8", f"node {node_id}", "iteration node has no software cost (costs.sw)")
            )
    return issues


def _ops_agree(nodes_by_id: dict[str, dict], node_costs: dict[str, dict]) -> list[Issue]:
    issues = []
    for node_id, node in nodes_by_id.items():
        claimed = node.get("ops_claimed")
        estimated = node_costs.get(node_id, {}).get("ops_estimated")
        if claimed is None or estimated is None:
            continue
        if not _within_factor(claimed, estimated, OPS_AGREEMENT_FACTOR):
            issues.append(
                warning(
                    "R9",
                    f"node {node_id}",
                    f"ops_claimed {claimed:g} and ops_estimated {estimated:g} differ by more than "
                    f"{OPS_AGREEMENT_FACTOR:g}x; check the loop ranges and body counts",
                )
            )
    return issues


def _within_factor(a: float, b: float, factor: float) -> bool:
    if a == 0 or b == 0:
        return a == b
    return max(a, b) / min(a, b) <= factor
