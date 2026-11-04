"""Sanity bounds on the latency of a costed TDG, without running a partitioner.

- all-software: every node on the CPU.
- best case: every node on its fastest option, no communication cost, no FPGA size limit.
Any real partition lies between the two (communication can push it above all-software).
"""

from dataclasses import dataclass

import networkx as nx

# "Nothing worth moving to hardware" when all-software is within this factor of the best case.
NOTHING_TO_GAIN_FACTOR = 1.05


@dataclass(frozen=True)
class LatencyBounds:
    all_software_s: float | None  # None when some iteration node has no software option
    best_case_s: float
    best_case_optimistic_s: float  # best case using the `lo` end of every estimate
    deadline_s: float | None
    warnings: tuple[str, ...]


def compute_latency_bounds(costed_tdg: dict) -> LatencyBounds:
    iteration_nodes = [node["id"] for node in costed_tdg["nodes"] if node["phase"] == "iteration"]
    costs = {cost["id"]: cost for cost in costed_tdg["costs"]["nodes"]}
    graph = _iteration_graph(costed_tdg, iteration_nodes)

    software = {node_id: _software_time(costs, node_id, "value") for node_id in iteration_nodes}
    all_software = None if None in software.values() else _critical_path_s(graph, software)
    best_case = _critical_path_s(
        graph, {n: _fastest_time(costs, n, "value") for n in iteration_nodes}
    )
    optimistic = _critical_path_s(
        graph, {n: _fastest_time(costs, n, "lo") for n in iteration_nodes}
    )

    deadline = costed_tdg.get("constraints", {}).get("deadline_s")
    return LatencyBounds(
        all_software_s=all_software,
        best_case_s=best_case,
        best_case_optimistic_s=optimistic,
        deadline_s=deadline,
        warnings=tuple(_warnings(all_software, best_case, optimistic, deadline)),
    )


def _iteration_graph(costed_tdg: dict, iteration_nodes: list[str]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(iteration_nodes)
    graph.add_edges_from(
        (edge["src"], edge["dst"])
        for edge in costed_tdg["edges"]
        if edge["src"] in graph and edge["dst"] in graph
    )
    return graph


def _software_time(costs: dict[str, dict], node_id: str, bound: str) -> float | None:
    sw = costs.get(node_id, {}).get("sw")
    return sw["time_s"][bound] if sw else None


def _fastest_time(costs: dict[str, dict], node_id: str, bound: str) -> float:
    cost = costs.get(node_id, {})
    options = [point["time_s"][bound] for point in cost.get("hw", [])]
    software = _software_time(costs, node_id, bound)
    if software is not None:
        options.append(software)
    if not options:
        raise ValueError(f"node {node_id} has no software or hardware cost")
    return min(options)


def _critical_path_s(graph: nx.DiGraph, node_time_s: dict[str, float]) -> float:
    """Longest chain of node times through the graph (nodes on parallel branches overlap)."""
    finish: dict[str, float] = {}
    for node_id in nx.topological_sort(graph):
        start = max((finish[pred] for pred in graph.predecessors(node_id)), default=0.0)
        finish[node_id] = start + node_time_s[node_id]
    return max(finish.values(), default=0.0)


def _warnings(
    all_software: float | None, best_case: float, optimistic: float, deadline: float | None
) -> list[str]:
    warnings = []
    if deadline is not None and optimistic > deadline:
        warnings.append(
            "no partition can meet the deadline: even with every node on its fastest option and "
            f"no communication, latency is at least {optimistic * 1e3:.3g} ms "
            f"(deadline {deadline * 1e3:.3g} ms)"
        )
    if all_software is not None and all_software <= best_case * NOTHING_TO_GAIN_FACTOR:
        warnings.append(
            "hardware cannot improve latency by more than "
            f"{(NOTHING_TO_GAIN_FACTOR - 1):.0%}; check the graph and the costs"
        )
    return warnings
