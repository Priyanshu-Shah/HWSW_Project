"""R1-R3: ids are unique, edges point at existing nodes, and the graph has no cycles."""

from collections import Counter

import networkx as nx

from hwsw.validation.id_listing import describe_ids
from hwsw.validation.issue import Issue, error


def check_graph_structure(tdg: dict) -> list[Issue]:
    node_ids = [node["id"] for node in tdg["nodes"]]
    return [
        *_duplicate_ids("node", node_ids),
        *_duplicate_ids("edge", [edge["id"] for edge in tdg["edges"]]),
        *_edges_reference_existing_nodes(tdg["edges"], node_ids),
        *_cycles(tdg["edges"], set(node_ids)),
    ]


def _duplicate_ids(kind: str, ids: list[str]) -> list[Issue]:
    return [
        error(
            "R1", f"{kind} {item}", f"{kind} id '{item}' is used {count} times; ids must be unique"
        )
        for item, count in Counter(ids).items()
        if count > 1
    ]


def _edges_reference_existing_nodes(edges: list[dict], node_ids: list[str]) -> list[Issue]:
    issues = []
    for edge in edges:
        for end in ("src", "dst"):
            if edge[end] not in node_ids:
                issues.append(
                    error(
                        "R2",
                        f"edge {edge['id']}",
                        f"{end} '{edge[end]}' does not exist; "
                        f"existing node ids: {describe_ids(node_ids)}",
                    )
                )
    return issues


def _cycles(edges: list[dict], node_ids: set[str]) -> list[Issue]:
    graph = nx.DiGraph(
        (edge["src"], edge["dst"])
        for edge in edges
        if edge["src"] in node_ids and edge["dst"] in node_ids
    )
    try:
        cycle = nx.find_cycle(graph)
    except nx.NetworkXNoCycle:
        return []
    path = " -> ".join([cycle[0][0], *(step[1] for step in cycle)])
    return [error("R3", "edges", f"edges form a cycle ({path}); the graph must be acyclic")]
