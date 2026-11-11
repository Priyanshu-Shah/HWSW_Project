"""Hand-checkable instances used by the tests."""

from __future__ import annotations

from .instance import Instance, from_graph_platform

# Topcuoglu, Hariri & Wu (2002), Fig. 3 / Table 2: 10 tasks, 3 processors; communication costs are the average costs.
_W = {1: (14, 16, 9), 2: (13, 19, 18), 3: (11, 13, 19), 4: (13, 8, 17), 5: (12, 13, 10), 6: (13, 16, 9),
      7: (7, 15, 11), 8: (5, 11, 14), 9: (18, 12, 20), 10: (21, 7, 16)}
_C = {(1, 2): 18, (1, 3): 12, (1, 4): 9, (1, 5): 11, (1, 6): 14, (2, 8): 19, (2, 9): 16, (3, 7): 23, (4, 8): 27,
      (4, 9): 23, (5, 9): 13, (6, 8): 15, (7, 10): 17, (8, 10): 11, (9, 10): 13}
HEFT_PAPER_RANK_U = {1: 108.0, 2: 77.0, 3: 80.0, 4: 80.0, 5: 69.0, 6: 63.333, 7: 42.667, 8: 35.667, 9: 44.333,
                     10: 14.667}


def heft_paper_example() -> Instance:
    pes = [{"id": f"p{j}", "type": f"cpu.p{j}", "kind": "cpu", "concurrency": "exclusive", "capacity": {}}
           for j in range(3)]
    platform = {"pes": pes, "interfaces": [{"id": "net", "kind": "bus", "connects": [p["id"] for p in pes],
                                            "setup_us": 0.0, "bandwidth_mb_s": 1.0, "shared": False}]}
    nodes = [{"id": f"n{i}", "implementations": [
        {"id": f"i{j}", "pe_type": f"cpu.p{j}", "cost": {"time_ms": float(_W[i][j])}} for j in range(3)]}
        for i in range(1, 11)]
    edges = [{"src": f"n{a}", "dst": f"n{b}", "bytes": c * 1000.0} for (a, b), c in _C.items()]
    graph = {"id": "heft_paper", "nodes": nodes, "edges": edges, "constraints": {}}
    return from_graph_platform(graph, platform, name="heft_paper")
