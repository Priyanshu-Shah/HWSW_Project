"""Costed TDG (v0.2, `TDG/schemas/tdg.schema.json`) + board file (`library/`) -> the partitioner's
IR TaskGraph + Platform (schema 0.1.0, `schemas/`). Decision D7 in `decisions.md`.

- Units: seconds -> ms, bytes/s -> MB/s (1e6 B), setup seconds -> µs.
- Each node becomes one implementation per way to run it: `sw` on a CPU core, plus one per HW design point
  (its `point` id) on the FPGA. `affinity` keeps only the allowed ones.
- `phase: setup` nodes and their edges are dropped: D3 counts one iteration only.
- Parallel edges (same src and dst, e.g. data + control) become one edge with the summed bytes.
- D3: one iteration, minimise its latency, `deadline_s` as a constraint. The arrival is always `one_shot`;
  the partitioner's streaming (pipelined) model is not used for the decision.
- Board: one exclusive PE per CPU core, one spatial PE for the FPGA fabric, the board's default interface
  between every pair of PEs, and DRAM (shared memory) between CPU cores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from schemas.ir import TaskGraph
from schemas.platform import Platform

FPGA_PE = "fpga0"
SW_IMPL = "sw"
# CostParam needs 0 < lo; an estimate whose `lo` is 0 keeps this much of its value as the low end.
MIN_LO_FACTOR = 1e-3


class ConversionError(ValueError):
    """The costed TDG or board cannot be turned into a partitioning problem."""


@dataclass
class Conversion:
    graph: TaskGraph
    platform: Platform
    setup_nodes: list[str] = field(default_factory=list)  # dropped (D3)


def cpu_pe_ids(board: dict) -> list[str]:
    return [f"cpu{i}" for i in range(int(board["cpu"]["cores"]))]


def convert(costed_tdg: dict, board: dict) -> Conversion:
    if "costs" not in costed_tdg:
        raise ConversionError("the TDG has no cost map (`costs`): run custom_hls first")
    if costed_tdg["platform"] != board["id"]:
        raise ConversionError(f"TDG is for board {costed_tdg['platform']!r}, board file is {board['id']!r}")
    if costed_tdg["costs"].get("edges"):
        raise ConversionError("per-edge cost overrides (`costs.edges`) are not supported yet")
    platform = convert_board(board)
    cpu_type, fpga_type = _pe_types(board)
    costs = {cost["id"]: cost for cost in costed_tdg["costs"]["nodes"]}

    setup = [node["id"] for node in costed_tdg["nodes"] if node["phase"] == "setup"]
    nodes = []
    for node in costed_tdg["nodes"]:
        if node["phase"] == "setup":
            continue
        cost = costs.get(node["id"])
        if cost is None:
            raise ConversionError(f"node {node['id']} has no cost")
        nodes.append(
            {
                "id": node["id"],
                "name": node["name"],
                "implementations": _implementations(node, cost, cpu_type, fpga_type),
            }
        )
    kept = {node["id"] for node in nodes}
    edges = _merged_edges([e for e in costed_tdg["edges"] if e["src"] in kept and e["dst"] in kept])
    deadline_s = costed_tdg.get("constraints", {}).get("deadline_s")
    graph = TaskGraph.model_validate(
        {
            "id": costed_tdg["graph"]["id"],
            "provenance": {
                "kind": "estimated",
                "source": "pre_TDG costed TDG (LLM pipeline + custom_hls)",
            },
            "arrival": {"kind": "one_shot"},
            "constraints": {"deadline_ms": deadline_s * 1e3} if deadline_s else {},
            "objective": {"weights": {"latency": 1.0}},
            "platform": {"id": board["id"]},
            "nodes": nodes,
            "edges": edges,
        }
    )
    return Conversion(graph=graph, platform=platform, setup_nodes=setup)


def convert_board(board: dict) -> Platform:
    cpu_type, fpga_type = _pe_types(board)
    cpus = cpu_pe_ids(board)
    pes = [{"id": pe_id, "type": cpu_type, "kind": "cpu", "concurrency": "exclusive"} for pe_id in cpus]
    pes.append(
        {
            "id": FPGA_PE,
            "type": fpga_type,
            "kind": "fpga",
            "concurrency": "spatial",
            "capacity": {key: float(value) for key, value in board["fpga"]["capacity"].items()},
        }
    )
    default = board["interfaces"]["default"]
    link = board["interfaces"][default]
    interfaces = [
        {
            "id": default,
            "kind": "axi_hp" if default.startswith("axi_hp") else "other",
            "connects": [*cpus, FPGA_PE],
            "setup_us": link["setup_s"] * 1e6,
            "bandwidth_mb_s": link["bandwidth_Bps"] / 1e6,
            "shared": True,
        }
    ]
    if len(cpus) > 1:
        interfaces.append(
            {
                "id": "dram",
                "kind": "shared_memory",
                "connects": cpus,
                "setup_us": 0.0,
                "bandwidth_mb_s": board["dram"]["bandwidth_Bps"] / 1e6,
                "shared": True,
            }
        )
    return Platform.model_validate(
        {
            "id": board["id"],
            "name": board.get("name"),
            "provenance": {"kind": "estimated", "source": f"library board file {board['id']}"},
            "pes": pes,
            "interfaces": interfaces,
        }
    )


def _merged_edges(edges: list[dict]) -> list[dict]:
    """The IR allows one edge per (src, dst); the TDG may have several (e.g. data + control):
    merge them into one carrying the sum of their bytes."""
    merged: dict[tuple[str, str], dict] = {}
    for edge in edges:
        key = (edge["src"], edge["dst"])
        if key in merged:
            merged[key]["bytes"] += edge["bytes"]
            merged[key]["name"] += f"+{edge['id']}"
        else:
            merged[key] = {"src": edge["src"], "dst": edge["dst"], "bytes": edge["bytes"], "name": edge["id"]}
    return list(merged.values())


def _pe_types(board: dict) -> tuple[str, str]:
    return f"cpu.{board['cpu']['name']}", f"fpga.{board['fpga']['part']}"


def _implementations(node: dict, cost: dict, cpu_type: str, fpga_type: str) -> list[dict]:
    affinity = node["affinity"]
    impls = []
    if affinity != "hw_only" and cost.get("sw"):
        impls.append({"id": SW_IMPL, "pe_type": cpu_type, "cost": {"time_ms": _param(cost["sw"]["time_s"])}})
    if affinity != "sw_only":
        for point in cost.get("hw", []):
            hw_cost = {
                "time_ms": _param(point["time_s"]),
                "resources": {key: float(value) for key, value in point["resources"].items()},
            }
            if point.get("ii_s"):
                hw_cost["ii_ms"] = point["ii_s"] * 1e3
            impls.append({"id": point["point"], "pe_type": fpga_type, "cost": hw_cost})
    if not impls:
        raise ConversionError(
            f"node {node['id']} ({affinity}) has no allowed cost: "
            + ("no sw cost" if affinity == "sw_only" else "no hw design point")
        )
    return impls


def _param(estimate: dict) -> dict:
    """{value, lo, hi} in seconds -> {expr, unc} in ms (unc = multiplicative range around the value)."""
    value = estimate["value"]
    if value <= 0:
        return {"expr": 0.0}
    lo = max(estimate["lo"] / value, MIN_LO_FACTOR)
    return {"expr": value * 1e3, "unc": (min(lo, 1.0), max(estimate["hi"] / value, 1.0))}
