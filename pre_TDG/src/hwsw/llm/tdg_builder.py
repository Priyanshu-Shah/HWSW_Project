"""Stage B: task spec -> graph-only TDG, in small steps, each checked before the next.

1. nodes: the list of tasks (id, name, phase, affinity, kind, one line of purpose)
2. edges: dependencies; sizes may be formulas, evaluated here (D4)
3. payloads: per node, the opaque time, the template parameters or the loop nest (D1)
4. validate -> repair: each error goes back to the step that made it (at most 3 rounds)
5. ops check: custom_hls costs the graph; nodes whose op count is off by more than 2x, or
   whose CPU time is implausible (over 100x the deadline), are redone once, then warned about
6. network check: when the spec names neural networks with their size, the network's
   conv2d/gemm nodes must add up to it within 2x; they are rescaled once, then warned about
"""

import json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

from hwsw.custom_hls import CostEstimationError, HlsConstants, estimate_costs, load_constants
from hwsw.llm.client import LlmError, OllamaClient
from hwsw.llm.formulas import evaluate_count, evaluate_formula
from hwsw.llm.schema_parts import tdg_ref, with_tdg_defs
from hwsw.platforms import Board
from hwsw.validation import ValidationReport, validate_tdg

PROMPT_PATH = Path(__file__).parent / "prompts" / "tdg_builder.md"
MAX_REPAIR_ROUNDS = 3
MAX_STEP_ATTEMPTS = 3
NODE_FIELDS = ("id", "name", "phase", "affinity", "kind")
NETWORK_OPS = ("conv2d", "gemm")  # the template ops that make up a neural network's MACs
SIZE_AGREEMENT_FACTOR = 2.0
# A node needing more CPU time than this many deadlines (or periods) per iteration is almost
# always a counting error (e.g. a loop range that is already a product of other loops).
IMPLAUSIBLE_TIME_FACTOR = 100

_FORMULA = {"type": ["integer", "number", "string"], "description": "a number or a formula"}
NODES_SCHEMA = with_tdg_defs(
    {
        "type": "object",
        "required": ["graph_id", "nodes"],
        "properties": {
            "graph_id": {**tdg_ref("identifier"), "description": "short snake_case task name"},
            "nodes": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": [*NODE_FIELDS, "purpose"],
                    "properties": {
                        "id": tdg_ref("identifier"),
                        "name": {"type": "string", "minLength": 1},
                        "phase": {"enum": ["setup", "iteration"]},
                        "affinity": {"enum": ["sw_only", "hw_only", "any"]},
                        "kind": {"enum": ["opaque", "loop_nest", "template"]},
                        "purpose": {"type": "string"},
                    },
                },
            },
        },
    }
)
EDGES_SCHEMA = {
    "type": "object",
    "required": ["edges"],
    "properties": {
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "src", "dst", "kind", "bytes"],
                "properties": {
                    "id": {"type": "string"},
                    "src": {"type": "string"},
                    "dst": {"type": "string"},
                    "kind": {"enum": ["data", "control"]},
                    "bytes": {**_FORMULA, "description": "bytes per iteration; 0 for control"},
                },
            },
        }
    },
}
PAYLOAD_SCHEMAS = {
    "opaque": with_tdg_defs(
        {"type": "object", "required": ["opaque"], "properties": {"opaque": tdg_ref("opaque")}}
    ),
    "template": with_tdg_defs(
        {
            "type": "object",
            "required": ["template"],
            "properties": {"template": tdg_ref("template")},
        }
    ),
    "loop_nest": with_tdg_defs(
        {
            "type": "object",
            "required": ["loop_nest", "ops_claimed", "ops_unit"],
            "properties": {
                "loop_nest": tdg_ref("loop_nest"),
                "ops_claimed": _FORMULA,
                "ops_unit": {"enum": ["ops", "macs"]},
            },
        }
    ),
}

_NODES_REQUEST = """Step 1: list the nodes of the graph, in execution order, each with one line
of purpose.

Check before answering:
- `template` only for work that IS one of the six operations: conv2d, gemm, pool, elementwise,
  reduce, memcpy. Resizing, color conversion, filtering, sorting, non-maximum suppression and
  similar are `loop_nest`.
- A neural network is never one node: use 3 to 8 `template` nodes (for example backbone stages
  as `conv2d`, a `gemm` or `conv2d` head), with any non-template parts as `loop_nest`.
- Loading weights or other one-time work is `phase: setup`."""
_EDGES_REQUEST = """Step 2: list the edges between these nodes.

$nodes

A `data` edge carries the output of `src` to `dst`; `bytes` is the size moved per iteration
(a formula is fine, e.g. "1280*720*3"). A `control` edge only orders two nodes; its `bytes` is 0.
The edges must form a directed acyclic graph. Every node except the first ones needs an input."""
_PAYLOAD_REQUESTS = {
    "opaque": """Estimate the CPU time of node $id ($name: $purpose) on this board, per
iteration, in seconds. Use `"source": "llm"`.""",
    "template": """Give the template operation and its parameters for node $id ($name: $purpose).
Its input and output sizes should match its edges.""",
    "loop_nest": """Describe node $id ($name: $purpose) as a loop nest (decision D1):
- `arrays`: every array read or written, with shape, dtype, and `dir` (`in`/`out`); input and
  output arrays should match the node's edges. Constant data such as coefficients: `"const": true`.
- `loops`: outermost first, perfectly nested, `range` = that loop's own trip count (the total
  number of iterations is the product of the ranges; do not put a product in one range).
- `body`: operation counts per innermost iteration (`mac`, `add`, `mul`, `div`, `cmp`, `logic`,
  `shift`, `sqrt`, `exp`, ...). One multiply-accumulate is one `mac`. `load`/`store` are not ops.
- `access`: per array, `streaming` (each element read once), `sliding_window` (with the loop vars
  in `reuse`), `reuse` (the whole array re-read) or `random`.
- `control`: data-dependent `branches` per innermost iteration; `control_heavy: true` for
  irregular code (data-dependent loop bounds, pointer chasing).
- `hints.unrollable`: loop vars that can safely run in parallel (advice only).
- `ops_claimed`: your own count of the operations = product of all loop ranges x the sum of
  the body counts (a formula is fine); `ops_unit`: `"macs"` if the body has only `mac`,
  else `"ops"`.""",
}


class TdgBuildError(LlmError):
    """No valid TDG within the repair budget."""


@dataclass
class TdgBuild:
    tdg: dict  # graph only (no `costs`); passes the validator
    warnings: list[str] = field(default_factory=list)  # left after the repair budget


class _Step:
    """One schema-checked conversation. Kept, so its answer can be repaired with context."""

    def __init__(self, client: OllamaClient, system: str, request: str, schema: dict):
        self._client = client
        self._schema = schema
        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": request},
        ]

    def ask(self, convert: Callable[[Any], Any] = lambda value: value) -> Any:
        """The reply, passed through `convert`; a ValueError from it is sent back to fix."""
        problem = ""
        for _ in range(MAX_STEP_ATTEMPTS):
            value = self._client.chat_json(self.messages, self._schema)
            self.messages.append({"role": "assistant", "content": json.dumps(value)})
            try:
                return convert(value)
            except ValueError as not_usable:
                problem = str(not_usable)
                self._request_fix([problem])
        raise TdgBuildError(f"no usable reply after {MAX_STEP_ATTEMPTS} attempts: {problem}")

    def fix(self, problems: list[str], convert: Callable[[Any], Any]) -> Any:
        self._request_fix(problems)
        return self.ask(convert)

    def _request_fix(self, problems: list[str]) -> None:
        listed = "\n".join(f"- {problem}" for problem in problems)
        self.messages.append(
            {
                "role": "user",
                "content": f"The checker found these problems:\n{listed}\n"
                "Reply with the whole corrected JSON.",
            }
        )


class TdgBuilder:
    def __init__(self, client: OllamaClient, board: Board, constants: HlsConstants | None = None):
        self._client = client
        self._board = board
        self._constants = constants or load_constants()

    def build(self, spec: dict) -> TdgBuild:
        system = _system_prompt(spec, self._board)
        self._network_note = _network_note(spec)
        nodes_request = _NODES_REQUEST + self._network_note
        nodes_step = _Step(self._client, system, nodes_request, NODES_SCHEMA)
        decomposition = nodes_step.ask(_check_node_list)
        self._client.log.write("stage_b", step="nodes", result=decomposition)
        nodes = decomposition["nodes"]

        node_lines = "\n".join(
            f"- {node['id']} {node['name']} ({node['kind']}, {node['phase']}): {node['purpose']}"
            for node in nodes
        )
        edges_request = Template(_EDGES_REQUEST).substitute(nodes=node_lines)
        self._edges_step = _Step(self._client, system, edges_request, EDGES_SCHEMA)
        edges = self._edges_step.ask(_convert_edges)

        self._node_steps: dict[str, _Step] = {}
        payloads = {node["id"]: self._payload(system, node, edges) for node in nodes}
        tdg = _assemble(spec, decomposition["graph_id"], nodes, edges, payloads)
        tdg = self._repair_until_valid(tdg)
        return self._check_network_size(self._check_ops(tdg), spec)

    def _payload(self, system: str, node: dict, edges: list[dict]) -> dict:
        connected = [edge for edge in edges if node["id"] in (edge["src"], edge["dst"])]
        request = (
            Template(_PAYLOAD_REQUESTS[node["kind"]]).substitute(node)
            + "\n\nIts edges: "
            + (json.dumps(connected) if connected else "none")
            + (self._network_note if node["kind"] == "template" else "")
        )
        step = _Step(self._client, system, request, PAYLOAD_SCHEMAS[node["kind"]])
        self._node_steps[node["id"]] = step
        return step.ask(_convert_payload)

    def _repair_until_valid(self, tdg: dict) -> dict:
        for _ in range(MAX_REPAIR_ROUNDS):
            problems = self._problems(tdg)
            self._client.log.write("stage_b", step="validate", problems=problems)
            if not problems:
                return tdg
            tdg = self._fix(tdg, problems)
        problems = self._problems(tdg)
        if problems:
            listed = "; ".join(f"{owner}: {', '.join(items)}" for owner, items in problems.items())
            raise TdgBuildError(f"TDG still invalid after {MAX_REPAIR_ROUNDS} repairs: {listed}")
        return tdg

    def _problems(self, tdg: dict) -> dict[str, list[str]]:
        """Errors grouped by the step that must fix them: a node id, or "edges"."""
        report = validate_tdg(tdg, self._board)
        if report.is_valid:
            try:
                estimate_costs(tdg, self._board, self._constants)
            except CostEstimationError as failure:
                return _group_by_owner(failure.problems, tdg)
            return {}
        return _group_by_owner([str(issue) for issue in report.errors], tdg)

    def _fix(self, tdg: dict, problems: dict[str, list[str]]) -> dict:
        nodes = {node["id"]: node for node in tdg["nodes"]}
        edges = tdg["edges"]
        for owner, items in problems.items():
            if owner == _GRAPH:  # made by this code from the spec, not by a step
                raise TdgBuildError("TDG has graph-level errors: " + "; ".join(items))
            if owner == "edges":
                edges = self._edges_step.fix(items, _convert_edges)
            else:
                node = nodes[owner]
                base = {key: node[key] for key in NODE_FIELDS}
                nodes[owner] = {**base, **self._node_steps[owner].fix(items, _convert_payload)}
        return {**tdg, "nodes": list(nodes.values()), "edges": edges}

    def _check_ops(self, tdg: dict) -> TdgBuild:
        """D1's 2x check and a plausibility check: redo each failing node once; what still fails
        is a warning."""
        mismatches = self._ops_problems(tdg)
        if mismatches:
            tdg = self._repair_until_valid(self._fix(tdg, mismatches))
            mismatches = self._ops_problems(tdg)
        warnings = [f"{node}: {' '.join(items)}" for node, items in mismatches.items()]
        if warnings:
            self._client.log.write("stage_b", step="ops_check", warnings=warnings)
        return TdgBuild(tdg, warnings)

    def _check_network_size(self, build: TdgBuild, spec: dict) -> TdgBuild:
        """The network's conv2d/gemm nodes must add up to its known size: rescale them once."""
        target_macs = sum(network["gmacs"] for network in spec.get("networks") or []) * 1e9
        if not target_macs:
            return build
        summary, problems = self._network_problems(build.tdg, target_macs, spec)
        tdg = build.tdg
        if problems:
            tdg = self._repair_until_valid(self._fix(tdg, problems))
            summary, _ = self._network_problems(tdg, target_macs, spec)
        if not summary:
            return TdgBuild(tdg, build.warnings)
        self._client.log.write("stage_b", step="network_check", warning=summary)
        return TdgBuild(tdg, [*build.warnings, f"network size: {summary}"])

    def _network_problems(
        self, tdg: dict, target_macs: float, spec: dict
    ) -> tuple[str, dict[str, list[str]]]:
        """("", {}) when the size agrees; else what is wrong, and a fix request per network node."""
        costed = estimate_costs(tdg, self._board, self._constants)
        macs = {cost["id"]: cost.get("ops_estimated") or 0 for cost in costed["costs"]["nodes"]}
        network_nodes = [
            node["id"]
            for node in tdg["nodes"]
            if node["kind"] == "template" and node["template"]["op"] in NETWORK_OPS
        ]
        total = sum(macs[node_id] for node_id in network_nodes)
        names = ", ".join(network["name"] for network in spec["networks"])
        if not network_nodes:
            return f"the graph has no conv2d/gemm template nodes for {names}", {}
        ratio = total / target_macs
        if 1 / SIZE_AGREEMENT_FACTOR <= ratio <= SIZE_AGREEMENT_FACTOR:
            return "", {}
        summary = (
            f"the network nodes ({', '.join(network_nodes)}) add up to {total / 1e9:.3g} GMACs, "
            f"but {names} has {target_macs / 1e9:.3g} GMACs per inference ({ratio:.2g}x)."
        )
        # Each node gets its own share (in proportion to its size now); told only the network's
        # target, every node would try to close the whole gap and the total would overshoot.
        shares = {
            node_id: target_macs * (macs[node_id] / total if total else 1 / len(network_nodes))
            for node_id in network_nodes
        }
        return summary, {
            node_id: [
                f"{summary} This node has {macs[node_id] / 1e9:.3g} GMACs; its share of the "
                f"network is about {shares[node_id] / 1e9:.3g} GMACs. Change its parameters "
                "(channels, resolution, kernel size; consistent with its edges) so that it has "
                "about that many multiply-accumulates."
            ]
            for node_id in network_nodes
        }

    def _ops_problems(self, tdg: dict) -> dict[str, list[str]]:
        costed = estimate_costs(tdg, self._board, self._constants)
        report: ValidationReport = validate_tdg(costed, self._board)
        r9 = [str(issue) for issue in report.warnings if issue.rule == "R9"]
        problems = _group_by_owner(r9, tdg)
        for node_id, problem in _implausible_times(costed):
            problems.setdefault(node_id, []).append(problem)
        return problems


def _system_prompt(spec: dict, board: Board) -> str:
    board_text = json.dumps(board.model_dump(), indent=1)
    return Template(PROMPT_PATH.read_text(encoding="utf-8")).substitute(
        spec=json.dumps(spec, indent=1), board=board_text
    )


def _implausible_times(costed: dict) -> list[tuple[str, str]]:
    time_basis = costed.get("constraints", {}).get("deadline_s") or costed["arrival"].get(
        "period_s"
    )
    if not time_basis:
        return []
    found = []
    for cost in costed["costs"]["nodes"]:
        seconds = cost.get("sw", {}).get("time_s", {}).get("value")
        if seconds and seconds > IMPLAUSIBLE_TIME_FACTOR * time_basis:
            ops = f" ({cost['ops_estimated']:.3g} ops)" if cost.get("ops_estimated") else ""
            found.append(
                (
                    cost["id"],
                    f"node {cost['id']}: takes {seconds:.3g} s on the CPU per iteration{ops}, "
                    f"over {IMPLAUSIBLE_TIME_FACTOR}x the {time_basis:.3g} s deadline: that is "
                    "implausible. Check the sizes: each loop's range is that loop's own trip count "
                    "(nested loops multiply), and everything is per iteration.",
                )
            )
    return found


def _network_note(spec: dict) -> str:
    networks = spec.get("networks") or []
    if not networks:
        return ""
    listed = "; ".join(
        f"{network['name']}: {network['gmacs']:g} GMACs per inference"
        + (f" (input {network['input']})" if network.get("input") else "")
        for network in networks
    )
    return (
        f"\n\nNeural network size: {listed}. Its `conv2d`/`gemm` template nodes together must "
        "add up to about that many multiply-accumulates."
    )


def _check_node_list(decomposition: dict) -> dict:
    ids = [node["id"] for node in decomposition["nodes"]]
    duplicates = sorted({node_id for node_id in ids if ids.count(node_id) > 1})
    if duplicates:
        raise ValueError(f"node ids must be unique; repeated: {', '.join(duplicates)}")
    for node in decomposition["nodes"]:
        if node["kind"] == "opaque" and node["affinity"] != "sw_only":
            raise ValueError(f"node {node['id']} is opaque, so its affinity must be sw_only")
    return decomposition


def _convert_edges(reply: dict) -> list[dict]:
    edges = []
    for edge in reply["edges"]:
        try:
            edges.append({**edge, "bytes": evaluate_count(edge["bytes"])})
        except ValueError as problem:
            raise ValueError(f"edge {edge['id']}: bytes {problem}") from problem
    return edges


def _convert_payload(reply: dict) -> dict:
    reply = {key: value for key, value in reply.items() if key in _PAYLOAD_KEYS}
    if "ops_claimed" in reply:
        try:
            return {**reply, "ops_claimed": evaluate_formula(reply["ops_claimed"])}
        except ValueError as problem:
            raise ValueError(f"ops_claimed: {problem}") from problem
    return reply


def _assemble(
    spec: dict, graph_id: str, nodes: list[dict], edges: list[dict], payloads: dict[str, dict]
) -> dict:
    tdg = {
        "graph": {
            "id": graph_id,
            "schema_version": "0.2",
            "provenance": {"kind": "llm", "source": spec["task_summary"]},
        },
        "arrival": spec["arrival"],
        "objective": {"latency": 1.0},
        "platform": spec["platform"],
        "nodes": [
            {**{key: node[key] for key in NODE_FIELDS}, **payloads[node["id"]]} for node in nodes
        ],
        "edges": edges,
    }
    if "deadline_s" in spec["timing"]:
        tdg["constraints"] = {"deadline_s": spec["timing"]["deadline_s"]}
    return tdg


_PAYLOAD_KEYS = ("opaque", "template", "loop_nest", "ops_claimed", "ops_unit")
_GRAPH = "graph"
_NODE_LOCATION = re.compile(r"^(?:node (\w+)|nodes\[(\d+)\])")


def _group_by_owner(problems: list[str], tdg: dict) -> dict[str, list[str]]:
    """Problems grouped by who must fix them: a node id, "edges", or "graph" (this code).

    Problems are validator issues (`error [R2] edge e1: ...`) or cost errors (`node n3: ...`).
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    node_ids = [node["id"] for node in tdg["nodes"]]
    for problem in problems:
        location = problem.split("] ", 1)[-1].split(": ", 1)[0]
        match = _NODE_LOCATION.match(location)
        if match and match.group(1) in node_ids:
            grouped[match.group(1)].append(problem)
        elif match and match.group(2) is not None and int(match.group(2)) < len(node_ids):
            grouped[node_ids[int(match.group(2))]].append(problem)
        elif location.startswith("edge"):
            grouped["edges"].append(problem)
        else:
            grouped[_GRAPH].append(problem)
    return dict(grouped)
