"""Stage B: task spec -> graph-only TDG, by small agents in a LangGraph graph (D6).

LLM agents, each with its own short conversation:
- researcher: per computation step, the standard method and its work per item (web search)
- decomposer: the list of nodes (id, name, phase, affinity, kind, one line of purpose)
- edge_builder: dependencies; sizes may be formulas, evaluated here (D4)
- opaque_estimator, template_designer, loop_nest_writer: one payload per node, by kind (D1)

Checkers (code, no LLM) send each problem back to the agent that made it, which fixes it in its
own conversation:
- validator: schema, rules R1-R12 and costability; at most 3 repair rounds, then it fails
- network_checker: a network's conv2d/gemm nodes must add up to its known size within 2x;
  rescaled once, then warned about
- ops_checker: nodes whose op count is off by more than 2x (R9), or whose CPU time is
  implausible (over 100x the deadline), are redone once, then warned about. It runs after the
  network check, and does not judge network nodes whose size that check already set.
"""

import json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from string import Template
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from hwsw.conversation import AssistantMessage
from hwsw.custom_hls import CostEstimationError, HlsConstants, estimate_costs, load_constants
from hwsw.llm.client import LlmError, OllamaClient
from hwsw.llm.formulas import evaluate_count, evaluate_formula
from hwsw.llm.graph_conversation import RECURSION_LIMIT
from hwsw.llm.graph_state import PipelineState
from hwsw.llm.schema_parts import tdg_ref, with_tdg_defs
from hwsw.llm.tool_loop import StepLimitError, run_tool_loop
from hwsw.llm.tools import Tool, object_schema
from hwsw.platforms import Board
from hwsw.validation import ValidationReport, validate_tdg

PROMPT_PATH = Path(__file__).parent / "prompts" / "tdg_builder.md"
RESEARCH_PROMPT_PATH = Path(__file__).parent / "prompts" / "research.md"
MAX_RESEARCH_STEPS = 10
MAX_REPAIR_ROUNDS = 3
MAX_STEP_ATTEMPTS = 3
NODE_FIELDS = ("id", "name", "phase", "affinity", "kind")
NETWORK_OPS = ("conv2d", "gemm")  # the template ops that make up a neural network's MACs
SIZE_AGREEMENT_FACTOR = 2.0
# A node needing more CPU time than this many deadlines (or periods) per iteration is almost
# always a counting error (e.g. a loop range that is already a product of other loops).
IMPLAUSIBLE_TIME_FACTOR = 100
# A total the spec states may be in FLOPs while nodes count MACs (2x), on top of D1's 2x.
STATED_TOTAL_FACTOR = 4.0

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
- Each matrix product, even a small one, is its own `gemm` node (a Kalman filter's H P H^T is
  two products, so two nodes).
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
Its input and output sizes should match its edges. Use the exact sizes from the spec, including
`batch` when the same operation runs on several inputs.""",
    "loop_nest": """Describe node $id ($name: $purpose) as a loop nest (decision D1):
- `arrays`: every array read or written, with shape, dtype, and `dir` (`in`/`out`); input and
  output arrays should match the node's edges. Constant data such as coefficients: `"const": true`.
- `loops`: outermost first, perfectly nested, `range` = that loop's own trip count (the total
  number of iterations is the product of the ranges; do not put a product in one range).
  One loop nest is one computation: steps that run one after another (such as A*B, then the
  result times C) are never nested inside each other, because nested loops multiply.
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


SUBMIT_NOTES = Tool(
    name="submit_notes",
    description="Submit the work notes: per computation step, the method and its work per item.",
    parameters=object_schema(
        {
            "notes": {
                "type": "array",
                "items": object_schema(
                    {
                        "step": {"type": "string"},
                        "method": {"type": "string"},
                        "work": {"type": "string", "description": "work per item, with arithmetic"},
                        "source": {"type": "string", "description": 'a URL, or "known"'},
                    },
                    required=["step", "method", "work"],
                ),
            }
        },
        required=["notes"],
    ),
)
PAYLOAD_AGENTS = {
    "opaque": "opaque_estimator",
    "template": "template_designer",
    "loop_nest": "loop_nest_writer",
}
CHECKS = ("network_checker", "ops_checker")
_STOP_RESEARCH = (
    "No more searching. Call submit_notes now, counting the work from what you know of each "
    "method."
)  # in this order, after the graph is valid


class TdgBuildError(LlmError):
    """No valid TDG within the repair budget."""


@dataclass
class TdgBuild:
    tdg: dict  # graph only (no `costs`); passes the validator
    warnings: list[str] = field(default_factory=list)  # left after the repair budget


class TdgBuilder:
    """Stage B alone: spec -> TDG (the `hwsw build` command and tests)."""

    def __init__(
        self,
        client: OllamaClient,
        board: Board,
        constants: HlsConstants | None = None,
        tools: list[Tool] | None = None,  # the researcher's lookup tools; default: none
    ):
        self._graph = StateGraph(PipelineState)
        add_tdg_builder(self._graph, client, lambda _: board, constants, tools or [])
        self._graph.add_edge(START, "researcher")

    def build(self, spec: dict) -> TdgBuild:
        state = self._graph.compile().invoke({"spec": spec}, {"recursion_limit": RECURSION_LIMIT})
        return TdgBuild(state["tdg"], state.get("warnings", []))


def add_tdg_builder(
    graph: StateGraph,
    client: OllamaClient,
    board_for: Callable[[str], Board],
    constants: HlsConstants | None,
    tools: list[Tool],
    then: str = END,
) -> None:
    """Adds the stage-B agents and checkers; the caller routes to `researcher` to enter."""
    agents = _BuildAgents(client, board_for, constants or load_constants(), tools)
    graph.add_node("researcher", agents.research)
    graph.add_node("decomposer", agents.decompose)
    graph.add_node("edge_builder", agents.build_edges)
    for kind, name in PAYLOAD_AGENTS.items():
        graph.add_node(name, agents.payload_agent(kind))
    graph.add_node("validator", agents.validate)
    graph.add_node("network_checker", agents.check_network)
    graph.add_node("ops_checker", agents.check_ops)

    graph.add_edge("researcher", "decomposer")
    graph.add_edge("decomposer", "edge_builder")
    work = [*PAYLOAD_AGENTS.values(), "edge_builder", "validator"]
    for name in ["edge_builder", *PAYLOAD_AGENTS.values()]:
        graph.add_conditional_edges(name, _next_work, work)
    for name in ["validator", *CHECKS]:
        graph.add_conditional_edges(
            name,
            lambda state: _next_after_check(state, then),
            [*PAYLOAD_AGENTS.values(), "edge_builder", *CHECKS, then],
        )


def _next_work(state: PipelineState) -> str:
    """The agent for the first open fix, else for the next node without a payload, else check."""
    if state.get("fixes"):
        owner = next(iter(state["fixes"]))
        return "edge_builder" if owner == "edges" else PAYLOAD_AGENTS[_kind(state, owner)]
    if state.get("todo"):
        return PAYLOAD_AGENTS[_kind(state, state["todo"][0])]
    return "validator"


def _next_after_check(state: PipelineState, then: str) -> str:
    if state.get("fixes"):
        return _next_work(state)
    checks = state.get("checks", {})
    return next((name for name in CHECKS if checks.get(name, 0) < 2), then)


def _kind(state: PipelineState, node_id: str) -> str:
    return next(node["kind"] for node in state["nodes"] if node["id"] == node_id)


class _Step:
    """One schema-checked conversation, kept in the state so its answer can be repaired."""

    def __init__(self, client: OllamaClient, schema: dict, messages: list[dict]):
        self._client = client
        self._schema = schema
        self.messages = list(messages)  # a copy: a failed step leaves the state as is

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


class _BuildAgents:
    """The node functions. Each takes the state and returns only the keys it changes."""

    def __init__(
        self,
        client: OllamaClient,
        board_for: Callable[[str], Board],
        constants: HlsConstants,
        tools: list[Tool],
    ):
        self._client = client
        self._board_for = cache(board_for)
        self._constants = constants
        self._tools = [*tools, SUBMIT_NOTES]

    # LLM agents

    def research(self, state: PipelineState) -> dict:
        get_stream_writer()(
            AssistantMessage(text="Building the task graph; this takes a few minutes.")
        )
        prompt = Template(RESEARCH_PROMPT_PATH.read_text(encoding="utf-8")).substitute(
            spec=json.dumps(state["spec"], indent=1)
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Write the work notes for this task."},
        ]
        try:
            call = run_tool_loop(self._client, messages, self._tools, MAX_RESEARCH_STEPS)
        except StepLimitError:  # still searching: it must write the notes from what it knows
            messages.append({"role": "user", "content": _STOP_RESEARCH})
            call = run_tool_loop(self._client, messages, [SUBMIT_NOTES], max_steps=3)
        notes = call.arguments["notes"]
        self._client.log.write("stage_b", step="research", result=notes)
        return {"research": notes}

    def decompose(self, state: PipelineState) -> dict:
        request = _NODES_REQUEST + _network_note(state["spec"])
        step = _Step(self._client, NODES_SCHEMA, self._opening(state, request))
        decomposition = step.ask(_check_node_list)
        self._client.log.write("stage_b", step="nodes", result=decomposition)
        nodes = decomposition["nodes"]
        return {
            "graph_id": decomposition["graph_id"],
            "nodes": nodes,
            "todo": [node["id"] for node in nodes],
            "payloads": {},
            "chats": {"nodes": step.messages},
            "fixes": {},
            "repairs": 0,
            "checks": {},
            "warnings": [],
        }

    def build_edges(self, state: PipelineState) -> dict:
        fixes = dict(state["fixes"])
        if "edges" in fixes:
            step = _Step(self._client, EDGES_SCHEMA, state["chats"]["edges"])
            edges = step.fix(fixes.pop("edges"), _convert_edges)
        else:
            node_lines = "\n".join(
                f"- {node['id']} {node['name']} ({node['kind']}, {node['phase']}): "
                f"{node['purpose']}"
                for node in state["nodes"]
            )
            request = Template(_EDGES_REQUEST).substitute(nodes=node_lines)
            step = _Step(self._client, EDGES_SCHEMA, self._opening(state, request))
            edges = step.ask(_convert_edges)
        return {"edges": edges, "fixes": fixes, "chats": {**state["chats"], "edges": step.messages}}

    def payload_agent(self, kind: str) -> Callable[[PipelineState], dict]:
        """One agent per node kind; it writes, or fixes, one node's payload per turn."""

        def write_payload(state: PipelineState) -> dict:
            fixes, todo = dict(state["fixes"]), list(state["todo"])
            if fixes:
                node_id = next(iter(fixes))
                step = _Step(self._client, PAYLOAD_SCHEMAS[kind], state["chats"][node_id])
                payload = step.fix(fixes.pop(node_id), _convert_payload)
            else:
                node_id = todo.pop(0)
                node = next(node for node in state["nodes"] if node["id"] == node_id)
                step = _Step(
                    self._client,
                    PAYLOAD_SCHEMAS[kind],
                    self._opening(state, _payload_request(node, state["edges"], state["spec"])),
                )
                payload = step.ask(_convert_payload)
            return {
                "payloads": {**state["payloads"], node_id: payload},
                "fixes": fixes,
                "todo": todo,
                "chats": {**state["chats"], node_id: step.messages},
            }

        write_payload.__name__ = PAYLOAD_AGENTS[kind]
        return write_payload

    # checkers (no LLM)

    def validate(self, state: PipelineState) -> dict:
        tdg = _assemble(state)
        problems = self._problems(tdg, self._board_for(state["spec"]["platform"]))
        self._client.log.write("stage_b", step="validate", problems=problems)
        if not problems:
            return {"tdg": tdg, "repairs": 0}
        if _GRAPH in problems:  # made by this code from the spec, not by an agent
            raise TdgBuildError("TDG has graph-level errors: " + "; ".join(problems[_GRAPH]))
        if state["repairs"] >= MAX_REPAIR_ROUNDS:
            listed = "; ".join(f"{owner}: {', '.join(items)}" for owner, items in problems.items())
            raise TdgBuildError(f"TDG still invalid after {MAX_REPAIR_ROUNDS} repairs: {listed}")
        return {"tdg": tdg, "fixes": problems, "repairs": state["repairs"] + 1}

    def check_network(self, state: PipelineState) -> dict:
        """The network's conv2d/gemm nodes must add up to its known size: rescale them once."""
        target_macs = _target_macs(state["spec"])
        summary, problems = "", {}
        if target_macs:
            summary, problems = self._network_problems(state, target_macs)
        return self._after_check(state, "network_checker", problems, [f"network size: {summary}"])

    def check_ops(self, state: PipelineState) -> dict:
        """D1's 2x check and a plausibility check: redo each failing node once."""
        problems = self._ops_problems(state)
        warnings = [f"{node}: {' '.join(items)}" for node, items in problems.items()]
        return self._after_check(state, "ops_checker", problems, warnings)

    def _after_check(
        self, state: PipelineState, name: str, problems: dict, warnings: list[str]
    ) -> dict:
        """First run: the problems go back to their agents. Second run: what is left is warned."""
        checks = dict(state["checks"])
        if problems and not checks.get(name):
            checks[name] = 1
            return {"checks": checks, "fixes": problems}
        checks[name] = 2
        if not problems:
            return {"checks": checks}
        self._client.log.write("stage_b", step=name, warnings=warnings)
        return {"checks": checks, "warnings": [*state["warnings"], *warnings]}

    def _problems(self, tdg: dict, board: Board) -> dict[str, list[str]]:
        """Errors grouped by the agent that must fix them: a node id, or "edges"."""
        report = validate_tdg(tdg, board)
        if report.is_valid:
            try:
                estimate_costs(tdg, board, self._constants)
            except CostEstimationError as failure:
                return _group_by_owner(failure.problems, tdg)
            return {}
        return _group_by_owner([str(issue) for issue in report.errors], tdg)

    def _costed(self, state: PipelineState) -> dict:
        board = self._board_for(state["spec"]["platform"])
        return estimate_costs(state["tdg"], board, self._constants)

    def _network_problems(
        self, state: PipelineState, target_macs: float
    ) -> tuple[str, dict[str, list[str]]]:
        """("", {}) when the size agrees; else what is wrong, and a fix request per network node."""
        macs = {
            cost["id"]: cost.get("ops_estimated") or 0
            for cost in self._costed(state)["costs"]["nodes"]
        }
        network_nodes = _network_nodes(state["tdg"])
        total = sum(macs[node_id] for node_id in network_nodes)
        names = ", ".join(network["name"] for network in state["spec"]["networks"])
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

    def _ops_problems(self, state: PipelineState) -> dict[str, list[str]]:
        costed = self._costed(state)
        report: ValidationReport = validate_tdg(costed, self._board_for(state["spec"]["platform"]))
        r9 = [str(issue) for issue in report.warnings if issue.rule == "R9"]
        problems = _group_by_owner(r9, state["tdg"])
        # The network check already sized the network's nodes from its published size.
        sized = set(_network_nodes(state["tdg"])) if _target_macs(state["spec"]) else set()
        for node_id, problem in _implausible_times(costed):
            if node_id not in sized:
                problems.setdefault(node_id, []).append(problem)
        for node_id, problem in _deep_loop_nests(state["tdg"]):
            problems.setdefault(node_id, []).append(problem)
        if not _target_macs(state["spec"]):
            for node_id, problem in _stated_total_problems(state["spec"], costed):
                problems.setdefault(node_id, []).append(problem)
        return problems

    def _opening(self, state: PipelineState, request: str) -> list[dict]:
        board = self._board_for(state["spec"]["platform"])
        return [
            {"role": "system", "content": _system_prompt(state["spec"], board, state["research"])},
            {"role": "user", "content": request},
        ]


def _payload_request(node: dict, edges: list[dict], spec: dict) -> str:
    connected = [edge for edge in edges if node["id"] in (edge["src"], edge["dst"])]
    return (
        Template(_PAYLOAD_REQUESTS[node["kind"]]).substitute(node)
        + "\n\nIts edges: "
        + (json.dumps(connected) if connected else "none")
        + (_network_note(spec) if node["kind"] == "template" else "")
    )


def _target_macs(spec: dict) -> float:
    return sum(network["gmacs"] for network in spec.get("networks") or []) * 1e9


def _network_nodes(tdg: dict) -> list[str]:
    return [
        node["id"]
        for node in tdg["nodes"]
        if node["kind"] == "template" and node["template"]["op"] in NETWORK_OPS
    ]


def _system_prompt(spec: dict, board: Board, notes: list[dict]) -> str:
    board_text = json.dumps(board.model_dump(), indent=1)
    notes_text = "\n".join(
        f"- {note['step']}: {note['method']}; {note['work']}"
        + (f" (source: {note['source']})" if note.get("source") else "")
        for note in notes
    )
    return Template(PROMPT_PATH.read_text(encoding="utf-8")).substitute(
        spec=json.dumps(spec, indent=1), board=board_text, notes=notes_text or "(none)"
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


def _deep_loop_nests(tdg: dict) -> list[tuple[str, str]]:
    """Loop nests with more loops than their largest array has dimensions + 1, and no sliding
    window: usually two steps in sequence written as nested loops (their counts multiply)."""
    found = []
    for node in tdg["nodes"]:
        if node["kind"] != "loop_nest":
            continue
        nest = node["loop_nest"]
        rank = max((len(array["shape"]) for array in nest["arrays"].values()), default=0)
        windowed = any(
            access.get("pattern") == "sliding_window" for access in nest.get("access", {}).values()
        )
        if windowed or len(nest["loops"]) <= rank + 1:
            continue
        ranges = " x ".join(str(loop["range"]) for loop in nest["loops"])
        found.append(
            (
                node["id"],
                f"node {node['id']}: has {len(nest['loops'])} nested loops ({ranges}) but its "
                f"largest array has only {rank} dimensions. Steps that run one after another "
                "must not be nested inside each other: nested loops multiply. Keep only the loops "
                "of the largest step and add the other steps' work to the body (their op count "
                "divided by this nest's iterations). If all the loops really belong to one "
                "computation, keep them.",
            )
        )
    return found


_TOTAL_NUMBER = re.compile(r"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([kMGT])?(?=[A-Za-z]*\b)")
_SCALE = {"k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}


def _stated_total(spec: dict) -> tuple[str, float] | None:
    """An assumption stating the total work, e.g. `total_flops: 8 * 2.147e9 = 1.718e10`."""
    for assumption in spec.get("assumptions") or []:
        key, value = assumption["key"].lower(), str(assumption["value"])
        if "total" not in key or not re.search(r"flop|ops|mac", key + value.lower()):
            continue
        numbers = _TOTAL_NUMBER.findall(value)
        if numbers:
            number, prefix = numbers[-1]  # the result of a formula comes last
            total = float(number) * _SCALE.get(prefix, 1)
            if total > 0:
                return f"{assumption['key']} = {value}", total
    return None


def _stated_total_problems(spec: dict, costed: dict) -> list[tuple[str, str]]:
    """The graph's ops must agree with a total the spec states; the largest node is told."""
    stated = _stated_total(spec)
    ops = {cost["id"]: cost.get("ops_estimated") or 0 for cost in costed["costs"]["nodes"]}
    if not stated or not any(ops.values()):
        return []
    text, target = stated
    total = sum(ops.values())
    ratio = total / target
    if 1 / STATED_TOTAL_FACTOR <= ratio <= STATED_TOTAL_FACTOR:
        return []
    largest = max(ops, key=ops.get)
    return [
        (
            largest,
            f"The spec states {text}, but the graph's nodes add up to {total:.3g} ops "
            f"({ratio:.2g}x). This node is the largest ({ops[largest]:.3g} ops): check its sizes "
            "against the spec (batch count, frames, channels, matrix sizes), per iteration.",
        )
    ]


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


def _assemble(state: PipelineState) -> dict:
    spec, nodes, payloads = state["spec"], state["nodes"], state["payloads"]
    tdg = {
        "graph": {
            "id": state["graph_id"],
            "schema_version": "0.2",
            "provenance": {"kind": "llm", "source": spec["task_summary"]},
        },
        "arrival": spec["arrival"],
        "objective": {"latency": 1.0},
        "platform": spec["platform"],
        "nodes": [
            {**{key: node[key] for key in NODE_FIELDS}, **payloads[node["id"]]} for node in nodes
        ],
        "edges": state["edges"],
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
