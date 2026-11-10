"""The shared state of the agent graph (D6): every agent reads it and returns the keys it changes.

Only plain JSON-like values, so LangGraph can checkpoint the state between conversation turns.
"""

from typing import TypedDict


class PipelineState(TypedDict, total=False):
    # Stage A: clarification
    clarify_messages: list[dict]  # the clarifier's conversation with the model
    questions: list[dict]  # waiting for the user's answers (Question fields)
    question_rounds: int
    spec: dict
    # Stage B: graph building
    research: list[dict]  # work notes: per computation step, the method and its work per item
    graph_id: str
    nodes: list[dict]  # the decomposer's node list (with `purpose`)
    edges: list[dict]
    payloads: dict[str, dict]  # node id -> opaque / template / loop_nest payload
    chats: dict[str, list[dict]]  # "nodes", "edges" or a node id -> that agent's conversation
    todo: list[str]  # node ids still without a payload
    fixes: dict[str, list[str]]  # "edges" or a node id -> problems its agent must fix
    repairs: int  # validate -> repair rounds since the graph was last valid
    checks: dict[str, int]  # check name -> 1 after its one fix, 2 when done
    tdg: dict
    warnings: list[str]
