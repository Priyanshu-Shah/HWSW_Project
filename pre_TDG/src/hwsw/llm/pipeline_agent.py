"""The whole LLM conversation as one agent graph (D6): stage A (clarifier, ask_user -> spec), then
stage B (researcher, decomposer, edge and payload agents, checkers -> task graph).

Implements `ConversationAgent`. Each conversation gets its own run folder with the LLM log,
`spec.json` and `tdg.json`. `agent_graph()` gives the same graph for drawing it.
"""

from collections.abc import Iterator
from pathlib import Path

from langgraph.graph import StateGraph

from hwsw.conversation import AssistantMessage, ConversationEvent, SpecReady, TdgReady
from hwsw.json_files import write_json
from hwsw.llm.clarify import add_clarifier, default_lookup_tools, first_messages
from hwsw.llm.client import OllamaClient
from hwsw.llm.graph_conversation import GraphConversation
from hwsw.llm.graph_state import PipelineState
from hwsw.llm.run_log import LOG_FILE_NAME, RunLog, new_run_dir
from hwsw.llm.tdg_builder import add_tdg_builder
from hwsw.llm.tools import Tool
from hwsw.platforms import load_board
from hwsw.repo_paths import LIBRARY_BOARDS_DIR


def agent_graph(
    board_ids: list[str], client: OllamaClient, boards_dir: Path, tools: list[Tool]
) -> StateGraph:
    graph = StateGraph(PipelineState)
    add_clarifier(graph, board_ids, client, boards_dir=boards_dir, tools=tools, then="researcher")
    add_tdg_builder(
        graph,
        client,
        lambda board_id: load_board(board_id, boards_dir),
        None,
        tools,
    )
    return graph


class PipelineAgent(GraphConversation):
    finished_text = "The task graph is done. Start a new conversation."

    def __init__(
        self,
        board_ids: list[str],
        *,
        boards_dir: Path = LIBRARY_BOARDS_DIR,
        client: OllamaClient | None = None,
        run_dir: Path | None = None,
        tools: list[Tool] | None = None,  # tests replace the network tools
    ):
        self.run_dir = run_dir or new_run_dir("chat")
        client = client or OllamaClient(log=RunLog(self.run_dir / LOG_FILE_NAME))
        tools = default_lookup_tools() if tools is None else tools
        super().__init__(agent_graph(board_ids, client, boards_dir, tools))
        self._boards_dir = boards_dir
        self.spec: dict | None = None
        self.tdg: dict | None = None

    def start(self, text: str) -> dict:
        return {"clarify_messages": first_messages(self._boards_dir, text)}

    def finished(self) -> bool:
        return self.tdg is not None

    def on_event(self, event: ConversationEvent) -> Iterator[ConversationEvent]:
        if isinstance(event, SpecReady):
            self.spec = event.spec
            write_json(self.run_dir / "spec.json", self.spec)
        yield event

    def after_turn(self, state: dict) -> Iterator[ConversationEvent]:
        if self.tdg is not None:
            return
        for warning in state["warnings"]:
            yield AssistantMessage(text=f"Check this node: {warning}")
        self.tdg = state["tdg"]
        write_json(self.run_dir / "tdg.json", self.tdg)
        yield TdgReady(tdg=self.tdg)
