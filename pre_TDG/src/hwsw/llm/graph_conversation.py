"""Runs a LangGraph agent graph as a `ConversationAgent`: one `handle` call per user turn.

Agents send events to the user through LangGraph's stream writer. The `ask_user` step pauses the
graph (an interrupt); the next `handle` call resumes it with the answers. If an agent fails, the
graph keeps its last finished step, and the next `handle` call runs it again from there.
"""

from collections.abc import Iterator

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.types import Command

from hwsw.conversation import AgentError, AssistantMessage, ConversationEvent, UserInput
from hwsw.llm.client import LlmError

# A graph step is one agent turn; a 15-node graph with repairs needs well over LangGraph's 25.
RECURSION_LIMIT = 500


class GraphConversation:
    finished_text = "This conversation is finished. Start a new one."

    def __init__(self, graph: StateGraph):
        self._graph = graph.compile(checkpointer=InMemorySaver())
        self._config = {
            "configurable": {"thread_id": "conversation"},
            "recursion_limit": RECURSION_LIMIT,
        }

    def handle(self, user_input: UserInput) -> Iterator[ConversationEvent]:
        snapshot = self._graph.get_state(self._config)
        if snapshot.interrupts:
            command = Command(resume={"answers": user_input.answers, "text": user_input.text})
        elif snapshot.next:  # an agent failed last time: run it again
            command = None
        elif self.finished():
            yield AssistantMessage(text=self.finished_text)
            return
        elif not user_input.text.strip():
            yield AssistantMessage(text="Describe the task you want to partition.")
            return
        else:
            command = self.start(user_input.text)
        try:
            for event in self._graph.stream(command, self._config, stream_mode="custom"):
                yield from self.on_event(event)
        except LlmError as failure:
            yield AgentError(message=f"{failure} (send any message to try again)")
            return
        snapshot = self._graph.get_state(self._config)
        if not snapshot.next:  # the graph reached its end
            yield from self.after_turn(snapshot.values)

    def start(self, text: str) -> dict:
        raise NotImplementedError

    def finished(self) -> bool:
        raise NotImplementedError

    def on_event(self, event: ConversationEvent) -> Iterator[ConversationEvent]:
        yield event

    def after_turn(self, state: dict) -> Iterator[ConversationEvent]:
        """Called once the graph has reached its end, with the final state."""
        yield from ()
