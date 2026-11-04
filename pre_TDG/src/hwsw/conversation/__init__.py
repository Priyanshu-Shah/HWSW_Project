"""The user-facing conversation: events the agent emits, the agent interface, and the stub agent."""

from hwsw.conversation.agent_protocol import AgentFactory, ConversationAgent
from hwsw.conversation.events import (
    AgentError,
    AssistantMessage,
    Assumption,
    AssumptionsNotice,
    ConversationEvent,
    Question,
    QuestionBatch,
    SpecReady,
    TdgReady,
    UserInput,
)
from hwsw.conversation.example_graphs import list_example_names, load_example_graph
from hwsw.conversation.scripted_agent import ScriptedAgent

__all__ = [
    "AgentError",
    "AgentFactory",
    "AssistantMessage",
    "Assumption",
    "AssumptionsNotice",
    "ConversationAgent",
    "ConversationEvent",
    "Question",
    "QuestionBatch",
    "ScriptedAgent",
    "SpecReady",
    "TdgReady",
    "UserInput",
    "list_example_names",
    "load_example_graph",
]
