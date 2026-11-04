"""In-memory store of running conversations (lost on restart; fine for a lab tool)."""

import uuid

from hwsw.conversation import AgentFactory, ConversationAgent


class UnknownSessionError(KeyError):
    pass


class ChatSessions:
    def __init__(self, agent_factory: AgentFactory):
        self._agent_factory = agent_factory
        self._agents: dict[str, ConversationAgent] = {}

    def create(self, board_ids: list[str]) -> str:
        session_id = uuid.uuid4().hex
        self._agents[session_id] = self._agent_factory(board_ids)
        return session_id

    def get(self, session_id: str) -> ConversationAgent:
        try:
            return self._agents[session_id]
        except KeyError as unknown:
            raise UnknownSessionError(session_id) from unknown
