"""The interface every conversation agent implements (the scripted stub now, the LLM later)."""

from collections.abc import Callable, Iterator
from typing import Protocol

from hwsw.conversation.events import ConversationEvent, UserInput


class ConversationAgent(Protocol):
    """One conversation. Holds its own state between turns.

    `handle` yields events as they are produced, so a front end can show a long answer
    (for example LLM tokens) while it is still being written.
    """

    def handle(self, user_input: UserInput) -> Iterator[ConversationEvent]: ...


# Builds a fresh agent for a new conversation, given the ids of the boards the user may pick.
AgentFactory = Callable[[list[str]], ConversationAgent]
