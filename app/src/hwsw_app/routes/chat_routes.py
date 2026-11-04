"""Conversations with the agent. Replies are streamed as server-sent events."""

from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from hwsw.conversation import AgentError, ConversationAgent, ConversationEvent, UserInput

from hwsw_app.chat_sessions import ChatSessions, UnknownSessionError
from hwsw_app.dependencies import get_service, get_sessions
from hwsw_app.pipeline_service import PipelineService
from hwsw_app.schemas import SessionCreated

router = APIRouter(prefix="/api/chat")

END_OF_REPLY = "event: end\ndata: {}\n\n"


@router.post("/sessions")
def create_session(
    sessions: Annotated[ChatSessions, Depends(get_sessions)],
    service: Annotated[PipelineService, Depends(get_service)],
) -> SessionCreated:
    return SessionCreated(session_id=sessions.create(service.board_ids()))


@router.post("/sessions/{session_id}/messages")
def send_message(
    session_id: str,
    user_input: UserInput,
    sessions: Annotated[ChatSessions, Depends(get_sessions)],
) -> StreamingResponse:
    try:
        agent = sessions.get(session_id)
    except UnknownSessionError as unknown:
        raise HTTPException(status_code=404, detail=f"unknown session '{session_id}'") from unknown
    return StreamingResponse(_server_sent_events(agent, user_input), media_type="text/event-stream")


def _server_sent_events(agent: ConversationAgent, user_input: UserInput) -> Iterator[str]:
    """One SSE message per event, then `end`. An agent failure becomes an `error` event."""
    try:
        for event in agent.handle(user_input):
            yield _format(event)
    except Exception as failure:  # noqa: BLE001 - the client must always get an answer
        yield _format(AgentError(message=f"{type(failure).__name__}: {failure}"))
    yield END_OF_REPLY


def _format(event: ConversationEvent) -> str:
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"
