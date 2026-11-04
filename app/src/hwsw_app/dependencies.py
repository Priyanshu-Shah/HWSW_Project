"""FastAPI dependencies: the objects created once in `create_app`."""

from fastapi import Request

from hwsw_app.chat_sessions import ChatSessions
from hwsw_app.pipeline_service import PipelineService


def get_service(request: Request) -> PipelineService:
    return request.app.state.service


def get_sessions(request: Request) -> ChatSessions:
    return request.app.state.sessions
