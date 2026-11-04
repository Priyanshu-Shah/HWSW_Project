"""The app: `create_app()` builds it, `hwsw-app` (or `python -m hwsw_app`) serves it."""

from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from hwsw.conversation import AgentFactory, ScriptedAgent

from hwsw_app.chat_sessions import ChatSessions
from hwsw_app.pipeline_service import PipelineService
from hwsw_app.routes import chat_routes, tdg_routes
from hwsw_app.settings import AppSettings, resolve_settings

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    settings: AppSettings | None = None, agent_factory: AgentFactory = ScriptedAgent
) -> FastAPI:
    """`agent_factory` is the one place to plug in the real LLM agent."""
    settings = settings or resolve_settings()
    app = FastAPI(title="HW/SW partitioning: pipeline front end")
    app.state.settings = settings
    app.state.service = PipelineService(settings)
    app.state.sessions = ChatSessions(agent_factory)

    app.include_router(tdg_routes.router)
    app.include_router(chat_routes.router)
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")  # keep last
    return app


def run(
    host: Annotated[
        str, typer.Option(help="Use 0.0.0.0 to reach it from another machine.")
    ] = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Serve the web page and its API."""
    settings = resolve_settings()
    typer.echo(f"boards: {settings.boards_source} ({settings.boards_dir})")
    typer.echo(f"open http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}")
    uvicorn.run(create_app(settings), host=host, port=port)


def cli() -> None:
    """Entry point of the `hwsw-app` command: parses --host and --port, then runs."""
    typer.run(run)
