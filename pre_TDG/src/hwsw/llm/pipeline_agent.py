"""The whole LLM conversation: stage A (questions -> spec), then stage B (spec -> task graph).

Implements `ConversationAgent`. Each conversation gets its own run folder with the LLM log,
`spec.json` and `tdg.json`.
"""

from collections.abc import Iterator
from pathlib import Path

from hwsw.conversation import (
    AgentError,
    AssistantMessage,
    ConversationEvent,
    TdgReady,
    UserInput,
)
from hwsw.json_files import write_json
from hwsw.llm.clarify import ClarificationAgent
from hwsw.llm.client import LlmError, OllamaClient
from hwsw.llm.run_log import LOG_FILE_NAME, RunLog, new_run_dir
from hwsw.llm.tdg_builder import TdgBuilder
from hwsw.platforms import load_board
from hwsw.repo_paths import LIBRARY_BOARDS_DIR


class PipelineAgent:
    def __init__(
        self,
        board_ids: list[str],
        *,
        boards_dir: Path = LIBRARY_BOARDS_DIR,
        client: OllamaClient | None = None,
        run_dir: Path | None = None,
        **clarify_options,  # passed to ClarificationAgent (tests replace its tools)
    ):
        self.run_dir = run_dir or new_run_dir("chat")
        self._client = client or OllamaClient(log=RunLog(self.run_dir / LOG_FILE_NAME))
        self._boards_dir = boards_dir
        self._clarify = ClarificationAgent(
            board_ids, client=self._client, boards_dir=boards_dir, **clarify_options
        )
        self.tdg: dict | None = None

    @property
    def spec(self) -> dict | None:
        return self._clarify.spec

    def handle(self, user_input: UserInput) -> Iterator[ConversationEvent]:
        if self.tdg is not None:
            yield AssistantMessage(text="The task graph is done. Start a new conversation.")
            return
        if self._clarify.spec is None:  # otherwise stage B failed before: try it again
            yield from self._clarify.handle(user_input)
            if self._clarify.spec is None:
                return
            write_json(self.run_dir / "spec.json", self._clarify.spec)
        yield from self._build_graph(self._clarify.spec)

    def _build_graph(self, spec: dict) -> Iterator[ConversationEvent]:
        yield AssistantMessage(text="Building the task graph; this takes a few minutes.")
        try:
            board = load_board(spec["platform"], self._boards_dir)
            result = TdgBuilder(self._client, board).build(spec)
        except LlmError as failure:
            yield AgentError(message=f"{failure} (send any message to try again)")
            return
        for warning in result.warnings:
            yield AssistantMessage(text=f"Check this node: {warning}")
        self.tdg = result.tdg
        write_json(self.run_dir / "tdg.json", self.tdg)
        yield TdgReady(tdg=self.tdg)
