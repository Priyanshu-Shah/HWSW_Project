"""Stage A: the clarifier agent. Talks with the user until the task spec is complete.

Two graph nodes (D6): `clarifier` runs the model until it asks the user something or submits a
valid spec; `ask_user` pauses the graph until the answers come, then hands them back.
`ClarificationAgent` runs just these two as a `ConversationAgent`.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from string import Template

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from hwsw.conversation import (
    AssistantMessage,
    Assumption,
    AssumptionsNotice,
    ConversationEvent,
    Question,
    QuestionBatch,
    SpecReady,
)
from hwsw.llm.client import LlmError, OllamaClient
from hwsw.llm.graph_conversation import GraphConversation
from hwsw.llm.graph_state import PipelineState
from hwsw.llm.tool_loop import run_tool_loop, tool_result
from hwsw.llm.tools import (
    Tool,
    board_list_text,
    fetch_page_tool,
    object_schema,
    platform_lookup_tool,
    web_search_tool,
)
from hwsw.repo_paths import LIBRARY_BOARDS_DIR
from hwsw.spec import load_spec_schema, validate_spec

PROMPT_PATH = Path(__file__).parent / "prompts" / "clarify.md"
MAX_QUESTION_ROUNDS = 3
MAX_QUESTIONS_PER_ROUND = 5
MAX_SPEC_ATTEMPTS = 3

ASK_USER = Tool(
    name="ask_user",
    description=(
        f"Ask the user up to {MAX_QUESTIONS_PER_ROUND} questions at once. Each has a short id, "
        "the question text, a suggested answer, and optional choices. Returns the answers."
    ),
    parameters=object_schema(
        {
            "message": {"type": "string", "description": "A short note shown before the questions"},
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_QUESTIONS_PER_ROUND,
                "items": object_schema(
                    {
                        "id": {"type": "string"},
                        "question": {"type": "string"},  # the name gpt-oss uses unprompted
                        "suggested_answer": {"type": "string"},
                        "choices": {"type": "array", "items": {"type": "string"}},
                    },
                    required=["id", "question", "suggested_answer"],
                ),
            },
        },
        required=["questions"],
    ),
)
SUBMIT_SPEC = Tool(
    name="submit_spec",
    description="Submit the finished task spec. Returns errors to fix if it is not valid.",
    parameters=object_schema({"spec": {"type": "object"}}),
)


def add_clarifier(
    graph: StateGraph,
    board_ids: list[str],
    client: OllamaClient,
    *,
    boards_dir: Path,
    tools: list[Tool],
    then: str = END,
) -> None:
    """Adds `clarifier` and `ask_user`, from START; once the spec is valid it goes to `then`."""
    agent = _Clarifier(board_ids, client, [platform_lookup_tool(boards_dir), *tools])
    graph.add_node("clarifier", agent.run)
    graph.add_node("ask_user", _ask_user)
    graph.add_edge(START, "clarifier")
    graph.add_conditional_edges(
        "clarifier",
        lambda state: "ask_user" if state.get("questions") else then,
        ["ask_user", then],
    )
    graph.add_edge("ask_user", "clarifier")


def first_messages(boards_dir: Path, task: str) -> list[dict]:
    return [
        {"role": "system", "content": _system_prompt(boards_dir)},
        {"role": "user", "content": task},
    ]


def default_lookup_tools() -> list[Tool]:
    return [web_search_tool(), fetch_page_tool()]


class ClarificationAgent(GraphConversation):
    finished_text = "The task spec is complete. Start a new conversation."

    def __init__(
        self,
        board_ids: list[str],
        *,
        client: OllamaClient | None = None,
        boards_dir: Path = LIBRARY_BOARDS_DIR,
        tools: list[Tool] | None = None,  # tests replace the network tools
    ):
        graph = StateGraph(PipelineState)
        tools = default_lookup_tools() if tools is None else tools
        add_clarifier(
            graph, board_ids, client or OllamaClient(), boards_dir=boards_dir, tools=tools
        )
        super().__init__(graph)
        self._boards_dir = boards_dir
        self.spec: dict | None = None

    def start(self, text: str) -> dict:
        return {"clarify_messages": first_messages(self._boards_dir, text)}

    def finished(self) -> bool:
        return self.spec is not None

    def on_event(self, event: ConversationEvent) -> Iterator[ConversationEvent]:
        if isinstance(event, SpecReady):
            self.spec = event.spec
        yield event


class _Clarifier:
    def __init__(self, board_ids: list[str], client: OllamaClient, tools: list[Tool]):
        self._board_ids = board_ids
        self._client = client
        self._tools = [*tools, ASK_USER, SUBMIT_SPEC]

    def run(self, state: PipelineState) -> dict:
        messages = list(state["clarify_messages"])  # a copy: a failed run leaves the state as is
        rounds = state.get("question_rounds", 0)
        write = get_stream_writer()
        spec_attempts = 0
        while True:
            call = run_tool_loop(self._client, messages, self._tools)
            if call.name == ASK_USER.name:
                if rounds < MAX_QUESTION_ROUNDS:
                    questions = _questions(call.arguments)
                    if call.arguments.get("message"):
                        write(AssistantMessage(text=call.arguments["message"]))
                    write(QuestionBatch(questions=questions))
                    return {
                        "clarify_messages": messages,
                        "questions": [question.model_dump() for question in questions],
                        "question_rounds": rounds + 1,
                    }
                messages.append(tool_result(call.name, _QUESTION_LIMIT_REACHED))
                continue
            spec_attempts += 1
            spec = call.arguments.get("spec")
            problems = self._spec_problems(spec)
            if not problems:
                messages.append(tool_result(SUBMIT_SPEC.name, "accepted"))
                assumptions = [Assumption(**item) for item in spec["assumptions"]]
                write(AssumptionsNotice(assumptions=assumptions))
                write(SpecReady(spec=spec))
                return {"clarify_messages": messages, "spec": spec, "questions": []}
            if spec_attempts >= MAX_SPEC_ATTEMPTS:
                raise LlmError("no valid spec after 3 attempts: " + "; ".join(problems))
            listed = "\n".join(f"- {problem}" for problem in problems)
            messages.append(tool_result(SUBMIT_SPEC.name, f"not valid, fix:\n{listed}"))

    def _spec_problems(self, spec: object) -> list[str]:
        if not isinstance(spec, dict):
            return ["`spec` must be a JSON object"]
        problems = [str(issue) for issue in validate_spec(spec, ready_for_tdg_builder=True).errors]
        if not problems and spec["platform"] not in self._board_ids:
            problems.append(
                f"platform '{spec['platform']}' is not a known board; use one of "
                + ", ".join(self._board_ids)
            )
        return problems


def _ask_user(state: PipelineState) -> dict:
    """Pauses the graph; resumed with {"answers": {question id: answer}, "text": a note}."""
    reply = interrupt(state["questions"])
    answers = {}
    for question in state["questions"]:
        answer = reply["answers"].get(question["id"], "").strip()
        if not answer or answer == question["suggested_answer"]:
            answers[question["id"]] = f"{question['suggested_answer']} (accepted the suggestion)"
        else:
            answers[question["id"]] = answer
    if reply["text"].strip():
        answers["user_note"] = reply["text"]
    result = tool_result(ASK_USER.name, json.dumps(answers, indent=1))
    return {"clarify_messages": [*state["clarify_messages"], result], "questions": []}


def _questions(arguments: dict) -> list[Question]:
    return [
        Question(
            id=item["id"],
            text=item["question"],
            suggested_answer=item["suggested_answer"],
            choices=item.get("choices", []),
        )
        for item in arguments["questions"]
    ]


_QUESTION_LIMIT_REACHED = (
    "The question limit is reached; do not ask again. Use the suggested defaults for anything "
    "still open (as assumptions with source 'default') and call submit_spec."
)


def _system_prompt(boards_dir: Path) -> str:
    boards = "\n".join(f"   - {line}" for line in board_list_text(boards_dir).splitlines())
    return Template(PROMPT_PATH.read_text(encoding="utf-8")).substitute(
        boards=boards,
        max_rounds=MAX_QUESTION_ROUNDS,
        spec_schema=json.dumps(load_spec_schema(), indent=1),
    )
