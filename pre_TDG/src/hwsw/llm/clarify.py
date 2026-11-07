"""Stage A: the clarification agent. Talks with the user until the task spec is complete.

Implements `ConversationAgent`: each `handle` call runs the model until it either asks the user
something (a question batch; the next `handle` brings the answers) or submits a valid spec.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from string import Template

from hwsw.conversation import (
    AgentError,
    AssistantMessage,
    Assumption,
    AssumptionsNotice,
    ConversationEvent,
    Question,
    QuestionBatch,
    SpecReady,
    UserInput,
)
from hwsw.llm.client import LlmError, OllamaClient
from hwsw.llm.tool_loop import FinalCall, run_tool_loop, tool_result
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


class ClarificationAgent:
    def __init__(
        self,
        board_ids: list[str],
        *,
        client: OllamaClient | None = None,
        boards_dir: Path = LIBRARY_BOARDS_DIR,
        tools: list[Tool] | None = None,  # tests replace the network tools
    ):
        self._board_ids = board_ids
        self._client = client or OllamaClient()
        lookup_tools = tools if tools is not None else [web_search_tool(), fetch_page_tool()]
        self._tools = [platform_lookup_tool(boards_dir), *lookup_tools, ASK_USER, SUBMIT_SPEC]
        self._messages = [{"role": "system", "content": _system_prompt(boards_dir)}]
        self._pending_questions: list[Question] = []
        self._question_rounds = 0
        self._spec_attempts = 0
        self.spec: dict | None = None

    def handle(self, user_input: UserInput) -> Iterator[ConversationEvent]:
        if self.spec is not None:
            yield AssistantMessage(text="The task spec is complete. Start a new conversation.")
            return
        if self._pending_questions:
            self._messages.append(tool_result(ASK_USER.name, self._answers_text(user_input)))
            self._pending_questions = []
        elif user_input.text.strip():
            self._messages.append({"role": "user", "content": user_input.text})
        else:
            yield AssistantMessage(text="Describe the task you want to partition.")
            return
        try:
            yield from self._run_until_user_needed()
        except LlmError as failure:
            yield AgentError(message=str(failure))

    def _run_until_user_needed(self) -> Iterator[ConversationEvent]:
        while True:
            call = run_tool_loop(self._client, self._messages, self._tools)
            if call.name == ASK_USER.name:
                if self._question_rounds < MAX_QUESTION_ROUNDS:
                    yield from self._ask(call)
                    return
                self._messages.append(tool_result(call.name, _QUESTION_LIMIT_REACHED))
            elif self._accept_spec(call.arguments.get("spec")):
                assumptions = [Assumption(**item) for item in self.spec["assumptions"]]
                yield AssumptionsNotice(assumptions=assumptions)
                yield SpecReady(spec=self.spec)
                return

    def _ask(self, call: FinalCall) -> Iterator[ConversationEvent]:
        self._question_rounds += 1
        self._pending_questions = [
            Question(
                id=item["id"],
                text=item["question"],
                suggested_answer=item["suggested_answer"],
                choices=item.get("choices", []),
            )
            for item in call.arguments["questions"]
        ]
        if call.arguments.get("message"):
            yield AssistantMessage(text=call.arguments["message"])
        yield QuestionBatch(questions=self._pending_questions)

    def _answers_text(self, user_input: UserInput) -> str:
        answers = {}
        for question in self._pending_questions:
            answer = user_input.answers.get(question.id, "").strip()
            if not answer or answer == question.suggested_answer:
                answers[question.id] = f"{question.suggested_answer} (accepted the suggestion)"
            else:
                answers[question.id] = answer
        if user_input.text.strip():
            answers["user_note"] = user_input.text
        return json.dumps(answers, indent=1)

    def _accept_spec(self, spec: object) -> bool:
        """Keeps a valid spec; otherwise sends the errors back (or gives up after a few tries)."""
        self._spec_attempts += 1
        problems = self._spec_problems(spec)
        if not problems:
            self.spec = spec
            self._messages.append(tool_result(SUBMIT_SPEC.name, "accepted"))
            return True
        if self._spec_attempts >= MAX_SPEC_ATTEMPTS:
            raise LlmError("no valid spec after 3 attempts: " + "; ".join(problems))
        listed = "\n".join(f"- {problem}" for problem in problems)
        self._messages.append(tool_result(SUBMIT_SPEC.name, f"not valid, fix:\n{listed}"))
        return False

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
