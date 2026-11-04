"""A stand-in for the clarification agent: fixed questions, then an example TDG. No LLM.

It exists so the front end and the rest of the pipeline can be built and tested before the LLM is
set up. It never invents sources: every assumption it reports comes from the user or is marked
as a default.

The answers only change the platform and the timing of the example graph; the graph itself is
one of the ready-made examples, picked by keywords in the task text.
"""

from collections.abc import Iterator

from hwsw.conversation.events import (
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
from hwsw.conversation.example_graphs import load_example_graph

_PREFERRED_BOARD = "zcu104"
_KEYWORDS_TO_EXAMPLE = (
    (("sobel", "edge"), "sobel_edge"),
    (("matrix", "matrices", "gemm", "matmul"), "matmul_batch"),
)
_DEFAULT_EXAMPLE = "det_pipeline"


class ScriptedAgent:
    def __init__(self, board_ids: list[str]):
        self._board_ids = board_ids
        self._task_text = ""
        self._stage = "waiting_for_task"

    def handle(self, user_input: UserInput) -> Iterator[ConversationEvent]:
        if self._stage == "waiting_for_task":
            yield from self._ask_questions(user_input.text)
        elif self._stage == "waiting_for_answers":
            yield from self._build_graph(user_input.answers)
        else:
            yield AssistantMessage(
                text="This is the scripted stand-in agent and it has finished. "
                "Start a new conversation to try again."
            )

    def _ask_questions(self, task_text: str) -> Iterator[ConversationEvent]:
        if not task_text.strip():
            yield AssistantMessage(text="Describe the task you want to partition.")
            return
        self._task_text = task_text
        self._stage = "waiting_for_answers"
        yield AssistantMessage(
            text="(Scripted stand-in for the LLM agent.) A few questions before I build the graph:"
        )
        yield QuestionBatch(questions=self._questions())

    def _questions(self) -> list[Question]:
        suggested_board = (
            _PREFERRED_BOARD if _PREFERRED_BOARD in self._board_ids else self._board_ids[0]
        )
        return [
            Question(
                id="platform",
                text="Which board will it run on?",
                suggested_answer=suggested_board,
                choices=self._board_ids,
            ),
            Question(
                id="rate_hz",
                text="How many iterations per second must it sustain?",
                suggested_answer="30",
            ),
            Question(
                id="resolution", text="Input size (width x height)?", suggested_answer="1280x720"
            ),
            Question(
                id="hard_real_time",
                text="Must every iteration meet the deadline?",
                suggested_answer="yes",
                choices=["yes", "no"],
            ),
        ]

    def _build_graph(self, user_answers: dict[str, str]) -> Iterator[ConversationEvent]:
        self._stage = "done"
        questions = {question.id: question for question in self._questions()}
        answers = {qid: user_answers.get(qid) or q.suggested_answer for qid, q in questions.items()}

        assumptions = [
            Assumption(
                key=qid,
                value=answer,
                source="user" if user_answers.get(qid) else "default",
                confidence="high" if user_answers.get(qid) else "low",
            )
            for qid, answer in answers.items()
        ]
        yield AssumptionsNotice(assumptions=assumptions)

        spec = self._spec(answers, assumptions)
        yield SpecReady(spec=spec)

        yield AssistantMessage(
            text="Here is the task graph. Use 'Cost it' to estimate each task's cost."
        )
        yield TdgReady(tdg=self._graph(spec))

    def _spec(self, answers: dict[str, str], assumptions: list[Assumption]) -> dict:
        period_s = 1 / float(answers["rate_hz"])
        hard = answers["hard_real_time"] == "yes"
        return {
            "task_summary": self._task_text.strip(),
            "platform": answers["platform"],
            "arrival": {"kind": "streaming", "period_s": period_s},
            "timing": {
                "hardness": "hard" if hard else "soft",
                "deadline_s": period_s,
                "derived_from": f"{answers['rate_hz']} per second, sequential (D3)",
            },
            "assumptions": [assumption.model_dump() for assumption in assumptions],
            "open_questions": [],
        }

    def _graph(self, spec: dict) -> dict:
        graph = load_example_graph(self._example_name())
        graph["platform"] = spec["platform"]
        graph["graph"]["provenance"] = {"kind": "synthetic", "source": "scripted stand-in agent"}
        if graph["arrival"]["kind"] != "one_shot":
            graph["arrival"] = spec["arrival"]
            graph["constraints"] = {"deadline_s": spec["timing"]["deadline_s"]}
        return graph

    def _example_name(self) -> str:
        text = self._task_text.lower()
        for keywords, example in _KEYWORDS_TO_EXAMPLE:
            if any(keyword in text for keyword in keywords):
                return example
        return _DEFAULT_EXAMPLE
