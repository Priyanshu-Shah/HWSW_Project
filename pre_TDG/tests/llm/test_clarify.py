import json

import pytest

from fake_ollama import FakeOllama, call, text
from hwsw.conversation import (
    AgentError,
    AssistantMessage,
    AssumptionsNotice,
    QuestionBatch,
    SpecReady,
    UserInput,
)
from hwsw.llm import ClarificationAgent
from hwsw.llm.clarify import MAX_QUESTION_ROUNDS
from hwsw.spec import validate_spec

BOARD_IDS = ["kv260", "pynq_z2", "zcu104"]

QUESTIONS = [
    {"id": "platform", "question": "Board?", "suggested_answer": "zcu104", "choices": BOARD_IDS},
    {"id": "fps", "question": "Frames per second?", "suggested_answer": "30"},
]
SPEC = {
    "task_summary": "Sobel edge detection on 1080p video",
    "platform": "kv260",
    "arrival": {"kind": "streaming", "period_s": 1 / 60},
    "timing": {"deadline_s": 1 / 60, "hardness": "hard", "derived_from": "60 fps, sequential (D3)"},
    "assumptions": [
        {"key": "fps", "value": "60", "source": "user", "confidence": "high"},
        {"key": "dtype", "value": "uint8", "source": "default", "confidence": "medium"},
    ],
    "open_questions": [],
}


def events_of_type(events, event_type):
    return [event for event in events if isinstance(event, event_type)]


def make_agent(fake: FakeOllama, boards_dir) -> ClarificationAgent:
    return ClarificationAgent(BOARD_IDS, client=fake.client(), boards_dir=boards_dir, tools=[])


def test_valid_spec_fixture():
    assert validate_spec(SPEC, ready_for_tdg_builder=True).is_valid


def test_questions_then_answers_then_spec(boards_dir):
    fake = FakeOllama(
        call("platform_lookup"),
        call("ask_user", message="Two questions:", questions=QUESTIONS),
        call("submit_spec", spec=SPEC),
    )
    agent = make_agent(fake, boards_dir)

    first = list(agent.handle(UserInput(text="Sobel edges on 1080p video")))
    assert [event.text for event in events_of_type(first, AssistantMessage)] == ["Two questions:"]
    (batch,) = events_of_type(first, QuestionBatch)
    assert [question.id for question in batch.questions] == ["platform", "fps"]

    second = list(agent.handle(UserInput(answers={"platform": "kv260", "fps": "30"})))
    assert events_of_type(second, SpecReady)[0].spec == SPEC
    assert len(events_of_type(second, AssumptionsNotice)[0].assumptions) == 2
    assert agent.spec == SPEC

    answers = json.loads(fake.requests[2]["messages"][-1]["content"])
    assert answers == {"platform": "kv260", "fps": "30 (accepted the suggestion)"}
    board_list = fake.requests[1]["messages"][-1]["content"]  # platform_lookup with no id
    assert "zcu104: Zynq UltraScale+" in board_list


def test_invalid_spec_is_sent_back_and_fixed(boards_dir):
    unknown_board = {**SPEC, "platform": "zcu999"}
    no_deadline = {**SPEC, "timing": {"hardness": "hard", "derived_from": "?"}}
    fake = FakeOllama(
        call("submit_spec", spec=unknown_board),
        call("submit_spec", spec=no_deadline),
        call("submit_spec", spec=SPEC),
    )
    events = list(make_agent(fake, boards_dir).handle(UserInput(text="Sobel, kv260, 60 fps")))
    assert events_of_type(events, SpecReady)
    feedback = [request["messages"][-1]["content"] for request in fake.requests[1:]]
    assert "'zcu999' is not a known board" in feedback[0]
    assert "deadline_s" in feedback[1]


def test_gives_up_after_three_invalid_specs(boards_dir):
    fake = FakeOllama(*[call("submit_spec", spec={"platform": "kv260"})] * 3)
    events = list(make_agent(fake, boards_dir).handle(UserInput(text="something")))
    (failure,) = events_of_type(events, AgentError)
    assert "no valid spec after 3 attempts" in failure.message


def test_question_rounds_are_capped(boards_dir):
    ask = call("ask_user", questions=QUESTIONS)
    fake = FakeOllama(*[ask] * (MAX_QUESTION_ROUNDS + 1), call("submit_spec", spec=SPEC))
    agent = make_agent(fake, boards_dir)
    events = list(agent.handle(UserInput(text="Sobel")))
    for _ in range(MAX_QUESTION_ROUNDS - 1):
        events = list(agent.handle(UserInput(answers={})))
        assert events_of_type(events, QuestionBatch)
    events = list(agent.handle(UserInput(answers={})))
    assert events_of_type(events, SpecReady)
    assert "question limit is reached" in fake.requests[-1]["messages"][-1]["content"]


def test_unreachable_model_is_reported_as_an_error_event(boards_dir):
    events = list(make_agent(FakeOllama(), boards_dir).handle(UserInput(text="Sobel")))
    assert events_of_type(events, AgentError)


@pytest.mark.parametrize("finished", [False, True])
def test_empty_input_or_finished_conversation_gets_a_message(boards_dir, finished):
    agent = make_agent(FakeOllama(), boards_dir)
    agent.spec = SPEC if finished else None
    events = list(agent.handle(UserInput(text="")))
    assert [type(event) for event in events] == [AssistantMessage]


def test_plain_text_reply_is_not_shown_to_the_user(boards_dir):
    fake = FakeOllama(text("Sure, let me think."), call("submit_spec", spec=SPEC))
    events = list(make_agent(fake, boards_dir).handle(UserInput(text="Sobel")))
    assert [type(event) for event in events] == [AssumptionsNotice, SpecReady]
