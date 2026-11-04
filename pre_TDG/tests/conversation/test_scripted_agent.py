import pytest
from pydantic import TypeAdapter

from hwsw.conversation import (
    AssistantMessage,
    AssumptionsNotice,
    ConversationEvent,
    QuestionBatch,
    ScriptedAgent,
    SpecReady,
    TdgReady,
    UserInput,
    list_example_names,
    load_example_graph,
)
from hwsw.custom_hls import estimate_costs
from hwsw.platforms import load_board
from hwsw.spec import validate_spec
from hwsw.validation import validate_tdg

BOARD_IDS = ["kv260", "pynq_z2", "zcu104"]


def events_of_type(events, event_type):
    return [event for event in events if isinstance(event, event_type)]


@pytest.fixture
def agent() -> ScriptedAgent:
    return ScriptedAgent(BOARD_IDS)


def test_first_task_is_answered_with_a_batch_of_questions(agent):
    events = list(agent.handle(UserInput(text="Detect objects in video")))
    (batch,) = events_of_type(events, QuestionBatch)
    assert {question.id for question in batch.questions} == {
        "platform", "rate_hz", "resolution", "hard_real_time",
    }  # fmt: skip
    platform = next(question for question in batch.questions if question.id == "platform")
    assert platform.choices == BOARD_IDS
    assert platform.suggested_answer == "zcu104"
    assert all(question.suggested_answer for question in batch.questions)


def test_empty_task_asks_again_and_a_real_task_then_starts_the_questions(agent):
    (message,) = agent.handle(UserInput(text="  "))
    assert isinstance(message, AssistantMessage)
    events = list(agent.handle(UserInput(text="Detect objects in video")))
    assert events_of_type(events, QuestionBatch)


def test_answers_produce_a_valid_spec_and_graph(agent, boards_dir):
    list(agent.handle(UserInput(text="Detect objects in video")))
    answers = {"platform": "kv260", "rate_hz": "25", "hard_real_time": "yes"}
    events = list(agent.handle(UserInput(answers=answers)))

    (spec_event,) = events_of_type(events, SpecReady)
    report = validate_spec(spec_event.spec, ready_for_tdg_builder=True)
    assert report.is_valid, report.format()
    assert spec_event.spec["platform"] == "kv260"
    assert spec_event.spec["arrival"]["period_s"] == pytest.approx(1 / 25)

    (tdg_event,) = events_of_type(events, TdgReady)
    board = load_board("kv260", boards_dir)
    assert validate_tdg(tdg_event.tdg, board).is_valid
    assert tdg_event.tdg["constraints"]["deadline_s"] == pytest.approx(1 / 25)
    assert "costs" not in tdg_event.tdg
    assert validate_tdg(estimate_costs(tdg_event.tdg, board), board).is_valid


def test_assumptions_never_claim_a_web_source(agent):
    list(agent.handle(UserInput(text="Detect objects in video")))
    events = list(agent.handle(UserInput(answers={"rate_hz": "60"})))
    (notice,) = events_of_type(events, AssumptionsNotice)
    sources = {assumption.key: assumption.source for assumption in notice.assumptions}
    assert sources["rate_hz"] == "user"
    assert sources["resolution"] == "default"  # not answered: the suggestion was used
    assert "web" not in sources.values()


@pytest.mark.parametrize(
    ("task", "expected_name"),
    [
        ("Run a Sobel edge filter on 1080p video", "sobel_edge"),
        ("Multiply two large matrices", "matmul_batch"),
        ("Detect objects in a camera stream", "det_pipeline"),
    ],
)
def test_task_wording_picks_the_example_graph(task, expected_name):
    agent = ScriptedAgent(BOARD_IDS)
    list(agent.handle(UserInput(text=task)))
    (tdg_event,) = events_of_type(agent.handle(UserInput()), TdgReady)
    assert tdg_event.tdg["graph"]["id"] == load_example_graph(expected_name)["graph"]["id"]


def test_one_shot_example_keeps_its_own_timing():
    agent = ScriptedAgent(BOARD_IDS)
    list(agent.handle(UserInput(text="Multiply two large matrices")))
    (tdg_event,) = events_of_type(agent.handle(UserInput()), TdgReady)
    assert tdg_event.tdg["arrival"] == {"kind": "one_shot"}
    assert "constraints" not in tdg_event.tdg


def test_after_finishing_the_agent_says_so(agent):
    list(agent.handle(UserInput(text="Detect objects")))
    list(agent.handle(UserInput()))
    (message,) = agent.handle(UserInput(text="again"))
    assert isinstance(message, AssistantMessage)
    assert "finished" in message.text


def test_events_round_trip_through_json(agent):
    adapter = TypeAdapter(ConversationEvent)
    for event in agent.handle(UserInput(text="Detect objects")):
        assert adapter.validate_json(event.model_dump_json()) == event


def test_examples_are_graph_only_and_listed():
    names = list_example_names()
    assert {"det_pipeline", "sobel_edge", "matmul_batch"} <= set(names)
    for name in names:
        graph = load_example_graph(name)
        assert "costs" not in graph and "$schema" not in graph
    with pytest.raises(KeyError, match="no example named 'nope'"):
        load_example_graph("nope")
