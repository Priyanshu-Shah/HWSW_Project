from fastapi.testclient import TestClient
from hwsw.conversation import AssistantMessage, UserInput
from hwsw.repo_paths import FIXTURE_BOARDS_DIR

from conftest import parse_sse
from hwsw_app.main import create_app
from hwsw_app.settings import AppSettings


def new_session(client) -> str:
    return client.post("/api/chat/sessions").json()["session_id"]


def say(client, session_id: str, **user_input) -> list[tuple[str, dict]]:
    response = client.post(f"/api/chat/sessions/{session_id}/messages", json=user_input)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


def test_task_is_answered_with_questions_then_the_stream_ends(client):
    events = say(client, new_session(client), text="Detect objects in video")
    assert [name for name, _ in events] == ["assistant_message", "question_batch", "end"]
    questions = events[1][1]["questions"]
    assert {question["id"] for question in questions} >= {"platform", "rate_hz"}
    platform = next(question for question in questions if question["id"] == "platform")
    assert platform["choices"] == ["kv260", "pynq_z2", "zcu104"]


def test_whole_conversation_ends_with_a_graph_that_can_be_costed(client):
    session_id = new_session(client)
    say(client, session_id, text="Run a Sobel filter on 1080p video")
    events = say(client, session_id, answers={"platform": "kv260", "rate_hz": "60"})

    assert [name for name, _ in events] == [
        "assumptions", "spec_ready", "assistant_message", "tdg_ready", "end",
    ]  # fmt: skip
    tdg = dict(events)["tdg_ready"]["tdg"]
    assert tdg["platform"] == "kv260"

    costed = client.post("/api/tdg/cost", json={"tdg": tdg}).json()
    assert costed["valid"] is True


def test_sessions_are_independent(client):
    first, second = new_session(client), new_session(client)
    say(client, first, text="Detect objects")
    events = say(client, second, text="Detect objects")
    assert "question_batch" in dict(events)  # the second conversation starts from the beginning


def test_unknown_session_is_a_404(client):
    response = client.post("/api/chat/sessions/nope/messages", json={"text": "hi"})
    assert response.status_code == 404


def test_agent_failure_reaches_the_client_as_an_error_event():
    class BrokenAgent:
        def __init__(self, board_ids):
            pass

        def handle(self, user_input: UserInput):
            yield AssistantMessage(text="starting")
            raise RuntimeError("model server unreachable")

    client = TestClient(create_app(AppSettings(FIXTURE_BOARDS_DIR, "fixtures"), BrokenAgent))
    events = say(client, new_session(client), text="anything")
    assert [name for name, _ in events] == ["assistant_message", "error", "end"]
    assert "RuntimeError: model server unreachable" in events[1][1]["message"]
