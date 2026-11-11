import httpx
import pytest

from fake_ollama import FakeOllama, call, text
from hwsw.llm import FinalCall, LlmError, run_tool_loop
from hwsw.llm.tools import Tool, object_schema

ADD = Tool(
    name="add",
    description="Add two numbers.",
    parameters=object_schema({"a": {"type": "number"}, "b": {"type": "number"}}),
    run=lambda a, b: str(a + b),
)
DONE = Tool(name="done", description="Finish.", parameters=object_schema({"x": {"type": "number"}}))


def tool_results(messages):
    return [message["content"] for message in messages if message["role"] == "tool"]


def test_tools_run_until_the_final_tool_is_called():
    fake = FakeOllama(call("add", a=1, b=2), call("done", x=3))
    messages = [{"role": "user", "content": "1+2?"}]
    assert run_tool_loop(fake.client(), messages, [ADD, DONE]) == FinalCall("done", {"x": 3})
    assert tool_results(messages) == ["3"]


@pytest.mark.parametrize(
    ("bad_call", "expected"),
    [
        (call("subtract", a=1), "no tool named 'subtract'"),
        (call("add", a="one", b=2), "invalid arguments for add"),
        (call("done", x="three"), "invalid arguments for done"),
    ],
)
def test_bad_calls_are_reported_to_the_model(bad_call, expected):
    fake = FakeOllama(bad_call, call("done", x=3))
    messages = []
    run_tool_loop(fake.client(), messages, [ADD, DONE])
    assert expected in tool_results(messages)[0]


def test_a_failing_tool_is_reported_to_the_model():
    def explode():
        raise RuntimeError("network down")

    broken = Tool("broken", "Fails.", object_schema({}), run=explode)
    fake = FakeOllama(call("broken"), call("done", x=1))
    messages = []
    run_tool_loop(fake.client(), messages, [broken, DONE])
    assert "network down" in tool_results(messages)[0]


def test_a_plain_text_reply_is_answered_with_a_reminder():
    fake = FakeOllama(text("I think the answer is 3"), call("done", x=3))
    messages = []
    run_tool_loop(fake.client(), messages, [ADD, DONE])
    assert "Call one of the tools" in messages[1]["content"]


def test_the_loop_stops_after_max_steps():
    fake = FakeOllama(*[call("add", a=1, b=1)] * 3)
    with pytest.raises(LlmError, match="within 3 steps"):
        run_tool_loop(fake.client(), [], [ADD, DONE], max_steps=3)


def test_a_malformed_tool_call_is_asked_again():
    malformed = httpx.Response(
        500,
        json={"error": 'error parsing tool call: raw=\'{"x":3}"}\', err=invalid character'},
    )
    fake = FakeOllama(malformed, call("done", x=3))
    messages = []
    assert run_tool_loop(fake.client(), messages, [ADD, DONE]) == FinalCall("done", {"x": 3})
    assert "not valid JSON (invalid character)" in messages[0]["content"]


def test_an_ollama_outage_is_not_retried():
    out_of_memory = httpx.Response(500, json={"error": "model requires more system memory"})
    with pytest.raises(LlmError, match="^Ollama at"):
        run_tool_loop(FakeOllama(out_of_memory).client(), [], [ADD, DONE])
