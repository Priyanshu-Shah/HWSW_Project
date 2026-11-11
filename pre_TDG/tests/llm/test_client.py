import httpx
import pytest

from fake_ollama import FakeOllama, text
from hwsw.llm import LlmError, RunLog
from hwsw.llm.client import OllamaUnavailableError, parse_json_reply

SCHEMA = {"type": "object", "properties": {"fps": {"type": "number"}}, "required": ["fps"]}


def test_chat_sets_context_size_and_temperature():
    fake = FakeOllama(text("hi"))
    assert fake.client().chat([{"role": "user", "content": "hello"}])["content"] == "hi"
    assert fake.requests[0]["options"] == {"num_ctx": 32768, "temperature": 0}
    assert "format" not in fake.requests[0]


def test_json_reply_is_parsed_from_a_fenced_block():
    fake = FakeOllama(text('Here:\n```json\n{"fps": 30}\n```'))
    assert fake.client().chat_json([], SCHEMA) == {"fps": 30}


def test_invalid_json_reply_is_sent_back_with_the_error():
    fake = FakeOllama(text('{"fps": "thirty"}'), text('{"fps": 30}'))
    assert fake.client().chat_json([], SCHEMA) == {"fps": 30}
    repair_request = fake.requests[1]["messages"][-1]["content"]
    assert "fps" in repair_request and "not valid" in repair_request


def test_json_reply_gives_up_after_the_attempts():
    fake = FakeOllama(text("no json"), text("still none"))
    with pytest.raises(LlmError, match="no valid JSON reply after 2 attempts"):
        fake.client().chat_json([], SCHEMA, attempts=2)


def test_unreachable_server_is_an_llm_error():
    with pytest.raises(LlmError, match="Ollama"):
        FakeOllama().client().chat([])


def test_ollama_error_text_is_reported():
    out_of_memory = httpx.Response(500, json={"error": "model requires more system memory"})
    with pytest.raises(OllamaUnavailableError, match="HTTP 500: model requires more system memory"):
        FakeOllama(out_of_memory).client().chat([])


def test_the_model_is_kept_loaded():
    fake = FakeOllama(text("hi"))
    fake.client().chat([])
    assert fake.requests[0]["keep_alive"] == -1


def test_calls_are_logged(tmp_path):
    fake = FakeOllama(text("hi"))
    client = fake.client()
    client.log = RunLog(tmp_path / "llm.jsonl")
    client.chat([{"role": "user", "content": "hello"}])
    assert '"kind": "llm_call"' in (tmp_path / "llm.jsonl").read_text()


@pytest.mark.parametrize("reply", ["", "text only", '{"broken": '])
def test_parse_json_reply_rejects_non_json(reply):
    with pytest.raises(ValueError):
        parse_json_reply(reply)
