"""A stand-in for Ollama's /api/chat: answers with scripted assistant messages, in order."""

import json

import httpx

from hwsw.llm import LlmSettings, OllamaClient


class FakeOllama:
    def __init__(self, *replies: dict):
        self.replies = list(replies)
        self.requests: list[dict] = []  # the JSON bodies received

    def client(self) -> OllamaClient:
        return OllamaClient(LlmSettings(host="http://fake"), transport=httpx.MockTransport(self))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        if not self.replies:
            return httpx.Response(500, json={"error": "no scripted reply left"})
        if isinstance(self.replies[0], httpx.Response):
            return self.replies.pop(0)
        message = {"role": "assistant", **self.replies.pop(0)}
        return httpx.Response(200, json={"message": message, "prompt_eval_count": 10})


def text(content: str) -> dict:
    return {"content": content}


def call(name: str, **arguments) -> dict:
    return {"tool_calls": [{"function": {"name": name, "arguments": arguments}}]}
