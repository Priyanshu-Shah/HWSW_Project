"""Chat with the model served by Ollama on the Thor (D2): plain replies, tool calls, JSON replies.

Ollama's native API is used (not its OpenAI-compatible one) because only the native API takes
`num_ctx` and the reasoning effort per call.
"""

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from hwsw.llm.run_log import RunLog
from hwsw.validation.json_schema_rules import check_against_schema

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gpt-oss:120b"
HOST_ENV_VAR = "HWSW_OLLAMA_HOST"
MODEL_ENV_VAR = "HWSW_LLM_MODEL"

_TOOL_CALL_PARSE_ERROR = "error parsing tool call"  # how Ollama reports a malformed tool call

# Warn in the run log when a prompt fills this much of the context (Ollama cuts silently).
_CONTEXT_WARNING_FRACTION = 0.9


class LlmError(RuntimeError):
    """The model could not be reached, or did not produce a usable answer."""


class OllamaUnavailableError(LlmError):
    """Ollama failed the request itself (down, model not loadable): retrying a task won't help."""


class ToolCallFormatError(LlmError):
    """The model wrote a tool call that Ollama could not parse (seen with gpt-oss: a stray quote
    in the JSON). Not an outage: asking the model again usually fixes it."""


@dataclass(frozen=True)
class LlmSettings:
    host: str = DEFAULT_HOST
    model: str = DEFAULT_MODEL
    num_ctx: int = 32768
    think: str = "medium"  # gpt-oss reasoning effort: low | medium | high
    timeout_s: float = 900.0
    # Never unload: on the Thor, unloading can leave the GPU memory unreleased (D2).
    keep_alive: int = -1

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] = os.environ) -> "LlmSettings":
        return cls(
            host=environment.get(HOST_ENV_VAR, DEFAULT_HOST),
            model=environment.get(MODEL_ENV_VAR, DEFAULT_MODEL),
        )


class OllamaClient:
    def __init__(
        self,
        settings: LlmSettings | None = None,
        log: RunLog | None = None,
        transport: httpx.BaseTransport | None = None,  # tests pass a mock transport
    ):
        self.settings = settings or LlmSettings.from_environment()
        self.log = log or RunLog()
        self._http = httpx.Client(
            base_url=self.settings.host, timeout=self.settings.timeout_s, transport=transport
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """One model turn. Returns the assistant message (`content`, `thinking`, `tool_calls`)."""
        body: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
            "think": self.settings.think,
            "keep_alive": self.settings.keep_alive,
            "options": {"num_ctx": self.settings.num_ctx, "temperature": 0},
        }
        if tools:
            body["tools"] = tools
        started = time.monotonic()
        try:
            response = self._http.post("/api/chat", json=body)
        except httpx.HTTPError as failure:
            raise OllamaUnavailableError(f"Ollama at {self.settings.host}: {failure}") from failure
        if response.is_error:
            error_text = _error_text(response)
            if error_text.startswith(_TOOL_CALL_PARSE_ERROR):
                raise ToolCallFormatError(error_text)
            status = f"HTTP {response.status_code}: {error_text}"
            raise OllamaUnavailableError(f"Ollama at {self.settings.host}: {status}")
        reply = response.json()
        prompt_tokens = reply.get("prompt_eval_count", 0)
        self.log.write(
            "llm_call",
            model=self.settings.model,
            messages=messages,
            tools=[tool["function"]["name"] for tool in tools or []],
            reply=reply["message"],
            prompt_tokens=prompt_tokens,
            output_tokens=reply.get("eval_count", 0),
            seconds=round(time.monotonic() - started, 2),
        )
        if prompt_tokens > _CONTEXT_WARNING_FRACTION * self.settings.num_ctx:
            self.log.write("warning", message=f"prompt uses {prompt_tokens} tokens of the context")
        return reply["message"]

    def chat_json(self, messages: list[dict], schema: dict, attempts: int = 3) -> Any:
        """A reply that is one JSON value passing `schema`; invalid replies are sent back to fix.

        Ollama's `format` (grammar-constrained output) returns nothing for gpt-oss on the Thor's
        build, so the schema goes into the prompt and the reply is checked here instead.
        """
        messages = [*messages, {"role": "user", "content": _json_instruction(schema)}]
        problems: list[str] = []
        for _ in range(attempts):
            text = self.chat(messages).get("content", "")
            try:
                value = parse_json_reply(text)
            except ValueError as not_json:
                problems = [str(not_json)]
            else:
                problems = [str(issue) for issue in check_against_schema(value, schema)]
                if not problems:
                    return value
            messages = [
                *messages,
                {"role": "assistant", "content": text},
                {"role": "user", "content": _fix_request(problems)},
            ]
        raise LlmError(f"no valid JSON reply after {attempts} attempts: {'; '.join(problems)}")


def _error_text(response: httpx.Response) -> str:
    try:
        return response.json()["error"]
    except (ValueError, KeyError, TypeError):
        return response.text[:300]


def parse_json_reply(text: str) -> Any:
    """The first JSON object or array in `text`, ignoring Markdown fences or words around it."""
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        raise ValueError("the reply contains no JSON object")
    try:
        value, _ = json.JSONDecoder().raw_decode(text, min(starts))
    except json.JSONDecodeError as failure:
        raise ValueError(f"the reply is not valid JSON: {failure}") from failure
    return value


def _json_instruction(schema: dict) -> str:
    return (
        "Reply with only one JSON value (no prose, no Markdown) that matches this JSON Schema:\n"
        + json.dumps(schema)
    )


def _fix_request(problems: list[str]) -> str:
    listed = "\n".join(f"- {problem}" for problem in problems)
    return f"That reply is not valid:\n{listed}\nReply again with only the corrected JSON."
