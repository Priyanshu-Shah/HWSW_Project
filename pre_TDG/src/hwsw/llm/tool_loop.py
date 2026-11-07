"""The agent loop: the model calls tools until it calls a final one (or runs out of steps)."""

import json
from dataclasses import dataclass

from hwsw.llm.client import LlmError, OllamaClient, ToolCallFormatError
from hwsw.llm.tools import Tool
from hwsw.validation.json_schema_rules import check_against_schema

DEFAULT_MAX_STEPS = 15


@dataclass(frozen=True)
class FinalCall:
    """The call that ended the loop. The caller must append its tool result before continuing."""

    name: str
    arguments: dict


def run_tool_loop(
    client: OllamaClient,
    messages: list[dict],
    tools: list[Tool],
    max_steps: int = DEFAULT_MAX_STEPS,
) -> FinalCall:
    """Extends `messages` in place. Tool failures go back to the model as text, not exceptions."""
    by_name = {tool.name: tool for tool in tools}
    definitions = [tool.definition() for tool in tools]
    for _ in range(max_steps):
        try:
            message = client.chat(messages, definitions)
        except ToolCallFormatError as bad_call:
            client.log.write("warning", message=str(bad_call)[-1000:])
            messages.append({"role": "user", "content": _invalid_tool_call(bad_call)})
            continue
        messages.append(_assistant_turn(message))
        calls = message.get("tool_calls") or []
        if not calls:
            messages.append({"role": "user", "content": _must_call_a_tool(by_name)})
            continue
        for call in calls:
            name = call["function"]["name"]
            arguments = _arguments_of(call)
            tool = by_name.get(name)
            result = _check(tool, name, arguments, by_name)
            if result is None and tool.run is None:
                return FinalCall(name, arguments)
            result = result or _run(tool, name, arguments)
            client.log.write("tool_call", name=name, arguments=arguments, result=result)
            messages.append(tool_result(name, result))
    raise LlmError(f"the model did not finish within {max_steps} steps")


def tool_result(name: str, content: str) -> dict:
    return {"role": "tool", "tool_name": name, "content": content}


def _assistant_turn(message: dict) -> dict:
    # gpt-oss needs its reasoning passed back alongside its tool calls.
    keep = ("content", "thinking", "tool_calls")
    return {"role": "assistant", **{key: message[key] for key in keep if message.get(key)}}


def _arguments_of(call: dict) -> dict | str:
    arguments = call["function"].get("arguments") or {}
    if isinstance(arguments, str):  # some models send the arguments as a JSON string
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def _check(
    tool: Tool | None, name: str, arguments: dict | str, by_name: dict[str, Tool]
) -> str | None:
    """An error message for the model if the call cannot be run, else None."""
    if tool is None:
        return f"error: there is no tool named '{name}'; tools: {', '.join(by_name)}"
    if not isinstance(arguments, dict):
        return f"error: the arguments of {name} must be a JSON object, got: {arguments}"
    problems = check_against_schema(arguments, tool.parameters)
    if problems:
        return f"error: invalid arguments for {name}: " + "; ".join(map(str, problems))
    return None


def _run(tool: Tool, name: str, arguments: dict) -> str:
    try:
        return tool.run(**arguments)
    except Exception as failure:  # noqa: BLE001 - any tool failure is reported to the model
        return f"error: {name} failed: {type(failure).__name__}: {failure}"


def _invalid_tool_call(bad_call: ToolCallFormatError) -> str:
    reason = str(bad_call).rsplit("err=", 1)[-1]
    return (
        f"Your last tool call could not be read: its arguments are not valid JSON ({reason}). "
        "Make the same call again with the arguments as one valid JSON object."
    )


def _must_call_a_tool(by_name: dict[str, Tool]) -> str:
    return f"Do not answer in plain text. Call one of the tools: {', '.join(by_name)}."
