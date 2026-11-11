"""A tool the model can call: its name, what it does, its arguments, and the code that runs it."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema of the arguments object
    # Returns text for the model. `None` marks a *final* tool: calling it ends the tool loop and
    # hands the call to the caller (for example `ask_user`, which must wait for the user).
    run: Callable[..., str] | None = None

    def definition(self) -> dict:
        """The tool as Ollama's `tools` list expects it."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def object_schema(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties),
    }
