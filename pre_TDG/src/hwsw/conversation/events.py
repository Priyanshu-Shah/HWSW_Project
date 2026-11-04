"""What the conversation agent says (events) and what the user sends back (UserInput).

This is the contract between the agent (stub today, LLM later) and any front end. A front end
renders events as they arrive; each event has a `type` it can switch on.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Question(BaseModel):
    id: str
    text: str
    suggested_answer: str = ""  # shown pre-filled; accepting it costs the user one click
    choices: list[str] = []  # if non-empty, the answer should be one of these


class Assumption(BaseModel):
    """Something decided without the user stating it. Mirrors `assumptions` in spec.schema.json."""

    key: str
    value: str
    source: Literal["web", "library", "default", "user"]
    url: str = ""
    confidence: Literal["low", "medium", "high"]


class AssistantMessage(BaseModel):
    type: Literal["assistant_message"] = "assistant_message"
    text: str


class QuestionBatch(BaseModel):
    """Several questions asked together (the agent asks in batches, not one by one)."""

    type: Literal["question_batch"] = "question_batch"
    questions: list[Question]


class AssumptionsNotice(BaseModel):
    type: Literal["assumptions"] = "assumptions"
    assumptions: list[Assumption]


class SpecReady(BaseModel):
    type: Literal["spec_ready"] = "spec_ready"
    spec: dict  # validates against spec.schema.json


class TdgReady(BaseModel):
    type: Literal["tdg_ready"] = "tdg_ready"
    tdg: dict  # graph part only; validates against TDG/schemas/tdg.schema.json


class AgentError(BaseModel):
    type: Literal["error"] = "error"
    message: str


ConversationEvent = Annotated[
    AssistantMessage | QuestionBatch | AssumptionsNotice | SpecReady | TdgReady | AgentError,
    Field(discriminator="type"),
]


class UserInput(BaseModel):
    """One turn from the user: free text, and/or answers to a question batch (by question id)."""

    text: str = ""
    answers: dict[str, str] = {}
