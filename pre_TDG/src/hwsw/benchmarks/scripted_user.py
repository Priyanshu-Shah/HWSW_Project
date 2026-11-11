"""Answers the agent's questions from a benchmark task's scripted answers, so runs are unattended.

A question gets the first scripted answer with a keyword in its id, else in its text (as the
offered choice it names, or typed in as is when it names none); anything else gets the agent's own
suggestion, as a user clicking "accept" would.
"""

from hwsw.benchmarks.tasks import BenchmarkTask
from hwsw.conversation import Question, QuestionBatch


def answer_batch(task: BenchmarkTask, batch: QuestionBatch) -> dict[str, str]:
    return {question.id: answer_question(task, question) for question in batch.questions}


def answer_question(task: BenchmarkTask, question: Question) -> str:
    # The id names the topic ("hardness"); the text may mention other things in passing
    # ("Is processing each 1 MiB chunk a hard deadline?"), so the id is matched first.
    for text in (question.id.replace("_", " "), question.text):
        for scripted in task.answers:
            if any(keyword.lower() in text.lower() for keyword in scripted.keywords):
                return _fit_choices(scripted.answer, question.choices)
    return question.suggested_answer


def _fit_choices(answer: str, choices: list[str]) -> str:
    """The offered choice the answer names, else the answer itself (choices are only offered)."""
    lowered = answer.lower()
    for choice in choices:
        if choice.lower() == lowered:
            return choice
    for choice in choices:
        if choice.lower() in lowered or lowered in choice.lower():
            return choice
    return answer
