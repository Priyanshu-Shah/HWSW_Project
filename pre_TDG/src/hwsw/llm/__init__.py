"""LLM pipeline (D2: Ollama on the Thor): stage A clarification agent, stage B TDG builder."""

from hwsw.llm.clarify import ClarificationAgent
from hwsw.llm.client import LlmError, LlmSettings, OllamaClient
from hwsw.llm.pipeline_agent import PipelineAgent
from hwsw.llm.run_log import RunLog, new_run_dir
from hwsw.llm.tdg_builder import TdgBuild, TdgBuilder, TdgBuildError
from hwsw.llm.tool_loop import FinalCall, run_tool_loop

__all__ = [
    "ClarificationAgent",
    "FinalCall",
    "LlmError",
    "LlmSettings",
    "OllamaClient",
    "PipelineAgent",
    "RunLog",
    "TdgBuild",
    "TdgBuildError",
    "TdgBuilder",
    "new_run_dir",
    "run_tool_loop",
]
