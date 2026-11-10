# pre_TDG/ — English task → costed TDG

Our team's work. Goal: a plain-English task → `tdg.costed.json` (valid TDG + cost map), which `post_TDG/` consumes.

Two parts:
- **LLM pipeline** (`src/hwsw/llm/`): a LangGraph graph of small agents and code checkers (D6).
- **custom_hls** (`src/hwsw/custom_hls/`): loop nests + board specs → HW design points and SW time → `costs`.

Rules:
- Plan: `implementation.md`. Decisions: `decisions.md` (D2 runtime, D6 agents).
- Format is defined by `../TDG/` (D1, D3, D4). Board specs come from `../library/`.
- **Never edit `../TDG/`, `../library/` or `../post_TDG/`. If a format change is needed, stop and tell the user** (other teams must be informed).
- `spec.json`, logs and custom_hls constants are internal.
- Code: `src/hwsw/`; tests mirror it. Check with `.venv/bin/python -m pytest` and `-m ruff check .`
- Status: validator, custom_hls, CLI, LLM stages A+B, benchmarks done; next: model comparison, calibration. Read `custom_hls/README.md` and `validation/README.md` before changing them.
