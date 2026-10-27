# pre_TDG/ — English task → costed TDG

Our team's work. Goal: a plain-English task → `tdg.costed.json` (valid TDG + cost map), which `post_TDG/` consumes.

Two parts:
- **LLM pipeline** (`src/llm/`): clarification agent (cited web search) + staged TDG builder (validate → repair).
- **custom_hls** (`src/custom_hls/`): loop nests + board specs → HW design points and SW time → `costs`.

Rules:
- Plan: `implementation.md`. Decisions: `decisions.md` (D2).
- Format is defined by `../TDG/` (D1, D3, D4). Board specs come from `../library/`.
- **Never edit `../TDG/`, `../library/` or `../post_TDG/`. If a format change is needed, stop and tell the user** (other teams must be informed).
- `spec.json`, logs and custom_hls constants are internal.
