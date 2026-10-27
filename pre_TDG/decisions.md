# pre_TDG Decisions

Active decisions for the pre_TDG work (LLM pipeline and custom_hls). Edit in place when one
changes; no history. Shared decisions (D1, D3, D4, D5) are in `../TDG/decisions.md`.

## D2. LLM host: Jetson AGX Thor + Ollama

- Models run locally on a Jetson AGX Thor (128 GB unified memory, ~273 GB/s) using Ollama.
- The pipeline uses Ollama's OpenAI-compatible API (`openai` package, custom `base_url`), so
  models and hosts can be swapped.
- Always set the context length explicitly (`num_ctx` ≥ 32k). Ollama's default is small and
  silently cuts off long prompts.
- Structured outputs use Ollama's `format` (JSON schema) + Pydantic validation.
- Model choice is open: candidates `gpt-oss:20b`, `qwen3:30b`, `gpt-oss:120b`; decided by the
  benchmark comparison (Phase 7 in `implementation.md`).
