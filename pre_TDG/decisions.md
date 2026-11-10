# pre_TDG Decisions

Active decisions internal to our team. Edit in place when one changes; no history. Decision ids
are global across the repo (D1, D3, D4, D5 are in `../TDG/decisions.md`; D2 and D6 are here).

## D2. LLM runtime: Ollama on the Jetson AGX Thor

- The model runs locally in Ollama on the Thor (docker container `ollama`, API on port 11434).
  Host and model are set by `HWSW_OLLAMA_HOST` (default `http://localhost:11434`) and
  `HWSW_LLM_MODEL` (default `gpt-oss:120b`).
- Default model: **gpt-oss:120b** (MXFP4, 65 GB), with `num_ctx` 32768, temperature 0, reasoning
  effort `medium`. The comparison with `gpt-oss:20b` and `qwen3:30b` (Phase 7) may change it.
- The client calls Ollama's **native** `/api/chat` with `httpx`, not the OpenAI-compatible API:
  only the native API takes `num_ctx` and the reasoning effort per call.
- **No grammar-constrained output.** On the Thor's Ollama build, `format` (a JSON schema or
  `"json"`) makes gpt-oss return nothing or hang. Structured output instead comes from either
  - a JSON reply checked against the schema, with errors sent back (`OllamaClient.chat_json`), or
  - a *final* tool whose arguments are the result (`ask_user`, `submit_spec`); checked the same way.
- The whole model must sit on the GPU (`/api/ps`: `size_vram` = `size`). The GPU shares system
  memory, so if the page cache is full Ollama loads only some layers onto the GPU and runs about
  6× slower. Fix: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`, then reload the model.
  Measured: about 21 tokens/s generation with all 37 layers on the GPU.
- **The model is never unloaded** (`keep_alive: -1` on every call). On the Thor, unloading it
  (Ollama's default after 5 idle minutes) left about 65 GB of GPU memory unreleased, even after
  restarting the container; reloading then failed ("model requires more system memory"). Only
  dropping the page cache (above) released it.

## D6. LLM stages as a LangGraph graph of small agents

- Stages A and B are one LangGraph `StateGraph` (`llm/pipeline_agent.py`; `hwsw agents` prints
  it as Mermaid). Each LLM agent does one job with its own short conversation:
  `clarifier`, `ask_user` (pauses the graph until the answers come), `researcher` (work per item
  for each computation step, from the web if needed), `decomposer` (nodes), `edge_builder`, and
  one payload agent per node kind (`opaque_estimator`, `template_designer`, `loop_nest_writer`).
- Checkers are plain code, not LLMs: `validator` (schema, R1-R12, costability), then
  `network_checker`, then `ops_checker`. Each sends a problem back to the agent that made it,
  which fixes it in its own conversation. Limits: 3 repair rounds; one fix per check, then
  what is left is a warning.
- The state holds only JSON-like values and is checkpointed in memory after every agent turn:
  a failed agent is run again on the user's next message, from the last finished turn.
- LangGraph only orchestrates. The model is still called through our client (D2), not
  LangChain's `ChatOllama`, whose structured output relies on Ollama's `format`, which hangs here.
- Agents run one after another: the Thor serves one request at a time, and parallel requests
  would need more context memory.
