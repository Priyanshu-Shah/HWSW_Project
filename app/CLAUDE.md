# app/ — web front end

A page to chat with the pipeline and see the TDG as a graph. Thin shell: **no pipeline logic here**; call `hwsw` (from `pre_TDG/`) functions only.

- Backend `src/hwsw_app/` (FastAPI): `routes/` = HTTP, `pipeline_service.py` = calls into `hwsw`, `chat_sessions.py`. Frontend `static/` (plain JS modules, no build step); `static/vendor/` is third-party, don't edit.
- Chat contract: events in `pre_TDG/src/hwsw/conversation/events.py`. The real LLM agent plugs in via `create_app(agent_factory=...)`; the scripted stand-in is used until then.
- Pure JS (`format`, `sse`, `graph_elements`) is unit-tested in `tests/js`. Don't put logic in DOM files.
- Never edit `TDG/`, `library/`, `post_TDG/`. Use text nodes, never HTML strings, for anything an LLM wrote.
- Run: `hwsw-app --port 8000`. Test: `pytest` from `app/`, using the `pre_TDG/.venv` environment.
