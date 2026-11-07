# app

A web page for the pipeline. Left: a conversation with the agent. Right: the task graph (CPU-only,
FPGA-only and either-side tasks coloured differently; setup tasks dashed), a JSON view, and the
validation issues. Buttons: **Load example**, **Validate**, **Cost it** (runs custom_hls and shows
each task's CPU time, FPGA options and the latency bounds), **Download JSON**.

## Status

By default the chat is answered by the **LLM pipeline** (`PipelineAgent` in
`pre_TDG/src/hwsw/llm/`, gpt-oss:120b on Ollama). Stage A looks up boards and facts on the web,
asks question batches, and ends with the assumptions and the task spec. Stage B then builds the
task graph, which takes a few minutes. Each conversation is logged in `pre_TDG/runs/<time>_chat/`.

`--agent scripted` uses `ScriptedAgent` instead (no LLM): it asks four fixed questions and then
hands over one of the ready-made example graphs, chosen by keywords. Everything after that
(validate, cost, bounds) is the real code.

When no board files exist in `library/boards`, the app uses the test fixture boards and shows a
yellow "fixtures" chip: those numbers are not authoritative.

## Run

The app shares the environment of `pre_TDG` (from the repository root):

On Windows the venv folder is `.venv/Scripts/` instead of `.venv/bin/`.

```text
pre_TDG/.venv/bin/python -m pip install -e "pre_TDG[dev]" -e "app[dev]"
pre_TDG/.venv/bin/hwsw-app --port 8000                # then open http://localhost:8000
pre_TDG/.venv/bin/hwsw-app --host 0.0.0.0             # reachable from another machine (no login: lab network only)
pre_TDG/.venv/bin/hwsw-app --agent scripted           # no LLM needed
```

On the Thor, run it there and open it from your laptop (same network, or an SSH tunnel:
`ssh -L 8000:localhost:8000 <thor>`). `HWSW_BOARDS_DIR` overrides the board folder.

Tests: `pre_TDG/.venv/bin/python -m pytest` from `app/` (API tests, plus the Node tests for the
pure JavaScript if Node is installed).

## How it fits together

```text
browser  --HTTP-->  hwsw_app/routes  -->  pipeline_service.py  -->  hwsw (validate, custom_hls)
   │                      └─ chat_routes  -->  ChatSessions  -->  ConversationAgent (stub now, LLM later)
   └── static/js: chat_panel, graph_view (Cytoscape + dagre), node_details, main
```

- Chat replies stream as server-sent events (`assistant_message`, `question_batch`, `assumptions`,
  `spec_ready`, `tdg_ready`, `error`, then `end`), so LLM tokens can stream the same way later.
- `static/vendor/` holds Cytoscape.js 3.30.2, dagre 0.8.5 and cytoscape-dagre 2.5.0 (all MIT),
  copied in so the page works without internet.

## Plugging in the real agent

Write a class that implements `ConversationAgent` (`handle(UserInput) -> Iterator[event]`) and pass
`create_app(agent_factory=...)` in `hwsw_app/main.py`. No front-end change is needed for the events
that exist; richer UI (for example per-question help, editing assumptions) is added in `chat_panel.js`.

## TODO

- [ ] Look at the page in a real browser and fix layout issues (written and tested without one)
- [x] Real agent (stages A and B)
- [ ] Token streaming into one growing message (replies take tens of seconds now)
- [x] Question answers are text boxes (choices offered in a list), so the user can answer
      outside the agent's choices
- [ ] Chat UI for the question batches: skip/"use all suggestions", editing earlier answers
- [ ] Show the specification and assumptions beside the graph, not only in the chat
- [ ] Show a node's design points on the graph (choose a point and see resource use)
- [ ] Partition tab once the post-TDG team has a function to call
- [ ] A single repo-level virtual environment instead of sharing `pre_TDG/.venv`
