# pre_TDG Implementation Plan

Owner: our team. **Input:** a plain-English task. **Output:** `tdg.costed.json`: a valid TDG
(graph part) plus its cost map (`costs`), as defined by `../TDG/schemas/tdg.schema.json`.
`post_TDG/` (the partitioner team) starts from this file.

Two tracks that meet in Phase 4 and Phase 8:
- **LLM track:** English task → `tdg.json` (Phases 1–4).
- **custom_hls track:** `tdg.json` + board specs → `tdg.costed.json` (Phases 5–6).

## 1. Goal and definition of done

A user types a small task, for example:

> "Real-time object detection on a 720p camera stream at 30 fps on a ZCU104."

The pipeline asks the questions it needs, states its assumptions with sources, writes a valid
TDG, and costs every node and edge.

**Done when:**
- ≥ 8 of the 10 benchmark tasks produce a valid costed TDG with no manual edits;
- ≥ 95% of generated TDGs pass the validator within 3 repair attempts;
- op counts are within 2× of the reference TDGs for ≥ 80% of nodes;
- custom_hls HW latency and resources are within ±50% of Vitis HLS on the calibration kernels;
- the clarification agent asks ≤ 2 rounds of questions on average;
- the post_TDG team has run at least 3 of our outputs through their partitioner.

## 2. Data flow

```text
user text
  │
  ▼  [A] clarification agent  (LLM + tools: ask_user, web_search, fetch_page, platform_lookup)
spec.json         ← internal: task summary, platform, timing + hardness, assumptions + sources
  │
  ▼  [B] TDG builder  (LLM, staged; validate → repair loop)
tdg.json          ← contract: graph part only (no `costs`)
  │
  ▼  [C] custom_hls  (+ board specs from ../library/)
tdg.costed.json   ← contract: graph + `costs`
  │
  ▼  validator + bounds check (sanity only, not a partitioner)
  │
  └──▶ handed to post_TDG/
```

Each stage reads and writes files, so any stage can be run alone, tested alone, or rerun.
Each run writes everything into `runs/<timestamp>_<graph_id>/`, including the LLM logs.

## 3. Dependencies on other folders

| Need | From | Until it exists |
|---|---|---|
| TDG schema + examples | `../TDG/schemas/` | exists |
| Board specs: `index.json`, `boards/<id>.json` (format in `../library/README.md`) | `../library/` | hand-made files for 2–3 boards in `tests/fixtures/platforms/` |

We never edit those folders. The validator lives here (`src/hwsw/validation/`).

## 4. Tech stack

| Concern | Choice |
|---|---|
| Language / packaging | Python 3.11+, `uv`, `ruff`, `pytest` |
| Schemas | `../TDG/schemas/*.json` are the source of truth. The validator checks plain dicts with `jsonschema`; no generated model classes. Sub-schemas are passed to Ollama `format` |
| File format | JSON everywhere; `jsonschema` for validation |
| Graph checks | `networkx` (DAG check, topological order, critical path) |
| LLM client | `openai` package → Ollama on Jetson AGX Thor (D2), `num_ctx ≥ 32768` |
| Models to compare | `gpt-oss:20b`, `qwen3:30b`, `gpt-oss:120b` |
| Web search | `ddgs` (DuckDuckGo, no API key), behind one `web_search` tool; Tavily as an alternative |
| Page fetch | `httpx` + `trafilatura` (main text only) |
| HLS calibration | AMD Vitis HLS (C synthesis reports only; no bitstreams) |
| CLI | `typer` + `rich` |
| Agent loop | hand-written for the MVP (~100 lines); move to LangGraph only if the state gets complex |
| Logging | one JSONL file per run: every prompt, response, tool call, token count, latency |

## 5. Layout

`[done]` exists and is tested. `[next]` is not written yet.

```text
pre_TDG/
  pyproject.toml          package `hwsw`, `hwsw` command; venv in .venv/ (git-ignored)
  src/hwsw/
    cli.py                [done] validate | cost | bounds      [next] clarify | build | run
    repo_paths.py  json_files.py                               [done] shared helpers
    platforms/            [done] board files -> typed Board
      board_model.py  board_loader.py
    validation/           [done] TDG schema + rules R1-R12, one module per group of rules
      json_schema_rules.py  graph_rules.py  loop_nest_rules.py  constraint_rules.py
      cost_map_rules.py  platform_rules.py  tdg_validator.py  issue.py  id_listing.py
    spec/                 [done] spec.schema.json, spec_validation.py
    custom_hls/           [done] loop nests + board -> cost map
      templates/          conv2d  gemm  pool  elementwise  reduce  memcpy
      loop_nest.py  loop_nest_analysis.py  hw_design_points.py  sw_time.py
      node_costing.py  estimate_tdg.py  bounds.py  constants.py  estimate_values.py  data_types.py
      constants.json      [next] written by calibration against Vitis HLS
    llm/                  [next] needs the Thor
      client.py  agent.py  clarify.py  tdg_builder.py  tools/  prompts/
  calibration/            [next] Vitis HLS reference kernels, reports, fit scripts
  benchmarks/             [next] tasks/ and reference/ (hand-made reference TDGs)
  tests/                  [done] mirrors src/hwsw; fixtures/ (platforms/, tdgs/), helpers.py
  runs/                   (git-ignored)
```

Run: `.venv/Scripts/python -m pytest` and `.venv/Scripts/python -m ruff check .` from `pre_TDG/`.

## 6. Phases (estimates for ~2 people per track)

### Phase 0: Setup (week 1, both tracks)
- Project skeleton, ruff, pytest, CI.
- Thor: install Ollama (check JetPack 7 / GPU use with `ollama ps`), max power mode,
  `OLLAMA_HOST=0.0.0.0`, pull the 3 candidate models.
- Smoke test from a laptop: one chat call, one tool call, one schema-constrained JSON reply,
  for each model. Record tokens/s.
- Vitis HLS installed and able to synthesize a small kernel for each board's part.

**Exit:** all three models answer a tool call and a schema-constrained reply over the lab network;
Vitis HLS produces a report for a toy kernel.

### Phase 1: Schemas, validator and spec (week 1–2, both tracks) — highest priority — DONE
- Validator `src/hwsw/validation/`: JSON Schema check first, then the semantic rules R1–R12 listed in
  `tdg.schema.json` (ids, references, DAG, `lo ≤ value ≤ hi`, platform capacity, `hw_only`
  points, 2× ops check, D3 objective and deadline rules). R5 needs the board files.
- Error messages are written for an LLM to read ("edge e4: dst 'n9' does not exist; existing
  ids: n1..n8"), because they feed the repair loop.
- Fixtures: 3 valid TDGs + cost maps and 10 invalid ones.
- `spec.schema.json` (internal, lives here, not in `../TDG/`). Example:

  ```json
  {
    "task_summary": "...",
    "platform": "zcu104",
    "arrival": {"kind": "streaming", "period_s": 0.0333},
    "timing": {"deadline_s": 0.0333, "hardness": "hard", "derived_from": "30 fps, sequential (D3)"},
    "assumptions": [
      {"key": "model", "value": "YOLOv8n, 8.7 GFLOPs @640", "source": "web", "url": "...", "confidence": "medium"}
    ],
    "open_questions": []
  }
  ```

  `open_questions` must be empty before stage B.

**Exit:** `hwsw validate` works on `tdg.json` and `tdg.costed.json`; `spec.json` has a schema.

### Phase 2: LLM infrastructure (week 2–3, LLM track)
- `client.py`: one function `chat(messages, tools=None, schema=None, model=...)`; sets
  `num_ctx`, temperature 0; uses Ollama `format` for schemas; Pydantic-validates; retries on
  invalid output; logs to the run's JSONL.
- `agent.py`: tool loop with a step limit, tool error handling (errors go back to the model
  as text), and time/token counters.
- Tools: `web_search` (ddgs, results cached on disk by query), `fetch_page` (cut to ~4k tokens),
  `platform_lookup` (reads `../library/index.json` and board files), `ask_user` (CLI prompt;
  takes a **list** of questions, each with a suggested default).

**Exit:** a test agent can look up a fact on the web and return it with a URL, on all 3 models.

### Phase 3: Stage A — clarification agent (week 4–5, LLM track)
- System prompt rules:
  1. Platform first: show the library list; never invent a board.
  2. Draft the task internally; ask only about gaps that block costing
     (sizes, rates, data types, model choice, timing hardness).
  3. Fill gaps in this order: library → web search (cite URL) → stated default → ask the user.
  4. Ask in batches of ≤ 5 questions, each with a suggested default; ≤ 3 rounds.
  5. Timing (D3): classify as hard / soft / none; set `deadline_s` with `derived_from`.
- Output: `spec.json`, validated; `open_questions` must be empty.
- For known NN models: search for params / GFLOPs / input size and record them as
  assumptions; the user confirms or picks another.
- Test with scripted users (the answers in `benchmarks/tasks/*.md`) so it runs unattended.

**Exit:** valid spec for all 10 benchmark tasks with scripted answers; average ≤ 2 rounds.

### Phase 4: Stage B — TDG builder (week 5–7, LLM track; needs Phase 5 for step 5)
Staged generation, one schema-constrained call per step, each checked before the next:
1. **Decompose:** node list: `id, name, phase, affinity, kind`, one line of purpose each.
   Rules: I/O and OS work → `opaque` + `sw_only`; standard ops → `template`;
   everything else → `loop_nest`; setup work → `phase: setup`.
2. **Edges:** data/control, with `bytes` written as a formula over spec values
   (e.g. `"1280*720*3"`). **Python evaluates the formula** to a literal (D4).
3. **Payloads:** per node, template params or a D1 loop nest + `ops_claimed`. One call per node
   (keeps prompts short; can run in parallel).
4. **Validate → repair:** run the validator; send errors back; ≤ 3 attempts per step.
5. **Ops check:** run custom_hls; for nodes off by more than 2×, regenerate that node
   once, then flag it in the run log.
- Write `constraints.deadline_s` and `arrival` from the spec (D3).
- NN models in the MVP: the backbone as a few `template` nodes (stages/blocks with
  representative conv params). Layer-exact expansion is post-MVP.

**Exit:** ≥ 95% valid TDGs within the repair budget on the benchmark set.

### Phase 5: custom_hls (week 2–6, custom_hls track) — built, uncalibrated

**Input:** `tdg.json` + a board file. **Output:** `tdg.costed.json`.

1. **Templates → loop nests** (`templates/`): `conv2d`, `gemm`, `pool`, `elementwise`, `reduce`,
   `memcpy`. Each turns `template.params` into a D1 loop nest, so everything below handles only
   one format.
2. **Analysis** (`loop_nest_analysis.py`), per loop nest:
   - ops = product of loop ranges × body ops (written as `ops_estimated`, same unit as `ops_claimed`);
   - off-chip bytes from `in`/`out` arrays; on-chip buffers from access patterns
     (sliding window → line buffer of (K−1)·W·bytes; reuse → full array; streaming → none);
   - loop-carried dependencies or `control_heavy` → II > 1.
3. **HW design points** (`hw_design_points.py`):
   - P from unrolling the inner loops (hints from the LLM are advisory), e.g. P ∈ {small, medium,
     large}; drop points that exceed the board capacity;
   - cycles = iterations / P × II + pipeline depth; `time_s` = max(compute time, memory time)
     (roofline with the board's DRAM bandwidth);
   - DSP = P × DSP-per-op (by dtype); BRAM18K = buffer bytes / 2.25 KB (with partitioning for P
     parallel reads); LUT/FF = P × per-op cost + control overhead;
   - `ii_s` for the whole node when pipelined; `lo`/`hi` from the calibration error (±30% to start);
     `source: roofline` until calibrated.
4. **SW model** (`sw_time.py`): ops / (freq × cores-used × ops_per_cycle(dtype) × efficiency),
   plus memory time for large arrays and a branch penalty for `control_heavy`.
   `source: profile_model`.
5. **Opaque nodes:** copy `opaque.sw_time_s` into the cost map, so the partitioner reads only `costs`.
6. **Edges:** no output unless a node needs an interface override (D4 default applies:
   `setup_s + bytes / bandwidth_Bps` from the board's default interface).
7. **Calibration** (`calibration/`): 10–15 reference kernels in HLS C++ (the 6 templates at 2–3
   sizes and dtypes + 3 irregular kernels). Synthesize each at the same P values custom_hls uses;
   fit the per-op constants in `constants.json`; set `lo`/`hi` from the remaining error.
   Estimates keep `source: roofline`; `hls_report` is only for numbers read directly from a
   Vitis HLS report.
8. **Tests:** the 3×3 conv on 2048×2048 int16 worked example matches the hand calculation;
   every template's op count matches its closed form; outputs pass the validator.

**Conventions as built** (all constants are in `constants.py`, uncalibrated until step 7):
- One multiply-accumulate counts as one op; `load`/`store` are not ops. `ops_unit: macs` counts only `mac`.
- Every array crosses DRAM once; reuse comes from on-chip buffers (line buffer for a sliding window,
  the whole array for `reuse`, const lookup tables for `random`). Large `reuse` arrays (for example
  gemm's B, untiled) can make a node not fit, which correctly gives it no hardware option.
- Design points: P = unroll hint x (1, 4, 16); if none fit the board the hint is ignored (P = 1, 4, 16).
- FPGA memory bandwidth is the smaller of DRAM and the board's default interface; the CPU gets DRAM.
- Software baseline is single-threaded (`sw_cores_used = 1`). Sources: HW `roofline`, SW `profile_model`.

**Exit:** `hwsw cost tdg.json` writes a valid `tdg.costed.json` for all example TDGs; HW latency
and resources within ±50% of Vitis HLS on the calibration kernels.

### Phase 6: Bounds check (week 4, custom_hls track) — DONE
A quick sanity check of a costed TDG without a partitioner (not a replacement for it):
- **All-SW latency:** longest path through iteration nodes using SW times.
- **Best case:** longest path using each node's fastest option, no comm, no resource limit.
- If the best case is above `deadline_s`, warn: no partition can meet the deadline.
- If all-SW ≈ best case, warn: nothing worth moving to HW (likely a graph or cost problem).

**Exit:** `hwsw bounds` runs on every benchmark output; used in Phase 7 scoring.

### Phase 7: Benchmarks and model comparison (week 6–8, both tracks)
- 10 tasks with hand-made reference TDGs:

  | # | Task | Exercises |
  |---|---|---|
  | t01 | Object detection, 720p @ 30 fps | NN templates, streaming deadline |
  | t02 | Sobel edge detection, 1080p @ 60 fps | loop nest, sliding window |
  | t03 | Image classification (MobileNetV2), one-shot | NN, no deadline |
  | t04 | FIR filter on audio, 48 kHz | streaming, tiny nodes |
  | t05 | FFT-based radar range processing | template + loop nest mix |
  | t06 | AES-128 encryption of a 1 Gb/s stream | control-heavy vs dataflow |
  | t07 | Matrix multiply 1024×1024, batch | gemm template, no deadline |
  | t08 | JPEG encoding of 1080p frames | mixed, branchy entropy coding |
  | t09 | Kalman filter at 1 kHz, hard 1 ms limit | tight latency deadline |
  | t10 | Video stabilization, 720p @ 30 fps | long pipeline |

- Scores:
  - spec: correct platform, timing classification, assumption sources present;
  - TDG: valid rate, node matching vs reference (F1 by name/role), edge F1,
    op-count error, bytes error;
  - cost map: all-SW and best-case bounds vs the reference TDG's bounds (large gaps point to a
    graph problem or a cost-model problem);
  - effort: questions asked, LLM calls, wall time, tokens.
- Run all 10 tasks × 3 models × 3 repeats → choose the default model; record it in `decisions.md`.

**Exit:** a results table in this folder; default model chosen.

### Phase 8: Integration and handoff (week 8–9, both tracks)
- `hwsw run "<task>"`: A → B → C → validate → bounds in one command, resumable from any
  stage's file.
- Run summary (`summary.md`): assumptions with sources, TDG diagram (Mermaid), costs per node,
  bounds, warnings (2× flags, low-confidence estimates, unreachable deadline).
- Deliver the benchmark outputs (`tdg.costed.json`) to the post_TDG team; their results become
  the end-to-end check of the whole project.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Open-weight model fails at multi-step tool use | staged calls with small schemas; validate + repair; compare 3 models early (Phase 0) |
| Wrong numbers (sizes, bytes, ops) | formulas evaluated by Python; 2× ops check; assumptions with sources |
| Web search returns noise or nothing | cached results; prefer model cards / papers; fall back to a stated default |
| Too many or too few questions | prompt rules + caps; measured in benchmarks |
| Ollama context cut silently | `num_ctx` set on every call; log prompt token counts and warn near the limit |
| custom_hls estimates far off | calibration against Vitis HLS; ±ranges recorded; bounds check flags odd results |
| Board files late or in a different shape | fixtures for 2–3 boards; format fixed in `../library/README.md`; changes go through the user |
| Contract drift with other teams | change rule (D5); schemas and examples they can test against |
