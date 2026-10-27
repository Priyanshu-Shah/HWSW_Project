# TDG Decisions

Active decisions that define the interface between teams. Edit in place when one changes;
no history. Decision ids (D1…) are global across the repo; each lives in exactly one file
(D2 is in `pre_TDG/decisions.md`).

**Any change here is a contract change (D5): tell the project lead first.**

## D1. Loop-nest description

Nodes of `kind: loop_nest` carry a structured loop-nest description, written by the LLM.
Formal definition: `$defs/loop_nest` in `schemas/tdg.schema.json`. custom_hls also generates this
format internally from `template` nodes.

```json
{
  "arrays": {
    "img":    {"shape": [2048, 2048], "dtype": "int16", "dir": "in"},
    "kernel": {"shape": [3, 3],       "dtype": "int16", "dir": "in", "const": true},
    "out":    {"shape": [2046, 2046], "dtype": "int16", "dir": "out"}
  },
  "loops": [
    {"var": "y", "range": 2046}, {"var": "x", "range": 2046},
    {"var": "ky", "range": 3},   {"var": "kx", "range": 3}
  ],
  "body": {"mac": 1},
  "access": {"img": {"pattern": "sliding_window", "reuse": ["ky", "kx"]}},
  "control": {"branches": 0},
  "hints": {"unrollable": ["ky", "kx"]}
}
```

- Loops are listed outermost first; `body` counts ops per innermost iteration; `hints` are advisory.
- The LLM describes **what** is computed (plain reference loops). It does not choose pragmas
  or parallelism.
- custom_hls tries parallelism levels P, pipelining and array partitioning, and returns HW design
  points (latency, DSP, BRAM, LUT, FF) that fit the board, plus the SW time.
- Edge sizes in bytes come from the `in`/`out` array sizes.
- custom_hls constants are calibrated against Vitis HLS on 10–15 reference kernels.
  Target accuracy: ±30–50%.
- Sanity check: if `ops_claimed` (LLM) and `ops_estimated` (custom_hls) differ by more than 2×,
  regenerate the node or flag it to the user.
- Irregular code (data-dependent loops, pointer chasing) → mark `control_heavy`;
  custom_hls gives it a high HW cost.

## D3. Timing model: the partitioner only minimizes latency

- The partitioner's only objective is **minimum latency of one iteration** (one pass through all
  `phase: iteration` nodes, end to end). `phase: setup` nodes are excluded.
- Task-dependent requirements (fps, response time) are turned into an optional latency deadline
  by the LLM stage and written to the TDG's `constraints.deadline_s`. The partitioner treats it as a
  constraint when present (`latency ≤ deadline_s`).
- Conversion rule (frames processed one after another, not pipelined):
  `deadline_s = min(1 / rate_hz, max_latency_s)`. A safe, sufficient condition: a pipelined
  design could reach the rate with a longer latency.
- No hard requirement stated ("as fast as possible") → no `deadline_s`; just minimize.
- Throughput is reported after partitioning (`1 / latency_s`), never optimized directly.

## D4. TDG schema v0.2

- **Format: JSON.** Formal definition: `schemas/tdg.schema.json` (JSON Schema draft 2020-12,
  plus semantic rules R1–R12 listed in it, enforced by the validator). Worked example:
  `schemas/examples/det_pipeline.tdg.json`. All teams exchange TDGs in this format.
- One document, two parts: the **graph** (nodes, edges, arrival, constraints, objective,
  platform), written by the LLM pipeline, and the **cost map** under `costs`, added by
  custom_hls (both in `pre_TDG/`). The cost map alone validates against `$defs/cost_map`.
- Node kinds: `opaque` (fixed SW cost), `loop_nest` (D1), `template` (op + params; custom_hls
  expands it into a loop nest).
- Every estimate is `{value, lo, hi, source}`. SI units. Numbers are literals: the LLM stage
  evaluates any formulas before writing the TDG.
- HW costs are a list of design points; the partitioner picks one per node.
- `schema_version` is the string `"0.2"`. `objective` must be `{latency: 1.0}` (D3).

## D5. Repo scope, teams and schema changes

- Folders and owners:
  - `pre_TDG/`: our team. LLM pipeline **and custom_hls**: English task → costed TDG (graph +
    cost map).
  - `post_TDG/`: the post-TDG team (partitioner and metrics). Starts from the costed TDG.
  - `library/`: the library team. Board specifications (format in `library/README.md`).
  - `TDG/`: this shared interface.
- The costed TDG JSON (`schemas/tdg.schema.json`) and the board files (`library/`) are the
  contract between the teams. **Any change to them** (fields, names, units, meaning) bumps
  `schema_version` and must be raised with the user first, so they can inform the other teams.
- Teams meet only at this contract: no team edits another's folder.
