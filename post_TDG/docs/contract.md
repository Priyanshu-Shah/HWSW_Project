# Integration contract: task graph (IR) → partitioner → Result

This file is for the task-generation (NLP) team: what to emit, and what you get back.

- **Machine-readable schemas** (JSON Schema 2020-12): `schemas/json/ir.schema.json`, `platform.schema.json`, `result.schema.json`, `ranking.schema.json`.
- **Source of truth:** the pydantic models in `schemas/*.py`. Regenerate the JSON files with `python -m schemas.export`; a test fails if they drift.
- **Version:** `schema_version` is `"0.1.0"`. Breaking changes bump it.
- **Worked examples:** `examples/golden/*.synthetic.json`. These are also the contract tests: if your output for the same spec differs in shape, `pytest` tells you where.

## What a task graph must contain

```jsonc
{
  "schema_version": "0.1.0",
  "id": "collision_detection",                       // [A-Za-z_][A-Za-z0-9_.-]*
  "provenance": {"kind": "synthetic", "source": "..."},
  "workload": {"width": 1280, "height": 720, "fps": 30},   // numbers the spec gave you
  "arrival": {"kind": "streaming", "period_ms": "1000 / fps"},
  "constraints": {"deadline_ms": 50},
  "objective": {"weights": {"latency": 1.0}},
  "platform": {"id": "zynq7020_class", "path": "../../platforms/zynq7020_class.synthetic.json",
               "allowed_pes": ["cpu0", "cpu1", "fpga0"]},   // optional subset of the library entry
  "nodes": [
    {"id": "preprocess",
     "implementations": [                              // one entry per way to run it (multiple-choice)
       {"id": "sw", "pe_type": "cpu.cortex_a9", "cost": {"time_ms": "9.0e-6 * width * height"}},
       {"id": "hw", "pe_type": "fpga.xc7z020",
        "cost": {"time_ms": {"expr": "2.5e-6 * width * height", "unc": [0.9, 1.2]},
                 "resources": {"LUT": 4200, "DSP": 12}}}
     ]}
  ],
  "edges": [{"src": "capture", "dst": "preprocess", "bytes": "2 * width * height"}]
}
```

Rules:

- **Costs** are numbers, expression strings over `workload` keys, or `{expr, unc}`.
  - `unc = [lo, hi]` multiplies the nominal value, and requires `0 < lo ≤ 1 ≤ hi`.
  - Allowed in expressions: `+ - * / // % **` and `min max ceil floor log2 sqrt`. Nothing else.
- **Edges** carry bytes, not times. The partitioner computes transfer time from the platform's interfaces.
- **The graph must be a DAG** with unique node ids. Unknown keys are errors, so typos fail loudly.
- **Fixed upstream decisions:** `"pinned_pe": "cpu0"` on the node.
- **Per-task deadline:** `"deadline_ms"` on the node, measured from iteration start.
- **Arrival:**
  - `one_shot`: latency only.
  - `periodic`: each iteration must finish within `period_ms`.
  - `streaming`: pipelined, so only the initiation interval must fit in `period_ms`.
- **Alternatives:**
  - Per-task implementation bins go in `implementations`.
  - Whole-application alternatives are separate files that share `"alternative": {"group": "...", "variant": "..."}`.
  - Which of the two you will produce is still an open question.
- **Cross-frame state:** an edge with `"distance": k` (k ≥ 1) means frame i + k uses the output of frame i, e.g. `{"src": "track", "dst": "track", "distance": 1}` for tracker state. Such edges may form cycles, including self-loops. They affect streaming throughput only.
- **Streaming buffers:** `arrival.buffers` (default 2) is the number of frames of buffering per edge.
- **Hierarchy:** a node may carry `"subgraph": {nodes, edges}`. Phase 0 solvers treat the node as atomic and use its `implementations`.
- **Synthetic numbers:** put `synthetic` in `provenance.kind` and `.synthetic.` in the file name. A test enforces this for files in this repo.

## What you get back (`Result`)

| field | meaning |
|---|---|
| `mapping` | node → `{pe, impl}` |
| `schedule`, `comms` | start/finish per task and per transfer (ms), nominal costs |
| `feasible` | all constraints met at nominal costs |
| `metrics.latency_ms`, `energy_mj`, `ii_ms`, `avg_power_mw` | `{nominal, lo, hi, p05, p50, p95, n_samples}` from Monte Carlo |
| `metrics.feasible_fraction` | share of samples meeting all constraints |
| `diagnostics.loop` | `none`, `fb1` (fix the graph/spec: no mapping can work) or `fb2` (mapping/estimate problem) |
| `diagnostics.reasons` | human-readable sentences, e.g. "lower bound violates deadline_ms: 23.91 vs limit 20 … even at optimistic costs" |
| `diagnostics.lower_bound` | latency/II/energy bounds, violated constraints, `certain` (violated even optimistically), `structural` problems |
| `diagnostics.relaxation` | when the ILP was infeasible: the smallest change per constraint that makes it feasible |

Infinite values (for example the steady-state latency of a stream whose input is faster than the mapping can
sustain) are written as the JSON string `"Infinity"`.

For fb1, `reasons` and `lower_bound.violations` name the constraint and the gap. Those are the numbers to take back to the user, e.g. "50 ms is impossible on this board; ≥ 23.9 ms is the floor".

## Running it

```
python scripts/run_case.py examples/golden/collision_detection.synthetic.json --out out/
```
