# TDG: contract status and open items

The TDG folder holds the interface between the teams: the schema, its examples and the
decisions behind them. The work plan for producing these files is in `../pre_TDG/implementation.md`.

## Files in the contract (D5)

| File | Written by | Read by | Defined by | Status |
|---|---|---|---|---|
| `tdg.json` (graph part) | pre_TDG | pre_TDG (custom_hls) | `schemas/tdg.schema.json` | v0.2 written |
| `tdg.costed.json` (same + `costs`) | pre_TDG (custom_hls) | post_TDG | `schemas/tdg.schema.json` | v0.2 written |
| `library/boards/*.json` | library team | pre_TDG, post_TDG | `library/README.md` | expectations written; `platform.schema.json` by library team |
| `spec.json` | pre_TDG | pre_TDG only | `spec.schema.json` (in pre_TDG) | internal; changes freely |

**Change rule:** any change to a contract file (fields, names, units, meaning) bumps
`schema_version` (or, for board files, is announced), updates `schemas/` and the examples, and
is raised with the project lead **before** merging, so the other teams can be informed.

## Items to clarify in schema v0.2 (not blocking)

Resolving any of these changes the contract, so each goes through the change rule.
1. **Deadline vs period:** `deadline_s > period_s` cannot be met with sequential frames (D3).
   The validator warns (R11); the LLM sets `deadline_s = min(period_s, max_latency)` for hard
   real-time tasks.
2. **Opaque cost in two places** (`opaque.sw_time_s` and the cost map). Proposal: custom_hls
   copies it into `costs`; the partitioner reads only `costs`.
3. **Comm on HW↔HW and SW↔SW edges:** state "0 in v0.2" explicitly in the conventions.
4. **Timing hardness** (hard / soft / none) is in `spec.json` only. Does the post_TDG team
   want it in the TDG?
5. **Details added when writing the formal schema** (not in the original draft; need sign-off):
   JSON instead of YAML; `schema_version` is a string; ids match `^[A-Za-z_][A-Za-z0-9_]*$`;
   unknown fields are rejected; `period_s` required for periodic/streaming and forbidden for
   one_shot; `ops_claimed` required for loop_nest nodes; dtype list
   (int8, uint8, int16, int32, fp16, fp32); loop-nest body op list; `control.control_heavy` flag;
   template op list and the parameters of each op (incl. conv2d `padding`, `groups`, `batch`);
   `costs.edges[].time_s` is an estimate object.
6. **Board file format** (`library/README.md`) is new: the post_TDG team reads
   `fpga.capacity` and `interfaces` from it.

## Post-MVP

Feedback loops (granularity split/merge, infeasible → change what is computed), layer-exact NN
expansion (ONNX), pipelined throughput (`period_s` used by the partitioner), energy objectives.
