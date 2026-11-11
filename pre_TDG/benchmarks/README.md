# benchmarks

Ten small tasks that exercise the whole LLM pipeline (Phase 7 of `../implementation.md`). Each is
run with a scripted user and the result is scored against a hand-made reference graph.

| id | task | exercises |
|---|---|---|
| t01 | object detection (YOLOv8n), 720p @ 30 fps, zcu104 | NN templates, streaming deadline |
| t02 | Sobel edges, 1080p @ 60 fps, kv260 | loop nest, sliding window |
| t03 | MobileNetV2 classification, one-shot, pynq_z2 | NN, no deadline |
| t04 | 256-tap FIR on 48 kHz audio, pynq_z2 | tiny nodes: hardware should not help |
| t05 | FMCW radar range FFT + CFAR, zcu104 | template + loop nest mix |
| t06 | AES-128 CTR on a 1 Gb/s stream, kv260 | table lookups, line-rate deadline |
| t07 | batch of 8 fp32 1024x1024 matmuls, zcu104 | gemm, no deadline |
| t08 | JPEG encoding, 1080p @ 30 fps, zcu104 | branchy entropy coding (`control_heavy`) |
| t09 | 12-state Kalman filter at 1 kHz, hard 1 ms, pynq_z2 | tight deadline, tiny matrices |
| t10 | video stabilization, 720p @ 30 fps, zcu104 | long pipeline, fan-out |

## Files

```text
make_benchmarks.py     defines every task and reference; run it after editing
tasks/<id>.json        task text, scripted answers, expected spec values, node roles
reference/<id>.tdg.json  reference graph-only TDG (valid; tests check it scores 1.0 against itself)
results/<time>_<model>_<think>.md / .json   one file pair per `hwsw bench` run
```

A reference is *one* reasonable graph, not the only right answer: node names, splits and exact
sizes may differ. Scores tell where to look, not a pass/fail.

## Run

```text
.venv/bin/hwsw bench --boards-dir tests/fixtures/platforms            # all 10, about 1 hour
.venv/bin/hwsw bench t02 t04 --model gpt-oss:20b --think low          # some tasks, another model
```

Each task's conversation, spec, graph and `score.json` are in `runs/<time>_bench_<id>/`.
After changing the scoring, `hwsw rescore benchmarks/results/<run>.json` scores a run again from
those files (no LLM).

## Scores

| column | meaning |
|---|---|
| spec ok | of 4: platform, arrival kind, timing hardness, deadline within 5% |
| valid | the generated TDG passes the validator |
| node F1 | generated nodes matched one-to-one to reference nodes: a word of the name starts with one of the reference node's `roles` |
| edge F1 | dependencies between matched nodes that the reference also has. Both graphs are first reduced to their matched nodes: a reaches b if there is an edge, or a path only through unmatched nodes, so an extra buffer step does not count as wrong |
| ops ratio | total ops (custom_hls `ops_estimated`), generated / reference |
| node ops 2x, bytes 2x | share of matched nodes / direct edges whose ops / bytes are within 2x of the reference |
| all-SW ratio, best ratio | latency bounds (`hwsw bounds`), generated / reference |
| rounds, questions, LLM calls, min | effort |

## Known limits of the references

Costs come from custom_hls, uncalibrated, on the fixture boards. With it: t07's 4 MB fp32 matrix
does not fit on chip (no tiling), so it has no hardware option; t01's convolutions gain only ~5x,
so even the best case misses 33 ms; t04 and t09 are too small for hardware to help (intended).
