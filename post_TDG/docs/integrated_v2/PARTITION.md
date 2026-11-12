# `partition()`: the combined partitioner (all data synthetic)

> Copied from the integrated package v2 (`../HWSW_integrated_v2/`). Here, `combo/partition.py` is ported as
> `post_tdg/portfolio.py` (decision D8); paths below refer to the v2 package.

One call that uses all three parts and returns the best plan found within a time budget:

```python
import sys; sys.path.insert(0, "combo")
from pathlib import Path
import partition as P
P.warmup()                                    # once: loads the GNN, compiles kernels, imports OR-Tools
rec = P.load_any(Path("HWSW_partitioner_handover_v1/examples/golden/collision_detection.synthetic.json"))
out = P.partition(rec, budget_s=1.0)          # Outcome: best Candidate + every candidate + the steps taken
result = P.to_result(rec, out)                # contract Result (validates against schemas/json/result.schema.json)
```

```
python combo/partition.py HWSW_partitioner_handover_v1/examples/golden/collision_detection.synthetic.json --budget 3 --out cd.result.json
python combo/partition.py HWSW_partitioner_handover_v1/data/labels/tier_a_v0_bus.synthetic.jsonl --index 5 --budget 0.1
```

**Input:** a `{graph, platform}` record (label records work as they are), or an IR graph JSON with
`platform.path` set.

**Model:** `models/gnn.pt`, the final Thor run-2 model (`PERSON3_RESULTS.md`). Override with `--model` or the
`COMBO_MODEL` environment variable.

**Budget:** everything after `warmup()` counts against it, including parsing, PEFT, the GNN, the search and
CP-SAT.

## What it does

| case | steps | evidence (`PERSON3_RESULTS.md`) |
|---|---|---|
| always first | compiled PEFT, as a safety net and as an extra start for the searches | 0.3–4 ms up to 500 tasks; optimal on some graphs where GA alone is not |
| remaining budget < 50 ms | SA from HEFT | at 10 ms the GNN can cost more than it gives on slower machines |
| ≤ 200 tasks | GA seeded with the GNN population and PEFT | 25–40 tasks: 1.053 → 1.027 at 100 ms, 1.043 → 1.023 at 1 s; 100 tasks: 1.321 → 1.052 at 1 s |
| > 200 tasks | GA from HEFT and PEFT | 500 tasks mixed, 1000 inconclusive |
| ≤ 40 tasks, budget ≥ 2 s | the GA step for 20% of the budget (0.1–1 s), then CP-SAT warm-started with every candidate | CP-SAT proves these optimal in well under a second |
| 41–100 tasks, budget ≥ 10 s | SA 1 s and GA + GNN 1 s, then CP-SAT warm-started with every candidate | Soham: SA warm start, mean 1.040, worst 2.16 → 1.32 |

- **Scoring:** every candidate is scored by `sim.evaluate`. The best feasible one wins, and a proven optimum wins
  a tie.
- **Fallback:** if Thakar's compiled model cannot represent a case, the partitioner uses the handover's HEFT,
  PEFT and MAGELLAN.

## Checks (x86, 2 cores, final model)

**Golden cases.** All three equal `examples/golden/expected`:

| case | makespan | budget | how it was found |
|---|---:|---:|---|
| collision detection | 27.023 ms | 100 ms | GA + GNN |
| toy fork-join | 6.0 ms | 10 ms | PEFT |
| WiFi TX | 7.468 ms | 3 s | proven optimal |

**The result JSON validates against `result.schema.json`.**

**25–40 tasks, 10 test graphs** (mean ratio to the optimum, with the maximum):

| budget | 10 ms | 100 ms | 1 s |
|---|---|---|---|
| mean (max) | 1.061 (1.546) | 1.001 (1.005) | 1.000 (1.003) |

**60–100 tasks, 4 graphs:**

| budget | 100 ms | 1 s | 12 s (CP-SAT) |
|---|---:|---:|---:|
| mean ratio | 1.157 | 1.132 | 1.053 |

**200 tasks, 1 s:** 224.2 ms, against 278.2 ms for PEFT.

**Limit: at 1,000 tasks the budget is not met.** Building the problem plus PEFT alone takes about 4 s on the
test machine, so budgets under about 10 s overrun. The plan returned is still valid.
