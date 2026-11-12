# Person 3 (ML methods): final results

> Copied from `../HWSW_integrated_v2/PERSON3_RESULTS.md`. Paths (`combo/`, `training/`) refer to that package;
> the model is `models/gnn_r2_all_base_s1.pt` here (identical file to v2 `models/gnn.pt`).

All data are **synthetic** (Tier A generator v0, platform `zynq7020_class`). All schedules are scored with the
handover's `sim.evaluate` (list scheduling, shared bus, capacity checks). Ratios are makespan ÷ reference, where
the reference is the CP-SAT optimum for test graphs, the best known schedule for the 60–100-task set, and the best
schedule any run found for the unlabelled 100–1000-task graphs. "Mean" is Thakar's metric: per graph, the median
over seeds; then the mean over graphs. p-values are two-sided sign tests on per-graph wins and losses.

Question from `team_split.md`: *can a learned method beat the best heuristic at the same runtime, and does it
generalise to larger graphs?*

**Short answer.**
- **On its own, no.** The GNN imitates exact solutions, and on its own it still loses to PEFT and MAGELLAN.
- **As the starting point for Thakar's GA, yes, at the same runtime.** It significantly improves the best search
  method from 100 ms to 1 s on graphs of 25–40 and 60–100 tasks.
  - With 1 s on 25–40 tasks it reaches 1.023 of the optimum, against 1.043 from HEFT's start, 1.068 for MAGELLAN
    and 1.212 for PEFT.
- **Size generalisation.** It helps up to 200 tasks, which is 5× the largest training graph.
  - At 100 tasks: 1.052 at 1 s against 1.321 from HEFT's start.
  - At 500 tasks the results are mixed; at 1,000 tasks they are inconclusive.

## 1. Model and training

- **Model** (`solvers/gnn`, unchanged from the handover):
  - A 4-layer, 64-unit edge-aware GNN over tasks, options and edges.
  - Three heads: an option head (one choice per task), plus task priority and transfer priority heads.
  - Loss: cross-entropy on the optimal option, plus a pairwise ranking loss on the optimal order.
- **Training data:**
  - The handover's 854 optimal labels.
  - 17,409 new optimal 10–40-task labels and 1,148 new optimal 40–60-task labels. These were generated on the
    AGX Thor with the handover's `make_labels.label_one`; seeds 20000+ and 60000+ keep them disjoint from
    test, train and large.
- **Validation:** 300 graphs of 10–40 tasks plus 100 graphs of 40–60 tasks, all held out. The epoch kept is the
  one with the best mean greedy ratio on validation.
- **Training budget:** 60 min, one CPU core per model, on the Thor.
- **Delivered model:** `models/gnn.pt` = `r2_all_base_s1`.

| training set | graphs | val ratio (greedy) | val option accuracy (10–40 / 40–60) |
|---|---:|---:|---:|
| subset | 2,000 | 1.796 | 0.711 / 0.678 |
| subset | 6,000 | 1.606 | 0.765 / 0.715 |
| all 10–40 | 17,969 | 1.542 | 0.782 / 0.731 |
| all 10–40 + 40–60 (3 seeds) | 19,028 | **1.500 – 1.505** | 0.776 / 0.73 |
| same, 128 × 6 (only 11 epochs in 60 min) | 19,028 | 1.557 | 0.769 / 0.727 |
| same, option-loss weight ×3 | 19,028 | 1.526 | 0.779 / 0.735 |

- **More data helps steadily and has not saturated.** The full-data models stopped at the 60-minute cap after
  about 23 epochs, still improving.
- **Adding 40–60-task graphs helps at every size.** On test, greedy improves from 1.650 to 1.430 on 25–40 tasks
  and from 1.632 to 1.433–1.489 on 60–100 tasks.
- **The settings changes did not beat more data.** The bigger model was undertrained, and the reweighted loss did
  not win.
- **Option accuracy rose from about 0.73 (run 1) to 0.78.** The fastest-option baseline is about 0.70. Choosing
  each task's option independently is still the main limit (see §5).

## 2. GNN alone vs heuristics (unseen test graphs)

| | 10–24 tasks | 25–40 | 60–100 |
|---|---:|---:|---:|
| GNN greedy, run 1 (754 graphs) | 1.598 | 1.957 | 1.843 |
| GNN greedy, final model | 1.274 | 1.434 | 1.433 |
| GNN best of 64 samples, final | 1.073 | 1.167 | 1.294 |
| optimal mapping + GNN priorities | 1.053 | 1.166 | 1.402 |
| HEFT | 1.441 | 1.609 | 1.269 |
| PEFT | 1.172 | 1.203 | 1.201 |
| MAGELLAN | 1.034 | 1.082 | 1.144 |

- **Best-of-64 sampling** beats HEFT everywhere and PEFT up to 40 tasks. It still loses to MAGELLAN, which takes
  13–200 ms against roughly 5–10 ms for the GNN.

## 3. GNN + search at equal runtime (the deliverable "vs HEFT / MAGELLAN at equal runtime")

**Setup** (`combo/heur_combo.py`; Thor run 2 with the final model, 3 seeds):
- **Starts compared:**
  - `heft`: Thakar's default start.
  - `gnn`: the GNN's greedy mapping.
  - `gnnpop`: one GNN pass, up to 256 sampled mappings scored with Thakar's compiled evaluator, and the best 8
    used as GA seeds.
- **The GNN's time is inside each budget.**
- Every plan is re-verified, and Thakar's simulator agreed with `sim.evaluate` on all 12,760 plans of this run
  (0 disagreements).

**25–40 tasks (89 graphs)**

| budget | GA from HEFT | GA + GNN (best start) | p | best other method at this budget |
|---|---:|---:|---:|---|
| 10 ms | 1.210 | **1.104** (gnnpop) | 1e-5 | tabu + GNN 1.081 (not significant vs tabu from HEFT, 1.139); SA from HEFT 1.115 |
| 100 ms | 1.053 | **1.027** (gnnpop) | 0.014 | SA 1.076, tabu 1.095 (HEFT start) |
| 1 s | 1.043 | **1.023** (gnn) | 0.005 | SA 1.061, tabu 1.069 (HEFT start) |
| 10 s | 1.035 | **1.017** (gnn) | 0.68 | — |

For reference, the constructive heuristics on the same graphs:

| method | ratio | time |
|---|---:|---:|
| MAGELLAN | 1.068 | 25 ms |
| PEFT | 1.212 | 0.3 ms |
| HEFT | 1.391 | 0.2 ms |

**60–100 tasks (27 graphs)**

| budget | GA from HEFT | GA + GNN population | p |
|---|---:|---:|---:|
| 100 ms | 1.155 | **1.093** | 0.019 |
| 1 s | 1.106 | **1.073** | 0.5 |
| 10 s | 1.088 | **1.065** | 0.63 |

For reference, the constructive heuristics on these graphs:

| method | ratio | time |
|---|---:|---:|
| MAGELLAN | 1.144 | 191 ms |
| PEFT | 1.265 | — |
| HEFT | 1.348 | — |

**Confirmed independently on Thakar's laptop.** That run used the weaker run-1 model, 5 seeds and all 203 test
graphs:
- 25–40 tasks: GA 1.073 → 1.043 at 100 ms (p 3×10⁻⁶) and 1.050 → 1.028 at 1 s (p 0.017).
- 10–24 tasks: 1.026 → 1.006 at 100 ms (p 0.0015).

**Where it does not help.**
- SA and tabu at 100 ms or more: they converge to the same answers whatever the start.
- At 10 ms on a slower machine, the GNN's own 5–15 ms costs more than it gives (laptop: SA and tabu lose,
  p ≤ 0.012).

## 4. Size generalisation (100–1000 tasks, 5 graphs per size, unlabelled)

**Setup:**
- Graphs from the same generator, seeds 90000+ (`data_big/`).
- One common reference per graph: the best schedule found by any run with either model.
- 2 seeds; GA results shown. "r1" = run-1 model, "final" = this model.

| tasks | budget | GA from HEFT | GA + GNN (r1) | GA + GNN (final) |
|---:|---|---:|---:|---:|
| 100 | 1 s | 1.321 | 1.141 | **1.052** |
| 100 | 10 s | 1.238 | 1.127 | **1.040** |
| 200 | 1 s | 1.076 | **1.041** | 1.048 |
| 200 | 10 s | 1.066 | **1.023** | 1.038 |
| 500 | 1 s | **1.204** | 1.198 | 1.266 |
| 500 | 10 s | 1.093 | 1.072 | **1.065** |
| 1000 | 10 s | 2.017 | 1.031 | 1.712 |

For reference, the constructive heuristics:

| tasks | MAGELLAN | PEFT |
|---:|---|---:|
| 100 | 1.244 (665 ms) | 1.506 |
| 200 | 1.075 (3.6 s) | 1.247 |
| 500 | — | 1.269 |
| 1000 | — | 2.287 |

**At 1,000 tasks:**
- Even HEFT's construction takes about 1.2 s, so no search finishes within 1 s.
- At 10 s the differences between runs are larger than the effect, so these numbers are noisy.

**The GNN alone degrades with size.** At 100 tasks the final model's greedy mapping scores 1.88, against 3.22 for
the run-1 model and 2.11 for HEFT. **Used as a starting point for search**, the final model generalises well up
to 200 tasks.

## 5. Limits and what would come next

1. **The per-task option head is the bottleneck.** Accuracy is 0.78, and the optimal mapping scheduled with the
   GNN's priorities is still at 1.17 on 25–40 tasks. The next steps on the list would address it:
   - several optimal or near-optimal mappings per graph as training targets (many graphs have more than one);
   - DAgger-style sequential decoding, where each task's option is chosen given the others.
2. **Training was cut off by the time cap.** Longer training or the 128 × 6 model trained to convergence are
   cheap things to try on the Thor.
3. **The GCPS / TGFF benchmarks (20–1,000 tasks, different cost model) were not run.** A converter to the IR
   format is needed. Size generalisation was tested on Tier A graphs of up to 1,000 tasks instead.
4. **The published ML baseline (GCPS) collapses to greedy** in our reproduction (handover `reference_results/`).
   The GNN + GA result above is the ML comparison that stands.

## 6. Where everything is

| item | file |
|---|---|
| final model | `models/gnn.pt` (`r2_all_base_s1`; architecture in `HWSW_partitioner_handover_v1/solvers/gnn`) |
| combined partitioner | `combo/partition.py`, documented in `PARTITION.md` |
| GNN + search experiments | `combo/heur_combo.py`, `combo/bridge.py` (incl. `GnnSampler`) |
| exact warm-start experiment | `combo/exact_combo.py` (Soham's v2 protocol changes are in his copy) |
| training, labels, evaluation | `training/gen_labels.py`, `training/gnn_run.py`, `training/gnn_run2.py`, `training/summarize2.py` |
| 100–1000-task graphs | `data_big/tier_a_big.synthetic.jsonl` (`combo/make_big.py`) |
| raw summaries of every run | `results/` (Thor runs 1–2, Thakar rounds 1–2, Soham round 1) |
| new exact labels (≈ 820 MB) | on the Thor, in `THOR_RUN_v1/out/labels/` (not in this zip) |
