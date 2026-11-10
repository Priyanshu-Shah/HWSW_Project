# Decisions log

One line per assumption or decision, with the reason. Newest at the bottom.

## 2026-10-14 — QuickEst intake (task 5)

- **Vendored layout:** `third_party/quickest/quickest-master/`, keeping the zip's top folder, so files stay byte-identical to upstream and are easy to diff.
- **Scope:** only `quickest-master.zip` was extracted. The other zips and folders in the root (HLSyn, DAC-SDC, bnn-fpga, facedetect, hoopa, e3s) are untouched, because they are out of scope for this task.
- **No pickles loaded:** `.pkl` files are only scanned with `pickletools`, because unpickling untrusted files can execute code and all the pickles can be regenerated from the CSVs.
- **Upstream scripts not run:** they are Python 2. RAE is reimplemented in `scripts/profile_quickest.py` with the same formula as `analyze.py:score_REA`, because the project targets Python 3.11+ and no Python 2 interpreter is installed. The metric is unchanged.
- **Interpreter:** the local Python 3.13 (has pandas and numpy) was used for profiling, because it is the only interpreter here with pandas. The code targets 3.11+.
- **Primary file:** `data1.csv`, not `data.csv`, because it is identical plus `CP_impl` (verified by an assert in the script).
- **Device mapping:** `Device_Index` → part is inferred from the `*_avail` columns (0=XC7V585T, 1=XC7Z020, 2=XC7A100T, 3=XC7K160T), because the release has no device names and the capacities match the Xilinx datasheets.
- **`CP_impl = -1` means missing:** treated as NaN (191 rows), because a negative clock period is physically impossible.
- **Timing-met ground truth:** `CP_impl ≤ Target_CP`, because the release has no WNS or met flag and `CP_impl` looks like the achieved period. Both HLS decision rules (with and without clock uncertainty) are reported.
- **RAE denominator:** the mean is taken over the evaluated subset (per device, per split), matching the paper's definition and `score_REA`.
- **Split protocol for the baseline:** a random 20% of rows, as in paper Sec. V, repeated 1,000× with seed 20181024 (the commit date), reporting median and p5–p95. One split is too noisy per device (~50 rows).
- **Generated vs hand-written docs:** numbers live in a generated `quickest_profile.md`, and `quickest.md` is the hand-written card that cites them, so stats can be regenerated without editing prose.
- **Tier B classification:** QuickEst is real tool output, used only to calibrate Bind, never as a case-study test set.

## 2026-10-14 — QuickEst follow-up (owner's answers)

- **Full dataset:** email the QuickEst authors for the full 65-design set and design names. Meanwhile, work with the 57 released designs. *(Owner decision.)*
- **LegUp files (`data2.csv`, `data3.csv`):** ignored. Device and provenance are unknown. *(Owner decision.)*
- **HLS reports at Bind time:** come from a pre-built library, with HLS run once per common algorithm and the results stored. Not run on demand. *(Owner decision.)*
- **Old tools and parts:** the Vivado 2017.1 / 7-series calibrator is acceptable for the first case study. Newer data (HLSyn, HLSFactory) comes later. *(Owner decision.)*

## 2026-10-14 — Phase 0 scaffold

- **Environment:** a project `.venv` on Python 3.13, because the global interpreters are shared, and 3.13 meets the 3.11+ rule.
- **ILP stack:** PuLP pinned `<4` + HiGHS (`highspy`) by default, with CBC as fallback. PuLP 4.0 changed the modelling API and dropped the bundled CBC. Variables are created with `prob.add_variable`, which works on 3.3 and 4.x.
- **No OR-Tools yet:** PuLP + HiGHS covers the brief's "open solver" with lighter dependencies. CP-SAT is the first candidate if the ILP scaling issue below needs fixing.
- **Top-level packages:** they match the brief's layout literally (`schemas/`, `estimate/`, `sim/`, …), and none clashes with a stdlib module name.
- **Units:** ms / mJ / mW / bytes / MB/s (10⁶ B) / µs setup / USD, encoded in field names so files are self-describing.
- **Costs are `Param`s:** a number or a safe expression over workload parameters, plus a multiplicative `(lo, hi)` interval. The brief requires costs as functions of workload, and intervals are the agreed uncertainty model.
- **Uncertainty sampling:** uniform and independent per quantity inside each interval. This is the simplest choice that matches "bounded intervals"; the open question on intervals vs distributions may change it.
- **Same-PE communication is free:** data stays in local memory or cache. Memory-bandwidth contention is not modelled yet.
- **Interface choice:** the fastest interface for the transfer's size. Deterministic, and it matches a designer choosing DMA for large blocks.
- **Contention model:** shared interfaces serialise transfers in scheduling order (non-insertion). It is simple, deterministic and conservative.
- **Exclusive vs spatial PEs:** CPUs and fixed accelerators run one task at a time; FPGA tasks run concurrently, limited by resource capacity. This is the standard HW/SW partitioning abstraction for fabric.
- **Static energy:** every *allowed* PE is charged for the whole makespan, because the board stays powered and this keeps the ILP linear.
- **II estimate:** the bottleneck of exclusive-PE busy time, spatial-task II and shared-interface busy time. It is a throughput bound, not a modulo schedule; true pipelined scheduling comes later.
- **Periodic vs streaming:** periodic needs makespan ≤ period; streaming needs II ≤ period, so the latency deadline is separate. This follows the brief's distinction between one-shot/periodic and pipelined video.
- **Objective normalisation:** each metric is divided by its lower bound (usd by 1), so weights compare dimensionless terms. The default weights are latency only until the open question on objective weights is answered.
- **ILP model scope:** exact without bus contention and without the interface term of the II, because modelling those would need many more binaries. Every ILP mapping is re-simulated with contention for reporting.
- **Infeasibility explanation:** an elastic relaxation (minimise Σ slack/limit) instead of IIS, because neither CBC nor HiGHS exposes IIS through PuLP. It reports which constraint fails and by how much, which is what fb2 needs.
- **fb1 rule details:** the decision uses the nominal lower bound, and the `certain` list records violations that persist at optimistic costs. That keeps the brief's rule while showing confidence.
- **Ranking:** infeasible-at-nominal candidates rank after feasible ones; per sample, infeasibility adds a 10⁶ penalty; identical mappings are merged. Otherwise an infeasible all-HW mapping ranked first, and duplicate solver answers looked like order flips.
- **Common random numbers:** all candidates in a ranking share the same cost samples, because uncertainty lives in the costs, not in the mapping.
- **Hierarchy:** `subgraph` is accepted by the schema, and solvers treat the node as atomic, so refinement can be added without breaking the contract.
- **Synthetic labelling enforced:** files with synthetic or mixed provenance must contain `.synthetic.` in their name, and a test checks this. This is the brief's rule, made mechanical.
- **Golden snapshots:** regression snapshots of our own outputs, explicitly not ground truth, so behaviour changes are visible in diffs.
- **Default demo platform `zynq7020_class`:** XC7Z020 (the Pynq-Z1 part), because it is consistent with the QuickEst calibration data the owner accepted for the first case study. The board choice itself is still open.

## Open items found in Phase 0 (not decided)

- **ILP scaling:** the joint mapping + scheduling MILP proves optimality at 10 tasks in 0.2 s, but only reaches "feasible" at 20–40 tasks within 60 s (docs/model.md). Options: CP-SAT (OR-Tools; a new dependency), a time-indexed model, or a mapping-only ILP for large graphs. This affects how large the ILP-labelled training graphs can be.
- **GA and Kernighan–Lin baselines:** not implemented yet.

## 2026-10-19 — Partitioning literature survey

- **Survey method:** six parallel research slices (exact, classical, metaheuristic, ML, streaming/real-time/uncertainty, datasets) over the local PDFs and the web. Every reference is tagged opened / metadata / from memory. This keeps unverified claims visible, per the "never present invented numbers as real" rule.
- **Output layout:** synthesis in `docs/survey/partitioning_survey.md`, full per-paper entries in `docs/survey/slices/`. The synthesis stays readable and the evidence stays auditable.
- **Correction:** arXiv 2205.06773 is Casini et al. (JSA 2022), not the WATERS 2019 challenge description, which is Rehm et al. (MICPRO 2021). Fixed in `PROJECT_OVERVIEW.md`.
- **Correction:** the "ILP practical up to ~400 nodes" figure in the brief holds only for the no-schedule binary model (Trindade & Cordeiro: 329 nodes). With scheduling, exact MILPs stall at 15–30 tasks in the literature, consistent with our HiGHS result.
- **Correction:** the GCN benchmark (Zheng et al. 2021) is a per-instance optimiser, not a trained model. Comparisons must count its per-instance training time.
- **Nothing implemented from the survey yet.** The recommendations in §7 of the survey await owner review (CP-SAT dependency, board access, baseline order).

## 2026-10-20 — Solver baselines and CP-SAT (survey §7.1–7.2, approved by owner)

- **OR-Tools added** (`ortools>=9.10`, 82 MB installed) for CP-SAT. The owner approved the survey roadmap ("go ahead"), which named this dependency.
- **CP-SAT uses the same semantics as the ILP** (no contention), so the two cross-check each other in tests. Integer units: µs, nJ, ×1000 resources, cents. Float objective coefficients are supported by CP-SAT 9.15.
- **Infeasibility reports reuse the ILP's elastic relaxation**, because CP-SAT has no slack-minimisation equivalent we need to maintain twice.
- **Simulator `insertion` flag (default off).** HEFT/PEFT are insertion-based in the papers, and our default stays non-insertion so existing results and snapshots are unchanged.
- **Heuristics return a `Plan`** (mapping, list order, insertion), so the single evaluator reproduces each heuristic's own schedule.
- **HEFT/PEFT adaptation:** a processor equals a legal (PE, implementation) option; averages are over options; the average edge cost is over distinct PE pairs; capacity check added (not in the papers). This is the minimal change that keeps the published algorithm recognisable.
- **List-scheduling ties:** ranks are rounded to 1e-9 before tie-breaking by graph order. Otherwise floating-point noise reorders equal-rank tasks (seen on the HEFT paper example).
- **KL = Mann et al. 2007 generalised:** re-simulation gain (their Alg. 5), permissive 4th-order penalty with q = 2, feasible-only best, LIFO window ε = 0.95. `kl` = classic one move per node per pass; `kl2` = dynamic locking (≤ 5) + 10 random restarts (Mann used 50; 10 keeps runtime low).
- **Tabu constants are ours** (tenure √n + 1, frequency weight 0.05), because Eles et al.'s exact constants could not be verified.
- **GA encoding:** option index + random keys (simplest valid-by-construction order), with capacity repair written back (Lamarckian). Erbas 2006 / Hou 2018 showed GAs without repair do not converge.
- **NSGA-II is exposed as an API returning a Pareto set**, not as a single-mapping solver, because a front does not fit the one-mapping `Result`. Ranking/reporting of fronts is future work.
- **ILP-scaling open item (2026-10-14) resolved for label generation:** CP-SAT proves optima up to 60 tasks (1 CPU + FPGA) and 40 tasks (2 CPUs + FPGA) within 30 s on synthetic DAGs, against 10 for the HiGHS MILP (`docs/benchmarks/exact_scaling.md`). LBBD stays the next step only if harder instances (tight resources, many implementations) need it.

## 2026-10-20 — Throughput evaluator, Tier A generator, first labels

- **Exact streaming period = maximum cycle ratio of the mapped graph** (Howard's policy iteration), with per-resource order frozen from the single-frame schedule. This is the standard SDF3-style evaluator recommended by the survey. Fixed order also makes corner bounds sound.
- **Contract additions (backward-compatible):** `edge.distance` (default 0) for cross-frame state, and `arrival.buffers` (default 2). Needed to express recurrences and finite buffering. Old files validate unchanged.
- **Cross-frame transfers are not put on the shared-bus order cycle** (only same-frame transfers are scheduled in a frame). This is a small under-count of bus load; noted in the code.
- **Feasibility for streaming:** search heuristics keep the cheap load bound; results add a `period_ms.exact` violation when only the exact period fails. Exact evaluation inside every search move would be costly; reporting must be exact.
- **Infinity in JSON** is the string "Infinity" (pydantic `ser_json_inf_nan="strings"`), so results with unsustainable streams round-trip.
- **Tier A v0 sources:** E3S (SW times, additive log model; type ids verified global across files), QuickEst (resources and their uncertainty), HLSyn v20 (implementation fronts). HW speedups, edge sizes, interfaces and time intervals are SYNTHETIC assumptions, listed in the data card.
- **HLSyn data was extracted to the session scratchpad only** (not the repo). Only derived statistics are committed, with CC BY-SA attribution in the fits file.
- **Labels:** CP-SAT, 10 s limit, HEFT warm start, canonical tie-break (fixed optimal makespan, then minimise Σ start times and option indices). Removes the label noise from non-unique optima reported by the DS3 imitation-learning paper. Latency-only objective in v0 for that reason.
- **Labels are for the no-contention model**; each record also stores the with-contention replay, so the gap is measured, not hidden.
- **CP-SAT rounding fix:** positive durations round up to ≥ 1 µs. Sub-µs E3S-derived tasks had rounded to 0 and shared CPU slots, giving a 1.7% decoder gap on Tier A seed 166; a regression test covers it. Max replay excess after the fix is 0.095%.
- **Canonicalisation fallback:** if the tie-break pass finds nothing within the time left, the proven-optimal first solution is kept (seeds 50 and 87 had lost their labels).
- **Tier A v0 label run:** 200 instances, 178 optimal / 22 infeasible (deadline sampled below what any mapping can reach), 35 s total. Summary in `docs/data_cards/tier_a_labels_summary.md`.

## 2026-10-20 — Shared-bus contention in the exact model

- **CP-SAT models shared-bus contention** with optional transfer intervals and `NoOverlap` per shared interface, so labels match the simulator. `cp.solve` defaults to it; `solve_cp` defaults to no contention so the ILP cross-checks stay like-for-like.
- **The simulator gained a unified task + transfer list mode, and labels carry transfer priorities.** With task priorities alone, transfers are booked at consumer placement, which missed contention optima by up to 49% (25/176 instances). The GNN decoder must therefore output edge (transfer) priorities as well as task priorities and PE/implementation choices.
- **Bus gap insertion** was also added to the simulator's insertion mode. HEFT/PEFT use it; it does not by itself reproduce CP bus orders.
- **Primary Tier A labels are now bus-contention labels** (`tier_a_v0_bus.synthetic.jsonl`: 176 optimal, 24 infeasible, all exactly replayable and feasible on the full model). The no-contention set is kept for comparison. No files were deleted.
- **Infeasibility reports under contention** still come from the no-contention ILP relaxation. When that relaxation needs no change, the result says the infeasibility is due to bus serialisation, rather than reporting an empty relaxation.

## 2026-10-21 — Papers and TGFF supplied by the owner (`Drops/`)

- **Read by three parallel agents; extracts in `docs/survey/slices/07–09`.** Survey §9 lists the corrections: Mann B&B ≤ 600 nodes; Shi 2016 is a weighted sum; the Wu 2013 "28%" is a best case against their own heuristic; GCPS's loss has no area or communication term, and "18×" is vs KL at 1,000 nodes.
- **Tabu search revised to Wiangtong 2002's selection rule:** improving non-tabu move unpenalised; otherwise a relative frequency penalty (0.005 × current cost per visit, from their Q = 80 ns on ~15k ns makespans); oldest move if all are tabu. Tenure √n + 1 clamped to [7, 20]; 400 iterations. Our earlier constants were guesses, and the absolute 0.05 penalty was negligible.
- **MAGELLAN-flat added:** steepest descent with permanent fixing; free tasks are re-completed by capacity-aware HEFT (`heft_complete`). The paper's suitability metric is defined only in an earlier paper, so EFT stands in for it (inferred).
- **TGFF 3.5 vendored** (`third_party/tgff`, all rights reserved: not to be redistributed outside the group) and built at `-O0` (`-O2` segfaults). The build reproduces the shipped example byte for byte. A `.tgff` parser (`datagen/tgff_io.py`) also reads E3S, whose files use lowercase "to" and per-row comment lines.
- **The `Drops/` folder is left as supplied** (nothing moved or deleted).

## 2026-10-22 — GCPS reproduction, chain DP oracle, GNN imitation (next steps after `Drops/`)

- **PyTorch is CPU-only, in an optional `ml` extra** (owner's choice, 2026-10-22). The core pipeline does not need it, and the GNN tests skip without it.
- **GCPS is reproduced in its own CPU + FPGA + bus model** (`solvers/gcps/`), not ours. Only that way can its Table 1 and Table 4 numbers be compared. LSSP reproduces Table 1 exactly; it is the check that the reproduction is faithful.
- **GCPS choices the paper leaves open** (cosine weights mapped to [0, 1], dropout, pre-training steps, cut-line rule) are listed in `solvers/gcps/model.py`. Only a *strict* improvement resets the stopping counter: with ties counting, 20-node runs never stopped, which contradicts the paper's ~0.5 s.
- **`torch.set_num_threads(1)` for GCPS.** The default thread count made its tiny per-step ops 11× slower.
- **GCPS result: the paper's ordering is not reproduced.** GCPS ties greedy-HGP from N = 100 and loses to it below. Cause, measured: the loss has its minimum at all-HW, so the GCN output saturates and X' ranks like HGP (Kendall τ 0.998 at N = 1000). GCPS stays a reference baseline only (`docs/benchmarks/gcps_reproduction.md`).
- **GCPS baselines KL/SA run under a 10 s wall-clock budget.** The paper's KL took ~150 s at N = 1000; a budget keeps the run bounded. Their AR at N ≥ 200 is therefore a lower bound on what unbudgeted KL/SA would reach.
- **Chain DP oracle (Wu 2012 / Shi 2016)** is exact for our full model on chains: execution is sequential, so contention cannot change the makespan. It is restricted to one capacity dimension, which is where the DP is polynomial. It is used to test the exact solvers on long chains, not as a partitioner.
- **GNN design: imitation of exact labels, not GCPS-style unsupervised training.** The GCPS loss has no area or communication term (above), whereas CP-SAT labels carry them.
  - The model outputs option logits *and* task + transfer priorities. Transfer priorities are required to reproduce bus-contention optima (2026-10-20).
  - Loss = cross-entropy + pairwise rank loss (NeuroSchedule). Decode = arg-max, then capacity repair, then the simulator; best-of-K Gumbel sampling is optional.
- **DS3 (GPL-3) is used only as an external cross-check of the simulator.** It is cloned into the session scratchpad and is not part of the repo or the workflow, per the owner's instruction of 2026-10-22.

## 2026-10-24 — GNN v0 training fixes and first diagnosis

- **Option features were unbounded.** `time / task's fastest option` reached 2.8e10 on E3S-derived tasks with near-zero options, so training cross-entropy spiked to the hundreds. Fix: the denominator is floored at 1e-3 × graph time scale, and unbounded ratios (times, transfer loads, deadline / lower bound) are log1p-compressed. All features are now ≤ ~10.
- **Training uses 1 thread, Adam `foreach`, and 8 graphs per step (gradient accumulation).** With 4 threads a step on these small graphs was 2× slower (same effect as GCPS). Accumulation amortises the optimiser step: 46 → 24 ms per graph.
- **Diagnosis (short run: 300 graphs, 8 epochs, 62 validation graphs):**
  - **The priority heads work.** The label's mapping with GNN priorities gives a median of 1.057 × optimum.
  - **The option head does not yet beat "pick the fastest option".** Its per-task accuracy is ~0.70; the fastest option alone scores 0.70. Independent arg-max decoding gives a median of 1.48.
  - **GNN-guided list scheduling reaches HEFT's level, but not better.** EFT restricted to each task's top-2 GNN options, ordered by GNN priority: median 1.16 vs HEFT 1.14.
  - Same-type PE symmetry is *not* the cause: CPUs are typed per speed class, and half of the platforms have one CPU.
- **The full 30-epoch run of 2026-10-24 was stopped at epoch 10** (unbounded features; ~3.8 h of the wall clock was host idle time). No model was saved.

## 2026-11-04 — Person 1 (exact methods and bounds), first pass

- **The critical-path lower bound is loose.** The optimum is a median 14% above it and up to 5.8× (all labels). On 60–100 tasks the median is 55%. It ignores PE load, transfers and the bus.
- **New mapping relaxation (`solvers/exact/relaxation.py`).** It keeps the mapping and drops ordering:
  - per-PE load;
  - paths with PE-pair-linked transfer times;
  - shared-bus load;
  - capacity and deadlines;
  - release/tail cuts per exclusive PE.
  It is valid with or without contention. It is solved as a MILP (HiGHS), and HiGHS's *dual* bound is reported, because HiGHS stops within its own relative gap.
  - **Validity:** on 176 proven optima, no bound exceeds the exact replay optimum.
  - **Tightness:** median optimum/bound 1.002, p90 1.033, max 1.269; 124/176 within 1%. Critical path: 1.140 / 1.727 / 3.23; LP version: 1.023 / 1.188 / 2.15.
  - **Speed:** median 0.11 s.
- **Compare labels with `replay.makespan_with_contention_ms`, not `label.makespan_ms`, below 0.2%.** CP-SAT rounds durations to whole µs. Seed 194's label is 5.962 ms, but the exact optimum is 5.96342, which the relaxation bound hits exactly.
- **LBBD (`solvers/exact/lbbd.py`).**
  - Master: the relaxation plus no-good cuts. Subproblem: CP-SAT with the mapping fixed, bus contention on. Deadline-infeasible mappings get exclusion cuts.
  - Cut value: the subproblem's lower bound minus 1 µs per task, to absorb µs rounding.
  - Tolerance 0.1%, set by the µs rounding.
  - Each master solve is capped (default 5 s), because HiGHS cannot close the master on some 70–100-task instances.
- **LBBD result on small graphs: correct, but not competitive at proving optimality.** 13/37 proven in 30 s, against all of them for CP-SAT (median 0.25 s). No-good cuts remove one mapping at a time, and many mappings tie in the relaxation. CP-SAT stays the exact solver for ≤ 40 tasks.
- **On 60–100 tasks, decomposition wins on the upper bound.** Scheduling the relaxation's mapping exactly beat CP-SAT's 20 s solution on all four hard instances:
  - 5003: 37.4 → 20.7 ms;
  - 5019: 600.1 → 494.9 ms;
  - 5027: 70.1 → 54.0 ms;
  - 5009: 173.3 → 169.0 ms.
  Gaps fell from 18–52% to 4–17%. Next to test: the relaxation mapping as a hint for full CP-SAT.
- **Benchmarking paused on 2026-11-04.** The machine was ~90% busy inside the WSL VM (`vmmemWSL`, Docker Desktop started 13:27). Wall-clock comparisons from that period contradict earlier runs: CP-SAT found no solution in 30 s where it had found one in 20 s. They are not used. Next: CP-SAT's deterministic time limit for reproducible budgets.

## 2026-11-06 — Exact methods on the 60–100-task set; a CP-SAT false optimum

- **Equal-budget comparison (20 s each, 30 instances, `docs/benchmarks/exact_large.md`).**
  - CP-SAT (HEFT warm start): best on 17/29, p90 gap 23%, max 35%.
  - Relax-and-schedule (`solvers/exact/hybrid.py`): best on 14/29, p90 11%, max 22%.
  - LBBD: best on 9/29, p90 10%, max 18%.
  - The decomposition methods win where CP-SAT's gap is large (5003, 5004, 5009, 5019, 5027). CP-SAT wins where it is already near-optimal.
  - Best known after all runs: **19 of 30 optimal within 0.1%** (was 13). References with mappings and orders are in `data/benchmarks/large_best_known.synthetic.json`. 5020 still has no schedule; 5014 is infeasible.
- **CP-SAT 9.15 returned a false OPTIMAL with start-time hints.**
  - On seed 5003, full CP-SAT hinted with a mapping *and* start times returned OPTIMAL at 22.315 ms in about 4 s, five times out of five. Yet 20.164 ms is feasible in the same model (found under another hint; simulator-checked).
  - The false optimum disappears with presolve off, or without the start hints. Our model is not at fault.
  - **Rule:** hint the mapping only, unless the mapping is fixed. `solve_cp` carries the warning; relax-and-schedule now hints the mapping only.
  - The superseded runs are kept as `hybrid_v1` in the runs file. Their schedules count; their bounds do not. With start hints the method had been best on 21/29.
- **Other uses are unaffected:**
  - labels, plain CP-SAT and LBBD subproblems use mapping-only hints or none;
  - no label marked optimal is beaten by any method;
  - a scan of all 120 run records found exactly one bound above a known schedule: the v1 5003 record.
- **The report raises an error instead of silently capping a bound above a known schedule.** The silent cap was what had hidden the v1 problem in the first table.
- **Scaling study (`docs/benchmarks/exact_scaling_tier_a.md`; 36 Tier A instances, 10–60 tasks, loose / tight FPGA, 20 s; 152 s of solver time).**
  - CP-SAT *without* the bus proves every instance, at a median ≤ 0.5 s per size.
  - *With* bus contention it proves everything up to 40 tasks (< 1 s). It is unproven on 3 of 12 at 50–60 tasks; there the relaxation bound beats CP-SAT's own bound.
  - Contention raises the optimum itself only slightly: median 1.000, max 1.046 over 32 pairs. So the cost of contention is in proving, not in the value.
  - The MILP proves ≤ 30 tasks here (one unproven at 30). This is better than the earlier random-DAG benchmark, because these deadlines are loose.
  - No relaxation bound exceeds a proven optimum.
- **CP-SAT µs rounding can make a feasible instance "infeasible".** Seed 8100 (10 tasks, tight FPGA): the deadline equals the all-SW makespan to the ns, and that is also the bus optimum. CP-SAT rounds each duration to whole µs and reports infeasible; with +10 µs it is optimal at 32.74 ms.
  - Checked all 152 infeasible labels: 141 are proven infeasible by the relaxation, which uses exact durations. Of the other 11, only one becomes feasible with 1 µs per task of slack, and its schedule still misses the original deadline in the simulator.
  - **No label is affected.** Instance generators should keep deadlines more than n µs away from the optimum, or the solver should be given that slack.
- **Chains (`docs/benchmarks/chain_dp_vs_cpsat.md`, exact DP as reference).**
  - CP-SAT does not prove even 50-task chains in 30 s; its bound is 0.70–0.81 of the optimum.
  - Its solution drifts from 1.000 (50 tasks) to 1.186 (1,000 tasks), worse than HEFT's 1.142 at 1,000.
  - The DP is exact in ≤ 1.6 s. The mapping relaxation is exact on chains (no ordering freedom), within HiGHS's 0.01% tolerance, taking 0.1 s at 50 tasks and 34 s at 1,000.
  - **CP-SAT optimality claims are a ≤ 40-task tool. Beyond that, use the relaxation bound or structure-specific oracles.**

## 2026-11-07 — Exact component finished for hand-over

- **Single entry point `solvers.exact.solve_exact(problem, T, warm_starts)`** with `certify` and `run` (pipeline `Result`).
  - Order: relaxation bound and infeasibility proof → caller's warm starts + HEFT → chain DP → CP-SAT with a mapping-only hint (all of the budget for ≤ 40 tasks, 40% above that) → relax-and-schedule for the rest.
  - The best simulator-checked plan wins. Bounds get a margin of 1 µs per task; a bound above a known schedule is dropped and reported, never trusted.
  - Integration guide: `docs/handover/exact_component.md`.
- **Bus release/tail cuts** added to the relaxation, treating the shared bus as one more exclusive resource.
- **HiGHS feasibility tolerances are set to 1e-9.** With the bus cuts at default tolerances, HiGHS presolve returned 232.37 ms on test seed 195, where 227.40 ms is feasible: an invalid bound. Presolve off or tight tolerances both give 227.398.
  - Re-validated on 1,043 proven optima (176 test, 854 training, 13 large): zero violations.
  - Tightness, optimum / bound: median 1.001; worst 1.226 (test), 1.209 (training), 1.049 (large).
- **The relaxation now bounds the makespan for any objective,** and can drop deadlines (giving the floor value) and use any cost view. LBBD and relax-and-schedule still require a latency-only objective.
- **`lower_bound(problem, strong=True)`, opt-in:** the relaxation bound at nominal and optimistic costs upgrades the deadline / periodic-period checks. This is a sharper fb1, with a better "≥ X ms" number.
  - Result schema: new optional field `LowerBound.latency_ms_relaxation`.
  - Off by default, so the golden snapshots are unchanged; `solvers.exact.run` turns it on.
- **The CP-SAT µs-rounding false infeasibility is handled:** `solve_exact` re-solves with n µs of deadline slack and checks against the exact deadline (test: scaling seed 8100).
- **Mapping-only hints verified safe.** Four optimality claims made under mapping hints (5008, 5009, 5017, 5019) were re-checked with unhinted CP-SAT. Three were proven again at identical values; 5009 came within 0.04%, with a bound below the claim.
- **The two CP-SAT "overruns" (seed 1092: 847 s; 5005: 1,500 s) were laptop Modern Standby.** Windows Kernel-Power events 506/507 cover both windows. CP-SAT respects its limit (5005 rerun: 24.1 s for a 24 s limit).
- **Large set after `close_large_gaps.py` (60 s per open instance, warm-started): 22/30 optimal, 1 infeasible, 7 open** (gaps 0.12–8.2%, plus 5020 at 34%). 5020 now has a schedule meeting its deadline (337.3 ms vs 342). 5009 is proven optimal at 149.13 ms, down from 157.4.
