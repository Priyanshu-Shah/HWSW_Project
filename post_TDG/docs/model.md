# Partitioner model (Phase 0)

This file states exactly what the code computes, so results can be interpreted and challenged.

## Units

| quantity | unit | note |
|---|---|---|
| time | ms | `time_ms`, `deadline_ms`, `period_ms`, `ii_ms` |
| energy | mJ | `energy_mj`. 1 mW × 1 ms = 1 µJ = 0.001 mJ |
| power | mW | `power_mw`, `static_power_mw` |
| data | bytes | edge `bytes` |
| bandwidth | MB/s | 1 MB = 10⁶ bytes |
| setup latency | µs | interface `setup_us` |
| money | USD | `usd` |

## Pipeline

```
IR (TaskGraph) + Platform ──estimate.annotate──▶ Problem ──┬─▶ diagnostics.lower_bound
                                                           ├─▶ solvers.ilp / solvers.heuristics ─▶ mapping
                                                           └─▶ sim (single evaluator) ─▶ metrics + diagnostics ─▶ Result
```

## Bind / annotation (`estimate/`)

- Every cost is a `Param`: a number or an expression over workload parameters (`estimate/expr.py`: `+ - * / // % **`, `min max ceil floor log2 sqrt`), plus a multiplicative uncertainty interval `unc = (lo, hi)`.
- **Legal options** for a node are the (PE, implementation) pairs where the PE is in the case study's `allowed_pes` (action masking), `PE.type == implementation.pe_type`, and the node's `pinned_pe` (if any) matches.
- **Energy** per run is `energy_mj` if given. Otherwise it is `power_mw × time_ms / 1000`, and if neither is given it is 0.
- **Communication** between PEs *p ≠ q* uses the fastest interface connecting both for that transfer size: `setup_us/1000 + bytes/(bandwidth_mb_s·1000)` ms, with energy `bytes × energy_nj_per_byte × 1e-6` mJ. Same-PE communication is free. If no interface connects the two PEs, the combination is illegal.

## Simulator (`sim/`): the single evaluator

The simulator is a list scheduler over a ready list.

- **Priority:** upward rank (longest path to a sink, using the mapped task and communication times). A solver can override it with its own order; the ILP passes its start-time order.
- **Unified mode:** if the priority list also contains transfer items `"src->dst"`, transfers are list-scheduled as items of their own. A transfer is ready when its producer is placed and takes its interface in priority order; a task is ready when all its data has arrived. This reproduces any schedule whose per-resource orders are given (used for CP-SAT contention labels).
- **Exclusive PEs** (CPU cores, fixed-function accelerators) run one task at a time, non-preemptive. By default there is no gap insertion. With `insertion=True` a task may start in an earlier idle gap that is long enough; HEFT/PEFT use this, as in the published algorithms.
- **Spatial PEs** (FPGA fabric) give every task its own hardware, so tasks run concurrently. Their resources add up against `capacity`.
- **Contention:** transfers on a `shared` interface are serialised in the order they are scheduled (non-insertion). `contention=False` removes this, which matches the ILP model.
- **Energy** = task energy + transfer energy + Σ(static power of every *allowed* PE) × makespan.
- **Initiation interval (II)** estimate, the throughput bottleneck: the maximum of
  - busy time of each exclusive PE;
  - `ii_ms` of each spatial task;
  - busy time of each shared interface;
  - each transfer on a non-shared interface.
- **Constraints checked** (`sim/evaluate.py`):
  - end-to-end deadline;
  - per-task deadlines;
  - period: periodic arrival needs makespan ≤ period, streaming needs II ≤ period;
  - every PE capacity;
  - energy cap; average-power cap (energy / period, or energy / makespan for one-shot);
  - USD cap;
  - platform budgets.
- **Objective:** Σ wₖ · metricₖ / referenceₖ over latency, energy and usd. Each reference is that metric's lower bound, which makes the terms dimensionless; the usd reference is 1.

## Streaming throughput (`sim/throughput.py`)

- **Exact period:** the maximum cycle ratio (Σ durations / Σ tokens) of the mapped graph under self-timed execution, computed with Howard's policy iteration.
- **What the graph contains:**
  - task actors and transfer actors (cross-PE edges);
  - same-frame data edges (0 tokens) and buffer back-edges (`arrival.buffers` tokens, default 2);
  - cross-frame edges (`edge.distance` tokens), e.g. tracker state;
  - per exclusive PE, a cycle in the single-frame schedule's order;
  - per FPGA task, a self-loop of weight `ii_ms`;
  - per shared interface, a cycle in transfer order.
- **Steady-state latency** at the input period comes from a self-timed run (frame k released at k·period). It is `inf` if the input is faster than the period.
- **Relation to the old estimate:** the old `ii_ms` (max resource busy time) is the load lower bound. The exact period is ≥ it; it is larger with single buffering, recurrences, or a transfer chain that cannot overlap.
- **Uncertainty:** with the order frozen, the analysis is monotone, so `period_bounds_ms` at the interval corners are sound. Monte Carlo samples use the same frozen order.
- **Reporting:** for streaming arrival, `Result.metrics.throughput` holds the period interval, sound bounds, load bound, steady latency and critical cycle.
  - If the exact period exceeds `arrival.period_ms` while the load bound does not, a `period_ms.exact` violation is added, so `feasible` is false and the loop is fb2.
  - Search heuristics still use the cheaper load bound internally.
- **Checks:** hand-computed cases in `tests/test_throughput.py` (buffering, recurrence, corners, overload).

## Uncertainty and metrics (`sim/montecarlo.py`, `metrics/`)

- **Sampling:** every uncertain quantity draws a factor uniformly from its `unc` interval, independently. The default is 1,000 samples with a fixed seed.
- **Absolute metrics:**
  - nominal value (all factors = 1);
  - sample envelope (`lo`, `hi`), which approximates the bounded range;
  - p05 / p50 / p95;
  - `feasible_fraction`, the fraction of samples that meet every constraint.
  - Asymmetric intervals mean the nominal value can sit below p05.
- **Relative metrics:**
  - Candidates are ranked by (infeasible at nominal, nominal objective).
  - All candidates are evaluated on the *same* samples (common random numbers).
  - A candidate infeasible in a sample ranks after every feasible one in that sample.
  - Reported per ranking:
    - `pairwise_confidence[i][j]`, the fraction of samples where i is strictly better than j;
    - `order_preserved_fraction`;
    - mean Kendall τ-b against the nominal order.
  - Identical mappings from different solvers are merged.
- **Metric quality** (`metrics/fidelity.py`): MAPE against measured values, and Kendall τ-b against a measured ranking. Our τ-b matches `scipy.stats.kendalltau` to within 1e-16.

## Lower bound and feedback loop (`diagnostics/`)

- **Lower bound:**
  - every task on its fastest legal option, with zero communication and unlimited parallelism (critical path);
  - II bound = the largest single-task minimum;
  - energy bound = Σ minimum task energy + static energy over the bound latency.
  - It is computed at nominal costs, and again at the low end of every interval to decide "certain" violations.
- **Structural checks:**
  - a node with no legal option;
  - a node pinned outside the allowed PEs;
  - a node whose every option exceeds its PE's capacity;
  - nodes forced onto one PE whose minimum footprints together exceed its capacity.
- **Decision:**
  1. Structural problem, or the lower bound violates a constraint → **fb1** (task generation / spec). The reason says whether the violation holds even at optimistic costs or only at nominal costs.
  2. Otherwise, if the best mapping violates a constraint, or no mapping was found → **fb2** (Bind / Partition). It includes the violations and, when the ILP was infeasible, the minimal relaxation.
  3. Otherwise → **none**. The reason notes it if fewer than 100% of samples are feasible.

## ILP baseline (`solvers/ilp/`)

- Joint mapping + start times. It extends Niemann & Marwedel (https://ls12-www.cs.tu-dortmund.de/daes/media/documents/publications/downloads/1997-daes-journal.pdf) with communication delay that depends on the PE pair.
  - binary `x[i,k]` per legal option;
  - start time `s[i]` and makespan `C`;
  - communication `cm[e] ≥ c(p,q)·(X[i,p] + X[j,q] − 1)`;
  - big-M ordering binaries for independent node pairs that can share an exclusive PE;
  - capacity, deadline, period/II, energy, power and USD constraints;
  - objective = the same normalised weighted sum as the simulator.
- **Exact for:** the simulator *without* contention and without the interface term in the II. The returned mapping is always re-simulated with contention for the reported metrics.
- **Infeasibility:**
  - CBC and HiGHS have no IIS through PuLP. Instead we solve an *elastic* model with a non-negative slack on each relaxable constraint, minimising Σ slack/limit. Non-zero slacks are reported as `relaxation`: which constraint, and by how much.
  - Missing routes and missing options are hard, so they show up as structural.
- **Solvers:** HiGHS by default (via `highspy`); `--ilp-solver cbc` uses PuLP's bundled CBC.

**Scaling measured on this machine** (synthetic random DAGs, 1 CPU + 1 FPGA, 60 s limit, HiGHS):

| tasks | status | time (s) | ILP makespan vs greedy |
|---:|---|---:|---|
| 10 | optimal | 0.2 | equal |
| 20 | feasible (time limit) | 60 | −8% |
| 30 | feasible (time limit) | 60 | −8% |
| 40 | feasible (time limit) | 60 | −3% |

The big-M disjunctive formulation does not prove optimality beyond about 10–15 tasks on a single shared CPU. The "≈400 nodes" figure in the literature is for mapping-only formulations (see `docs/survey/partitioning_survey.md` §1).

## CP-SAT model (`solvers/cp/`)

- **Same semantics as the ILP**, so it is exact without contention.
- **Variables:** one optional interval per (task, option), sharing the task's start/end variables, with exactly one present.
- **Constraints:**
  - `AddNoOverlap` per exclusive PE;
  - precedence `s_j ≥ e_i + c(p,q)`, half-reified on "i on p, j on q"; missing routes are forbidden pairs;
  - capacity, deadline, period/II, energy, power and $ constraints as in the ILP.
- **Objective:** identical to the ILP's (float coefficients).
- **Units:** times are integer µs, energies integer nJ, resources ×1000. Rounding moves results by ≤ about 1 µs per task.
- **Solving:** warm-started from HEFT; 8 workers; reports status, best bound and gap.
- **Infeasibility:** answered by the ILP's elastic relaxation (`minimal_relaxation`).
- **Checks:** agrees with the ILP optimum on all goldens and on random DAGs (tests). Scaling vs the ILP: `docs/benchmarks/exact_scaling.md`.
- **Bus contention (`contention=True`; default in `cp.solve`).**
  - Each cross-PE transfer is an optional interval (present iff its endpoints sit on that PE pair) on its interface.
  - `AddNoOverlap` per shared interface; the consumer starts after the transfer ends; in streaming mode the bus busy time is ≤ the period.
  - The bus order is free.
  - The solution returns transfer start times, and `CpOutcome.priority()` gives the unified task + transfer list that the simulator replays exactly.
- **Scaling** (`docs/benchmarks/exact_scaling.md`; synthetic random DAGs, one seed per size, 30 s limit):
  - 1 CPU + FPGA: CP-SAT proves optimality up to **60 tasks in 0.2 s**. HiGHS proves only 10 tasks and finds no solution at 60.
  - 2 CPUs + FPGA: CP-SAT proves optimality up to 40 tasks (≤ 15 s); at 60 tasks it reaches a 0.6% gap at the time limit.
  - The HiGHS incumbents were sometimes worse than HEFT (40 tasks, 2 CPUs: 124.9 vs HEFT 112.6 vs optimum 96.0).
  - HEFT's gap to the optimum on these instances is 4–19%.
  - One seed per size, so treat these as indicative.

## Heuristic baselines (`solvers/heuristics/`)

- `greedy_eft`: HEFT-style. Nodes go in decreasing mean upward rank, and each takes the option with the earliest finish, skipping options that would exceed capacity.
- `all_sw`: greedy restricted to CPU options.
- `all_hw`: greedy restricted to non-CPU options; nodes without one stay on CPU.
- `sa`: simulated annealing over single-node option moves, scored by objective + 10 × Σ relative violation, with geometric cooling and a fixed seed.
- `exhaustive`: every combination. Used to cross-check the ILP in tests.
- `heft`, `peft`: published HEFT (Topcuoglu et al. 2002) and PEFT (Arabnejad & Barbosa 2014). Insertion-based.
  - A "processor" is a legal (PE, implementation) option; averages are taken over a node's options.
  - The average edge cost is the mean transfer time over distinct PE pairs.
  - Extension: options that would exceed a PE capacity are skipped unless none fits.
  - Checked against the paper's 10-task example: upward ranks and schedule length 80 reproduced exactly.
  - Known weakness: under tight FPGA resources the greedy choice can block a later, more valuable kernel. PEFT puts optical flow on the FPGA first in `collision_detection`, after which the CNN no longer fits the DSP budget (369 ms vs 27 ms).
- `kl`, `kl2`: Kernighan–Lin per Mann, Orbán & Farkas 2007, generalised to our model.
  - Moves = change one node's option; gain = change of the re-simulated objective (their Alg. 5).
  - Permissive cost: 4th-order penalty up to violation ratio q = 2, forbidden beyond.
  - Only feasible solutions can become the best.
  - LIFO tie-breaking within (1 − 0.95) of the best improvement.
  - Dynamic locking (`kl2`: ≤ 5 moves per node per pass, plus 10 random restarts).
- `tabu`: selection rule of Wiangtong, Cheung & Luk 2002. Take the best improving non-tabu move unpenalised; otherwise the best non-tabu move with a relative frequency penalty (0.005 × current cost per previous move of that node); if every move is tabu, the oldest one. Aspiration on a new global best. Tenure √(movable nodes) + 1 clamped to [7, 20]; 400 iterations.
- `magellan`: MAGELLAN-flat (Chatha & Vemuri 2001). Steepest descent: each step tries every option for every free task, re-completes all other free tasks with capacity-aware HEFT, and permanently fixes the best improving task. O(n²·K) completions. The paper's suitability metric is replaced by earliest finish (inferred).
- `ga`: genetic algorithm.
  - Chromosome = option index per node + random-key priorities (Wang et al. 1997, random-key form), decoded by the list scheduler.
  - Capacity repair written back to the genome.
  - Constraint-domination tournaments, (μ+λ) survival.
  - Seeded with HEFT, greedy EFT and all-SW. Defaults: pop 40, 60 generations, pc 0.9, pm 1/n.
- `nsga2()` (API, not a single-mapping solver): NSGA-II (Deb et al. 2002) with the same encoding and constraint-domination. Returns the deduplicated non-dominated set for chosen objectives (default latency, energy).
- Every heuristic returns a `Plan` (mapping, list order, insertion flag), so the simulator reproduces its schedule exactly.

## Mapping relaxation: lower bound (`solvers/exact/relaxation.py`)

- **What it keeps.** The mapping decisions x[t, k] (one option per task), with:
  - capacity per (PE, resource);
  - load: C ≥ Σ task times on each exclusive PE;
  - paths: f_b ≥ f_a + comm_ab + d_b, with comm_ab ≥ c(p, q)·(X_ap + X_bq − 1), which is exact for an integer mapping;
  - shared-bus load: C ≥ Σ transfer times routed over each shared interface;
  - release/tail cuts per exclusive PE: C ≥ r + load of the tasks that cannot start before r + their minimum tail, with r and tails from fastest options and no communication;
  - deadlines.
- **What it drops.** The order of tasks on a PE and of transfers on the bus. Every feasible schedule satisfies the constraints, so the optimum is a lower bound with or without contention.
- **Solved by HiGHS as a MILP.** The reported value is the *dual* bound (HiGHS stops within its relative gap tolerance). `integer=False` gives the weaker LP bound.
- **Tightness on the 176 test optima:** optimum / bound median 1.002, p90 1.033, max 1.27; 124 within 1%. The critical-path bound: 1.140 / 1.73 / 3.23. Median solve time 0.1 s.
- **Infeasibility:** an infeasible relaxation proves that no mapping meets the deadline / capacity.

## LBBD (`solvers/exact/lbbd.py`)

- **Master:** the mapping relaxation plus Benders cuts (HiGHS). **Subproblem:** CP-SAT with the mapping fixed and contention on, i.e. the exact schedule of that mapping.
- **Cuts:**
  - optimality no-good: C ≥ S_k − (S_k − LB_k)·(#tasks whose option differs from mapping k). S_k is the subproblem's lower bound minus 1 µs per task, for CP-SAT's µs rounding; LB_k is the master bound before the cut.
  - exclusion: a mapping that cannot meet the deadline gets "at least one task changes".
- **Budget:** the first master solve gets 40% of the budget, later ones ≤ 5 s each. Stops at a 0.1% gap, when no mapping is left, or at the time limit.
- **Role:** exact in principle, but no-good cuts remove one mapping at a time. On ≤ 40 tasks CP-SAT proves optimality far faster. LBBD is kept for large instances and as a reference.

## Relax-and-schedule (`solvers/exact/hybrid.py`)

For large instances:
1. The mapping relaxation runs for 40% of the budget; it gives a bound and a mapping.
2. That mapping is scheduled exactly, for 20%.
3. Full CP-SAT runs for the rest, hinted with that mapping and its start times.

The best simulator-checked schedule is returned, with bound = max(relaxation bound, CP-SAT bound). It is not a proof of optimality unless the gap closes; the gap is always reported. Results: `docs/benchmarks/exact_large.md`.

## Chain DP oracle (`solvers/exact/chain_dp.py`)

- Exact dynamic programme after Wu, Sun & Srikanthan 2012 and Shi et al. 2016: E(i, j, a) = min over t of E(i−1, t, a − a_ij) + c(t → j) + time_ij, which is O(n·K²·A).
- Scope: a single chain, latency objective, at most one finite (PE, resource) capacity with integer values. It raises `ValueError` otherwise.
- On a chain, execution is sequential, so the makespan is Σ task times + Σ cross-PE transfer times whether or not the bus is contended. The DP is therefore exact for the full model.
- Use: an oracle for the CP-SAT / MILP models on long chains. It matches CP-SAT on random 12-node chains, and solves a 1,000-node chain in about 2 s (`tests/test_chain_dp.py`).

## GCPS reference (`solvers/gcps/`)

- A reproduction of Zheng, Liang & Xiong 2021 in **their own model**, which is separate from ours:
  - one CPU, one FPGA with an area budget C = AP·Σ area, and one bus;
  - a binary partition y;
  - schedule length from their LSSP scheduler (`lssp.py`), which reproduces their Table 1 exactly.
- `instance.py`: TGFF 3.5 topologies (task degree 3, exactly N tasks) with their Table 2 cost ranges. Synthetic.
- `model.py`: the per-instance, unsupervised 2-layer GCN with cosine edge weights.
  - Loss Σ(t_h·x + t_s·(1 − x)); X' = 0.7·X + 0.3·HGP; a greedy cut line, then LSSP; their stopping rule.
  - Choices the paper leaves open are listed in the module docstring.
- `baselines.py`: greedy by HW gain per area (HGP), KL and SA on LSSP.
- Result (`docs/benchmarks/gcps_reproduction.md`): GCPS converges to the greedy-HGP solution. Its loss is minimised by all-HW, so the GCN output saturates. It is a reference baseline, not part of the pipeline.

## GNN partitioner (`solvers/gnn/`, needs the `ml` extra: CPU PyTorch)

- **Features** (`data.py`): index-free and normalised per graph, so one model applies to any size.
  - Task features (15): times, ranks, degrees, transfer load, plus graph context such as deadline / lower bound and FPGA pressure.
  - Option features (11): time ratios, PE kind, resource use / capacity.
  - Edge features (2): mean transfer time and bytes.
- **Model** (`model.py`):
  - 4 layers of directed message passing (separate predecessor and successor messages, residual, LayerNorm), hidden size 64.
  - Three heads: one logit per legal (PE, implementation) option; one priority score per task; one per edge (the transfer).
- **Training: imitation of CP-SAT bus-contention labels.** The loss is cross-entropy on the labelled option per task, plus a pairwise logistic rank loss on the labelled unified task + transfer order (NeuroSchedule style).
- **Decode** (`decode.py`):
  1. arg-max option per task;
  2. capacity repair (move the task that loses least probability off an over-full PE);
  3. a unified priority list of tasks and cross-PE transfers, sorted by score (the list scheduler starts the highest-priority *ready* item, so precedence is always respected);
  4. evaluation by the simulator.
  - `samples = K` adds Gumbel noise to logits and scores and keeps the best of greedy + K samples.
- Every mapping is scored by the same simulator as all other solvers. Results: `docs/benchmarks/gnn_imitation.md`.

## Not modelled yet

- hierarchical expansion of `subgraph`, which solvers currently treat as atomic;
- whole-graph alternatives across files, beyond the `alternative` tag;
- FPGA reconfiguration;
- DVFS;
- memory bandwidth contention on DDR;
- pipelined (software-pipelined) schedules for streaming;
- correlated uncertainty.
