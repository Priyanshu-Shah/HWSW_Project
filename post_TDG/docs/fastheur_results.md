# Person 2 results: heuristics and metaheuristics baseline

All instances are **synthetic** (Tier A generator, GCPS/TGFF graphs converted to our model). Scores follow `format/model.md`
(list scheduling, shared bus, capacity checks). Every plan reported here was re-simulated by the pure-Python reference
simulator and had to satisfy all PE capacities and reproduce the claimed makespan exactly (`hwsw/bench.py: run_task`).

## 1. Answer: best non-learned method per runtime budget

Metric: makespan / reference. Per instance the median over seeds, then the **mean over instances** (lower is better, 1.000 = as
good as the best known). Reference per instance = min(proven CP-SAT optimum, CP-SAT best-found label, best makespan found by any
run of any method). Tier A references are therefore real optima or CP-SAT bests. GCPS has no external optimum, so GCPS ratios
mean "distance to the best our own runs found".

Short version (details and ties are in the tables below; the top 2-3 methods are often statistically indistinguishable):

* **10 ms:** simulated annealing for most sizes. Tabu for GCPS N = 20-50. At 60-100 tasks and at N = 1000 nothing beats the
  fallback/HEFT by much in 10 ms (PEFT, MAGELLAN and SA are within noise of each other).
* **100 ms:** tabu (GCPS N <= 100) or SA (GCPS N >= 200, Tier A 60-100). MAGELLAN or GA on 10-40 tasks.
* **1 s:** tabu for GCPS N = 50-200 (N = 20 is a tie at 1.000), SA for GCPS N >= 500. GA / NSGA-II (best makespan of the archive) on Tier A.
* **10 s:** GA / NSGA-II on Tier A (10-100 tasks), NSGA-II / GA / tabu on GCPS N <= 500, SA at N = 1000.
* HEFT is 20-50% above the best at every size. PEFT beats HEFT on Tier A but not on GCPS. GCLP and MIBS are far behind on makespan
  by design (they minimise area under a deadline, see `docs/ALGORITHMS.md`).

**The bar for Person 3** is the "best method" column below (same instances, same budgets, same metric).
Per-class files: `results/wide/tables/ranked_tables.md`, `winners.csv`, `rank_*.csv`. Plot: `results/wide/anytime.png`.

### Wide run (headline): all solvable instances, 5 seeds (0-4)

176 Tier A test instances (87 with 10-24 tasks, 89 with 25-40), 27 Tier A large (60-100 tasks; the 3 without a CP-SAT
solution are excluded), 60 GCPS (10 per size).

### Core run: 30 seeds (0-29), sampled instances

12 Tier A test, 6 Tier A large, 3 GCPS per size. This is the full 30-seed protocol of the brief, but with few instances, so a
method that is best here is not necessarily best overall. Use the wide run for conclusions and the core run for seed-level
statistics.

#### wide

| class | n inst | budget | best method (mean ratio) | runner-up | HEFT | PEFT | MAGELLAN |
|---|---:|---:|---|---|---:|---:|---:|
| TierA 10-24 | 87 | 10 ms | **sa** 1.034 | magellan 1.039 | 1.225 | 1.150 | 1.039 |
| TierA 10-24 | 87 | 100 ms | **ga** 1.023 | sa 1.023 | 1.225 | 1.150 | 1.036 |
| TierA 10-24 | 87 | 1000 ms | **nsga2** 1.006 | ga 1.008 | 1.225 | 1.150 | 1.036 |
| TierA 10-24 | 87 | 10000 ms | **nsga2** 1.003 | ga 1.006 | 1.225 | 1.150 | 1.036 |
| TierA 25-40 | 89 | 10 ms | **sa** 1.108 | kl1 1.138 | 1.391 | 1.212 | 1.191 |
| TierA 25-40 | 89 | 100 ms | **magellan** 1.074 | ga 1.075 | 1.391 | 1.212 | 1.074 |
| TierA 25-40 | 89 | 1000 ms | **ga** 1.051 | nsga2 1.063 | 1.391 | 1.212 | 1.068 |
| TierA 25-40 | 89 | 10000 ms | **ga** 1.042 | nsga2 1.042 | 1.391 | 1.212 | 1.068 |
| GCPS N20 | 10 | 10 ms | **tabu** 1.020 | sa 1.059 | 1.486 | 1.504 | 1.279 |
| GCPS N20 | 10 | 100 ms | **tabu** 1.016 | sa 1.024 | 1.486 | 1.504 | 1.243 |
| GCPS N20 | 10 | 1000 ms | **nsga2** 1.000 | ga 1.005 | 1.486 | 1.504 | 1.243 |
| GCPS N20 | 10 | 10000 ms | **nsga2** 1.000 | ga 1.003 | 1.486 | 1.504 | 1.243 |
| GCPS N50 | 10 | 10 ms | **tabu** 1.104 | sa 1.175 | 1.471 | 1.478 | 1.410 |
| GCPS N50 | 10 | 100 ms | **tabu** 1.019 | sa 1.079 | 1.471 | 1.478 | 1.207 |
| GCPS N50 | 10 | 1000 ms | **tabu** 1.007 | ga 1.012 | 1.471 | 1.478 | 1.162 |
| GCPS N50 | 10 | 10000 ms | **nsga2** 1.002 | ga 1.005 | 1.471 | 1.478 | 1.162 |
| TierA 60-100 | 27 | 10 ms | **peft** 1.227 | magellan 1.231 | 1.309 | 1.227 | 1.231 |
| TierA 60-100 | 27 | 100 ms | **sa** 1.151 | tabu 1.163 | 1.309 | 1.227 | 1.167 |
| TierA 60-100 | 27 | 1000 ms | **ga** 1.098 | sa 1.102 | 1.309 | 1.227 | 1.115 |
| TierA 60-100 | 27 | 10000 ms | **ga** 1.085 | tabu 1.086 | 1.309 | 1.227 | 1.115 |
| GCPS N100 | 10 | 10 ms | **sa** 1.302 | tabu 1.389 | 1.511 | 1.562 | 1.470 |
| GCPS N100 | 10 | 100 ms | **tabu** 1.053 | sa 1.139 | 1.511 | 1.562 | 1.421 |
| GCPS N100 | 10 | 1000 ms | **tabu** 1.012 | ga 1.042 | 1.511 | 1.562 | 1.309 |
| GCPS N100 | 10 | 10000 ms | **tabu** 1.005 | ga 1.006 | 1.511 | 1.562 | 1.309 |
| GCPS N200 | 10 | 10 ms | **sa** 1.390 | magellan 1.447 | 1.468 | 1.492 | 1.447 |
| GCPS N200 | 10 | 100 ms | **sa** 1.233 | tabu 1.323 | 1.468 | 1.492 | 1.441 |
| GCPS N200 | 10 | 1000 ms | **tabu** 1.031 | sa 1.102 | 1.468 | 1.492 | 1.374 |
| GCPS N200 | 10 | 10000 ms | **ga** 1.006 | tabu 1.007 | 1.468 | 1.492 | 1.271 |
| GCPS N500 | 10 | 10 ms | **sa** 1.376 | magellan 1.381 | 1.386 | 1.409 | 1.381 |
| GCPS N500 | 10 | 100 ms | **sa** 1.305 | tabu 1.371 | 1.386 | 1.409 | 1.378 |
| GCPS N500 | 10 | 1000 ms | **sa** 1.165 | tabu 1.254 | 1.386 | 1.409 | 1.374 |
| GCPS N500 | 10 | 10000 ms | **tabu** 1.015 | ga 1.045 | 1.386 | 1.409 | 1.332 |
| GCPS N1000 | 10 | 10 ms | **magellan** 1.267 | sa 1.268 | 1.268 | 1.804 | 1.267 |
| GCPS N1000 | 10 | 100 ms | **sa** 1.247 | tabu 1.266 | 1.268 | 1.286 | 1.266 |
| GCPS N1000 | 10 | 1000 ms | **sa** 1.157 | tabu 1.235 | 1.268 | 1.286 | 1.265 |
| GCPS N1000 | 10 | 10000 ms | **sa** 1.016 | tabu 1.037 | 1.268 | 1.286 | 1.260 |

#### core

| class | n inst | budget | best method (mean ratio) | runner-up | HEFT | PEFT | MAGELLAN |
|---|---:|---:|---|---|---:|---:|---:|
| TierA 10-24 | 6 | 10 ms | **magellan** 1.001 | sa 1.003 | 1.109 | 1.058 | 1.001 |
| TierA 10-24 | 6 | 100 ms | **ga** 1.001 | magellan 1.001 | 1.109 | 1.058 | 1.001 |
| TierA 10-24 | 6 | 1000 ms | **ga** 1.000 | nsga2 1.000 | 1.109 | 1.058 | 1.001 |
| TierA 10-24 | 6 | 10000 ms | **ga** 1.000 | nsga2 1.000 | 1.109 | 1.058 | 1.001 |
| TierA 25-40 | 6 | 10 ms | **tabu** 1.098 | kl1 1.126 | 1.491 | 1.328 | 1.245 |
| TierA 25-40 | 6 | 100 ms | **magellan** 1.055 | tabu 1.098 | 1.491 | 1.328 | 1.055 |
| TierA 25-40 | 6 | 1000 ms | **magellan** 1.055 | kl2 1.057 | 1.491 | 1.328 | 1.055 |
| TierA 25-40 | 6 | 10000 ms | **tabu** 1.043 | kl2 1.043 | 1.491 | 1.328 | 1.055 |
| GCPS N20 | 3 | 10 ms | **tabu** 1.000 | sa 1.059 | 1.474 | 1.516 | 1.179 |
| GCPS N20 | 3 | 100 ms | **sa** 1.000 | tabu 1.000 | 1.474 | 1.516 | 1.168 |
| GCPS N20 | 3 | 1000 ms | **ga** 1.000 | sa 1.000 | 1.474 | 1.516 | 1.168 |
| GCPS N20 | 3 | 10000 ms | **ga** 1.000 | sa 1.000 | 1.474 | 1.516 | 1.168 |
| GCPS N50 | 3 | 10 ms | **tabu** 1.086 | sa 1.175 | 1.462 | 1.460 | 1.378 |
| GCPS N50 | 3 | 100 ms | **tabu** 1.008 | sa 1.077 | 1.462 | 1.460 | 1.148 |
| GCPS N50 | 3 | 1000 ms | **tabu** 1.002 | ga 1.011 | 1.462 | 1.460 | 1.096 |
| GCPS N50 | 3 | 10000 ms | **tabu** 1.002 | nsga2 1.004 | 1.462 | 1.460 | 1.096 |
| TierA 60-100 | 6 | 10 ms | **peft** 1.280 | sa 1.302 | 1.426 | 1.280 | 1.331 |
| TierA 60-100 | 6 | 100 ms | **kl2** 1.179 | sa 1.192 | 1.426 | 1.280 | 1.262 |
| TierA 60-100 | 6 | 1000 ms | **tabu** 1.145 | kl2 1.146 | 1.426 | 1.280 | 1.197 |
| TierA 60-100 | 6 | 10000 ms | **tabu** 1.109 | kl2 1.133 | 1.426 | 1.280 | 1.197 |
| GCPS N100 | 3 | 10 ms | **sa** 1.303 | tabu 1.371 | 1.487 | 1.534 | 1.466 |
| GCPS N100 | 3 | 100 ms | **tabu** 1.032 | sa 1.146 | 1.487 | 1.534 | 1.403 |
| GCPS N100 | 3 | 1000 ms | **tabu** 1.011 | ga 1.044 | 1.487 | 1.534 | 1.344 |
| GCPS N100 | 3 | 10000 ms | **ga** 1.007 | tabu 1.008 | 1.487 | 1.534 | 1.344 |
| GCPS N200 | 3 | 10 ms | **sa** 1.403 | magellan 1.464 | 1.484 | 1.506 | 1.464 |
| GCPS N200 | 3 | 100 ms | **sa** 1.227 | tabu 1.333 | 1.484 | 1.506 | 1.458 |
| GCPS N200 | 3 | 1000 ms | **tabu** 1.040 | sa 1.108 | 1.484 | 1.506 | 1.374 |
| GCPS N200 | 3 | 10000 ms | **ga** 1.009 | tabu 1.012 | 1.484 | 1.506 | 1.265 |
| GCPS N500 | 3 | 10 ms | **sa** 1.382 | magellan 1.385 | 1.392 | 1.414 | 1.385 |
| GCPS N500 | 3 | 100 ms | **sa** 1.306 | tabu 1.376 | 1.392 | 1.414 | 1.381 |
| GCPS N500 | 3 | 1000 ms | **sa** 1.164 | tabu 1.260 | 1.392 | 1.414 | 1.377 |
| GCPS N500 | 3 | 10000 ms | **tabu** 1.022 | ga 1.041 | 1.392 | 1.414 | 1.331 |
| GCPS N1000 | 3 | 10 ms | **magellan** 1.258 | kl1 1.258 | 1.259 | 1.930 | 1.258 |
| GCPS N1000 | 3 | 100 ms | **sa** 1.243 | tabu 1.257 | 1.259 | 1.277 | 1.257 |
| GCPS N1000 | 3 | 1000 ms | **sa** 1.151 | tabu 1.227 | 1.259 | 1.277 | 1.256 |
| GCPS N1000 | 3 | 10000 ms | **sa** 1.022 | tabu 1.037 | 1.259 | 1.277 | 1.251 |

## 2. How far from optimal? (honest gap)

* Tier A 10-24 tasks at 10 s: mean ratio 1.003 (best method). Tier A 25-40: 1.042. Tier A 60-100: 1.085.
* A few instances stay far from CP-SAT for every method. In the core sample `tierA_v0_s104` is 1.157 x optimum for the best of
  all 30 seeds x 13 methods. Large `s5028` is 1.098 x optimum (proven), and `s5011` is 1.107 x the CP-SAT best-found.
  Cause: our metaheuristics search over mappings (and priorities for GA/NSGA-II) and use the *default* decoder, which cannot
  choose transfer order or use bus gaps the way CP-SAT does. A unified-priority decoder already replays CP-SAT labels exactly
  (176/176) but was not made a search space.
* GCPS N = 1000 at 10 s: SA 1.016, tabu 1.037. These are distances to the best we found, not to an optimum.

## 3. Protocol, as run

* Methods: HEFT, PEFT, GCLP, MIBS (each also as `_lat`, latency target), MAGELLAN, KL1, KL2, tabu, SA, GA, NSGA-II.
* Budgets 10 ms, 100 ms, 1 s, 10 s. One 10 s run per (instance, method, seed) is read at the four budgets (anytime trace).
  **SA is run separately per budget** because its cooling schedule depends on the budget. Deterministic methods run once
  (wide) or 3 times (core, median time).
* Every method starts from the same free all-software fallback, held until the method's first solution is ready.
* Statistics (Erbas et al. style): per instance the 30 (core) or 5 (wide) seeds give a sample, and the instance score is its
  median. Per class and budget we report mean rank, the Wilcoxon signed-rank test of every method vs the leader over instances,
  the Vargha-Delaney A12 (> 0.5 means the method is worse than the leader) and, per instance, a two-sided rank-sum test over
  seeds (alpha 0.05, no multiple-test correction). Deterministic methods have one value per instance in the wide run, so their
  per-instance test shows 0/0 (no power). Use the Wilcoxon column for them.
* Seeds: stochastic methods use seeds 0-29 (core) and 0-4 (wide), passed to `numpy.random.default_rng`. Raw per-run records
  (traces, plans, orders) are in `results/raw/` and `results/raw_wide/`.
* Hardware and timing: 10 worker processes on a 12-thread laptop with 7.3 GB RAM, one thread per worker. Absolute times are
  indicative. Budgets are wall-clock, and SA overshoots its budget by a median of 9% at 10 ms (max 22%).

## 4. Deviations from the brief and known limitations

1. **Instance subset in the core run** (12/176, 6/27, 3/10 per GCPS size), because 30 seeds x 4 budgets x 13 methods on every
   instance would take about 14 h on this machine. The wide run covers all solvable instances but with 5 seeds, not 30.
2. **NSGA-II uses latency-area, not latency-energy**, on Tier A and GCPS. Those datasets contain no energy data (Tier A data card:
   "No energy or power in v0"). Latency-energy (`second="energy"`) is implemented and tested on the golden cases that have
   energy. On `wifi_tx` NSGA-II recovers all 12 points of the exact front (hypervolume ratio 1.000). On `collision_detection`
   the exact front is a single point (`results/nsga2_energy_golden.md`). In the single-objective ranking NSGA-II is read as
   "best makespan on its archive". Its front quality is in `results/*/tables/nsga2_hypervolume.md`.
3. **Reconstructed algorithms.** GCLP, MIBS and MAGELLAN contain details we inferred (global-criticality formula, extremity
   thresholds, MAGELLAN's suitability metric replaced by earliest finish time). KL, GA and NSGA-II are adapted (KL samples 256
   moves per step on large neighbourhoods). Faithful, adapted and inferred parts are documented per algorithm in
   `docs/ALGORITHMS.md`.
4. **GCLP and MIBS on GCPS degenerate to a HEFT-like greedy.** GCPS has no deadline, so the target becomes the critical-path
   bound, global criticality is always above 0.5, and each node takes its minimum finish time. Their GCPS rows are not a test of
   deadline-driven partitioning. On Tier A (with deadlines) they behave as designed and are weak on makespan.
5. **Unequal budget use.** KL1 stops at convergence (median use of a 10 s budget about 0%), KL2 uses about 24%, and MAGELLAN
   stops early on small graphs. They hold their result for the rest of the budget. The tables have a "budget used" column.
6. **No hyper-parameter tuning on `labels/train`.** Parameters (GA mu = lambda = 40, SA temperatures, tabu tenure, KL penalty)
   were fixed by design from the cited papers and checked in pilots on GCPS. The HEFT model was calibrated against the *test*
   labels (192/200 stored HEFT makespans reproduced after allowing insertion into idle bus gaps). That aligned semantics rather
   than tuned parameters, but it did use test data, so it is disclosed here.
7. **Infeasible instances** (24 test, 1 large, plus 2 large without a CP-SAT solution) are excluded from all tables.
8. Nominal costs only: no uncertainty intervals, streaming throughput or reconfiguration.
9. The `TGFF_3.5` generator was not used. GCPS graphs are the ones shipped in `benchmarks/gcps_tgff/`.

## 5. Validation evidence (all in `tests/`, 32 pass)

HEFT reproduces the published 10-task example (ranks and makespan 80). Compiled HEFT/PEFT equal the Python reference, including
fixed-option completions. The compiled evaluator equals the reference simulator. Unified replay reproduces 176 CP-SAT labels
exactly. The GCPS conversion reproduces the reference schedule length. PEFT reproduces the documented 369 ms failure on
`collision_detection`. Golden optima are reached.

## 6. Hand-offs

* **Person 1:** warm starts are in `results/wide/warm_starts/*.json` (263 files, one per instance: the best plan over all runs,
  with option per task, priority order, makespan, method, seed, budget and the proven optimum if known). All solvable large
  (60-100 task) instances are included. We still need from Person 1: better references (proven optima or tight lower bounds) for
  the unproven large graphs, and lower bounds for GCPS.
* **Person 3:** compare against the "best method" rows above on the same instances and budgets. Confirm the instance IDs and the
  hardware/thread definition with us. `results/wide/tables/all_values.csv.gz` holds every per-seed value.

## 7. Reproduce

```
python -m pytest tests -q
bash scripts/run_all.sh        # core run (about 1.5 h)
bash scripts/run_wide.sh       # wide run (about 1.7 h)
python scripts/analyze.py results/raw/*.jsonl --ref-extra results/raw_wide/*.jsonl --out results/core --warm
python scripts/analyze.py results/raw_wide/*.jsonl --ref-extra results/raw/*.jsonl --out results/wide --warm
python scripts/demo_nsga2_energy.py
```
