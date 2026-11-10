# Algorithms: what is implemented, what is faithful, what is inferred

Person 2 task list from `HWSW_partitioning_data_v1/README.md`: HEFT, PEFT, GCLP, MIBS, MAGELLAN, Kernighan–Lin (KL1, KL2),
tabu search, simulated annealing, genetic algorithm, NSGA-II, evaluated with 30 seeds and equal budgets.

All methods share one problem model (`hwsw/instance.py`), one scoring rule (`hwsw/sim.py`, the semantics of
`format/model.md`) and one anytime protocol (`hwsw/algos/base.py`). A solution is a **Plan**: one (PE, implementation)
option per task, an optional list order, and a flag for insertion scheduling. Every plan reported anywhere in this
folder was re-simulated by the pure-Python reference simulator and had to (a) respect every PE capacity and (b) give
exactly the makespan the algorithm claimed (`bench.run_task`).

Legend: **F** faithful to the published algorithm, **A** adapted to our model (multi-PE, multiple implementation bins,
shared bus), **I** inferred (the paper's detail was not available to us, so the choice is ours and is stated).

## Shared model and scoring

* Costs: nominal values only (the exact labels ignore uncertainty intervals too).
* Exclusive PEs (CPU cores) run one task at a time. Spatial PEs (FPGA) run mapped tasks concurrently; their resources add
  up against `capacity` (LUT/FF/DSP/BRAM18K, or `area` for GCPS). Capacity is a **hard constraint** for every reported
  solution.
* Cross-PE transfers use the fastest interface for their size; shared interfaces serialise transfers (bus contention).
* Objective: makespan (latency). `deadline_ms` is reported ("deadline met") but is not part of the search objective,
  except inside GCLP/MIBS, which are deadline-driven by design (see below).
* Two schedule decoders exist, both from `format/model.md`:
  * *default* (metaheuristics): ready list, priority = upward rank of the mapped graph (or explicit random keys for
    GA/NSGA-II), no insertion;
  * *insertion* (HEFT, PEFT, MAGELLAN completions): tasks may use idle gaps of an exclusive PE, and transfers may use idle
    gaps of a shared interface.
  The second point (gaps on the bus too) is not written in `model.md`; we found it by testing: with it our HEFT reproduces
  the stored HEFT makespan of 192 of 200 test instances exactly, without it 73.

## Constructive heuristics

### 1. HEFT — Topcuoglu, Hariri & Wu 2002 (F, A)
Upward rank `rank_u = w̄ + max(c̄ + rank_u(succ))`, tasks in decreasing rank, each task takes the option with the earliest
finish time (insertion based). Adaptation: a "processor" is a legal (PE, implementation) option; `w̄` is the mean time over
a task's options; `c̄` is the mean transfer time over ordered pairs of distinct connected PEs. Extension: options that would
overflow an FPGA are skipped unless none fits. Ties: lower energy, then lower option id.
*Check:* the 10-task example of the paper gives the published ranks and schedule length 80 (`tests/test_sim_and_heft.py`).
A compiled version (`hwsw/heftk.py`) is used in the benchmark; a test asserts it is identical to the Python reference.

### 2. PEFT — Arabnejad & Barbosa 2014 (F, A)
Optimistic cost table `OCT(t, p) = max_succ min_pw (OCT(s, pw) + w(s, pw) + c̄ if pw ≠ p)`, `rank_oct` = mean over options,
ready-list scheduling by `rank_oct`, option minimising `EFT + OCT`. Same adaptation as HEFT.
*Check:* hand-computed OCT values; it also reproduces the failure noted in `format/model.md` (369 ms vs 27 ms on
`collision_detection`, where PEFT puts optical flow on the FPGA and the CNN no longer fits).

### 3. GCLP — Kalavade & Lee 1994 (A, I)
Global criticality / local phase. Implemented as: static priority = HEFT upward rank; GC from a schedule estimate where every
unmapped node is provisionally software (target = deadline); nodes with a very large speedup and small area are *hardware
extremities*, nodes with ≈ no speedup are *software extremities*; otherwise GC > 0.5 → minimise finish time, else minimise
area (software). Two classes only: software (any CPU option) or the *fastest* hardware option. GC is refreshed every
`max(1, n/200)` nodes for large graphs. **I:** the exact GC formula, extremity thresholds (speedup ≥ 8 and area ≤ 25 % of the
binding capacity; speedup ≤ 1.1) and the omission of repellers are ours. `gclp_lat` replaces the deadline by the
critical-path lower bound so that it minimises latency.

### 4. MIBS — Kalavade & Lee 1997, implementation-bin selection (A, I)
Same skeleton as GCLP, but all implementation bins of a node are candidates and the choice minimises
`GC · normalised finish + (1 − GC) · area fraction`. `mibs_lat` as above.

Because GCLP/MIBS solve *"meet the deadline with minimum area"*, they leave slack on purpose; judged on makespan they
look weak. The tables show them faithfully (`gclp`, `mibs`) and with the latency target (`*_lat`), and report whether the
deadline is met.

### 5. MAGELLAN — Chatha & Vemuri 2001 (A, I)
Steepest descent with fixing: each step tries every option of every free task, completes the rest with capacity-aware HEFT
(fixed tasks keep their option and their resources stay reserved), fixes the single best (task, option), and stops when
nothing improves. `O(n² K)` HEFT completions, cut off by the budget (best completion so far is returned). The paper's
suitability metric is replaced by earliest finish time (inferred, as in the group's notes). The first completion is HEFT.

## Single-solution metaheuristics (`algos/local.py`)
All start from the HEFT mapping, move one task's option at a time, and are scored by the compiled simulator (default decoder).
Capacity-infeasible candidates are penalised during the search but never reported.

### 6. Kernighan–Lin, KL1 and KL2 (A)
Per Mann, Orbán & Farkas 2007 as summarised in `model.md`: a pass applies the best unlocked move even if it worsens the
cost, locks the task, and returns to the best *feasible* state seen; permissive cost `makespan·(1 + 16·excess⁴)`, forbidden
for `excess > 2`; LIFO tie-break among moves within 5 % of the best gain. KL1 locks a task after one move. KL2 uses dynamic
locking (≤ 5 moves per task per pass) and 10 random restarts after convergence. **Deviation:** when the neighbourhood has
more than 256 moves, 256 are sampled per step (otherwise a single pass is `O(n²K)` evaluations). KL1 stops at convergence,
so it can finish early and then simply holds its result.

### 7. Tabu search — Wiangtong, Cheung & Luk 2002 (F, A)
Selection rule as in `model.md`: best improving non-tabu move unpenalised; otherwise best non-tabu move with a frequency
penalty `0.005 · cost · (#earlier moves of that task)`; if all are tabu, the oldest; aspiration for a new global best;
tenure `√#movable + 1` clamped to [7, 20]. Runs until the budget (the `model.md` default of 400 iterations is replaced by the
wall-clock budget). Same 256-move sampling for large graphs.

### 8. Simulated annealing — Kirkpatrick et al. 1983, Eles et al. 1997 (F, A)
Single-task option moves, Metropolis acceptance, geometric cooling. The temperature follows the **wall clock**:
`T(t) = T_hi (T_lo/T_hi)^(t/budget)` with `T_hi` = mean |Δf| of 40 random moves and `T_lo = 10⁻³ T_hi`, so every budget gets a
complete cooling schedule. The chain runs inside numba in ~1.5 ms chunks. Because the schedule depends on the budget,
**SA is run separately for each of the four budgets** (all other methods are one 10 s run read at 10 ms / 100 ms / 1 s / 10 s).

## Population methods (`algos/evo.py`)

### 9. Genetic algorithm — Wang, Siegel, Roychowdhury & Maciejewski 1997 (A)
Chromosome = option id per task + random-key priority per task (decoded by the list scheduler with explicit priorities).
(μ+λ) with μ = λ = 40, binary tournaments by constraint domination (capacity excess first, then makespan), uniform crossover
on options and keys (p = 0.9), mutation 1/n per gene, capacity repair written back into the genome. Seeded with the HEFT
mapping (priorities from its order) and all-software; the rest are perturbed seeds (random-feasible mappings for n ≤ 200).

### 10. NSGA-II — Deb, Pratap, Agarwal & Meyarivan 2002 (F, A)
Same encoding and operators, fast non-dominated sorting with constraint domination, crowding distance, (μ+μ) elitist
survival. Objectives: **makespan and a second objective**. The README asks for latency–energy fronts, but Tier A and GCPS
contain **no energy data** (the Tier A data card: "No energy or power in v0"), so the benchmark uses *latency–area* (total
utilisation of capacity-limited PEs). Latency–energy is implemented (`second="energy"`) and demonstrated on the golden
case studies that do have energy numbers (`scripts/demo_nsga2_energy.py`). For the single-objective ranking table NSGA-II is
read as "best makespan on its archive"; its front quality is reported as hypervolume.

## Anytime protocol and what is charged
* Every run starts from the same free fallback (all tasks on their fastest CPU option); it is available at t = 0 and counts
  as "no improvement".
* The clock covers everything an algorithm does after the instance is loaded and the flat arrays (`Fast`) are built
  (≈ 3 ms at 200 tasks, 13 ms at 1000). Numba compilation is excluded (a warm-up runs first; the compiled code is cached).
* Budgets 10 ms / 100 ms / 1 s / 10 s. A method that has not produced its first solution by the budget holds the fallback.
* Timings were measured with 10 worker processes on a 12-logical-core laptop; absolute runtimes are indicative only.

## Known limitations
* Metaheuristics search over mappings (and priorities for GA/NSGA-II) with the *default* decoder; they do not choose
  transfer order or insertion. Closing the remaining gap to the CP-SAT optimum (which orders transfers freely) would need a
  unified-priority decoder.
* GCLP/MIBS/MAGELLAN contain reconstructed details (marked **I**).
* Nominal costs only; no uncertainty, energy (except the golden demo), streaming throughput or reconfiguration.
