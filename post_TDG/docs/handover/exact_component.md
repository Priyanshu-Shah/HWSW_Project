# Handover: the exact component (Person 1 → integration)

**For:** whoever combines the exact, heuristic and ML parts into one partitioner.
**Status:** complete and tested (`tests/test_exact_api.py`, `tests/test_relaxation_lbbd.py`, `tests/test_chain_dp.py`).
**Scope:** latency (makespan) objective with shared-bus contention. All numbers below are on **synthetic** Tier A
instances.

## 1. What it gives you

| You ask | Call | You get |
|---|---|---|
| "Is this spec possible at all?" | `lower_bound(problem, strong=True)` (`diagnostics`) | The smallest makespan any mapping can reach. If it exceeds the deadline, the constraint appears in `violations` and the feedback loop says fb1, with the number to report ("≥ X ms is the floor"). |
| "How good is this plan?" (from a heuristic or the GNN) | `certify(problem, plan)` | Simulator makespan, a valid lower bound, and the gap. `status == "optimal"` when the gap is ≤ 0.1%. |
| "Give me the best schedule you can in T seconds" | `solve_exact(problem, T, warm_starts=[...])` | The best simulator-checked plan, a valid bound, the gap, and a certificate saying how each was established. |
| The same, in the pipeline's `Result` format | `solvers.exact.run(problem, T, warm_starts)` | `Result` with metrics, intervals and diagnostics (strong bound included). |
| Training labels for the ML part | `scripts/make_labels.py` | CP-SAT optima with task + transfer order (unchanged; see §5). |

```python
from estimate import annotate
from solvers.exact import solve_exact, certify
from solvers.heuristics import plan

problem = annotate(graph, platform)
warm = [plan(problem, "peft"), gnn_plan]           # any Plan(mapping, priority, insertion)
res = solve_exact(problem, time_limit_s=30, warm_starts=warm)
res.status      # "optimal" | "feasible" | "infeasible" | "unknown"
res.plan        # Plan: evaluate(problem, res.plan.mapping, priority=res.plan.priority, insertion=res.plan.insertion)
res.makespan_ms, res.bound_ms, res.gap, res.method, res.certificate
```

`Plan` is the same type the heuristics return (`solvers.heuristics.list_sched.Plan`). A GNN decode should produce
one too: `Plan(mapping, priority)`, where `priority` lists tasks and cross-PE transfers (`"a->b"`).

## 2. How `solve_exact` decides

1. **Lower bound** from the mapping relaxation (≤ 15% of the budget, at most 20 s). If the relaxation is infeasible, the instance is **proven infeasible** (exact durations, any schedule).
2. **Incumbents:** your warm starts plus HEFT, scored by the simulator. The best one is the hint for CP-SAT.
3. **Chain with at most one resource budget** → exact DP: optimal in milliseconds to seconds, even at 1,000 tasks.
4. **CP-SAT** with bus contention:
   - ≤ 40 tasks: the whole budget, and it normally proves optimality in under 1 s;
   - > 40 tasks: 40% of the budget.
5. **Not proven yet** → relax-and-schedule for the rest: the relaxation's mapping, scheduled exactly, then full CP-SAT hinted with it.
6. **Result:** the best plan across all of the above. Bound = the strongest valid bound, never above the plan.

## 3. Recommended use in the combined partitioner

```
1. lb = lower_bound(problem, strong=True)
   if lb.violations / lb.structural:  return fb1 with the numbers   (no mapping can work)
2. candidates = [GNN plan (ms), best heuristic plan (ms–s)]
3. for c in candidates: r = certify(problem, c)          # ~0.1 s on ≤ 40 tasks
   if best r.gap <= target_gap:  return it               # e.g. 1–2%
4. res = solve_exact(problem, budget, warm_starts=candidates)
   return res.plan with res.gap                          # never worse than the best candidate
```

| graph | what to expect from `solve_exact` | budget that suffices |
|---|---|---|
| ≤ 40 tasks | proven optimal on all 24 scaling instances of this size (median < 1 s); on the 1,000 training labels, CP-SAT alone proved 854 of the 873 with a schedule within 10 s | 5–10 s |
| 50–60 tasks | 9 of 12 scaling instances proven in 20 s; the rest have gaps of a few % (bus contention) | 20–30 s |
| 60–100 tasks | 22 of 30 proven optimal, 1 proven infeasible; 7 open, with gaps 0.1–8.2% (one at 34%, see §6). Median gap 0%, p90 5.7% (`exact_large_closing.md`). | 60 s |
| chains, any size | exact (DP) | < 2 s at 1,000 tasks |
| > 100 tasks, general DAG | use heuristics / GNN for the plan and `certify` for the gap. CP-SAT is unreliable here: on 1,000-task chains it ended 18.6% above the optimum, worse than HEFT. | — |

The ML part should treat the exact component as a **teacher and a referee**:
- teacher: labels on ≤ 40 tasks; best-known references on 60–100 tasks;
- referee: `certify` gives a gap for every GNN output, on any size where the relaxation solves.

## 4. Guarantees

- **Every schedule is re-scored by the simulator** (exact durations, bus contention). No solver-internal makespan is ever reported as the result.
- **Bounds are valid.** The mapping relaxation never exceeded a proven optimum on 1,043 instances (176 test + 854 training + 13 large). It is exact on chains, matching the DP within 0.01%.
  - Tightness, optimum / bound: median 1.001, p90 1.03–1.04, worst 1.23 (critical path: 1.12–1.55 median, up to 5.8).
- **Infeasibility proofs** come only from the relaxation (exact durations), or from CP-SAT after the µs-slack re-check.
- **Tolerance:** "optimal" means within 0.1% (CP-SAT schedules in whole µs).

## 5. Pitfalls already handled — keep them handled if you touch the code

| Pitfall | Symptom | Handling |
|---|---|---|
| CP-SAT (OR-Tools 9.15) + **start-time hints** on a free mapping | false OPTIMAL (seed 5003: claimed 22.3 ms, 20.2 ms exists) | Hint the **mapping only**. Warned in `solve_cp`'s docstring. |
| CP-SAT **µs rounding** | false INFEASIBLE when the deadline is within n µs of the optimum (scaling seed 8100) | `solve_exact` re-solves with n µs of slack and checks against the exact deadline |
| HiGHS **presolve tolerances** | the relaxation reported 232.4 ms where 227.4 ms is feasible (seed 195) | Feasibility tolerances 1e-9 in `RelaxationModel.solve`. Do not loosen them. |
| Label precision | `label.makespan_ms` can be up to ~0.2% below the true optimum | Compare with `replay.makespan_with_contention_ms` |
| Wall-clock budgets | results depend on machine load (Docker/WSL once used 90% of the CPU) | Benchmark on an idle machine; report the budget |
| Laptop **Modern Standby** | a "60 s" call took 1,500 s (5005), and labelling once stalled for 14 min. Windows logs confirm the machine was in standby both times. CP-SAT itself respects its limit. | Disable sleep for long runs; never read a long wall time as a solver hang without checking |

## 6. Limits (not done)

- **Objectives other than latency:** `solve_exact` falls back to CP-SAT alone; there is no bound or certificate.
- **Streaming throughput (II):** not bounded or optimised by this component. The simulator's exact max-cycle-ratio evaluator scores it.
- **Benders (`lbbd`):** correct, but its no-good cuts are weak. It is kept as a reference; `solve_exact` does not use it.
- **Large-set seed 5020** (99 tasks, heavy bus traffic): a schedule of 337.3 ms meets the 342 ms deadline, but the bound is only 221 ms (34% gap). The relaxation is weakest when bus transfers dominate. Heuristics land at 1,400–2,700 ms there.
- **Open large instances:** 5000 (2.6%), 5003 (8.2%), 5004 (4.5%), 5005 (5.7%), 5008 (0.12%), 5027 (3.8%), 5020 (34%). Longer budgets (5 min+) would likely close several.

## 7. Files

| path | what |
|---|---|
| `solvers/exact/api.py` | `solve_exact`, `certify`, `run`, `ExactResult` |
| `solvers/exact/relaxation.py` | mapping relaxation (bound, Benders master) |
| `solvers/exact/hybrid.py` | relax-and-schedule |
| `solvers/exact/lbbd.py` | logic-based Benders (reference) |
| `solvers/exact/chain_dp.py` | exact DP for chains |
| `solvers/cp/model.py` | CP-SAT model (bus contention, canonical tie-break for labels) |
| `diagnostics/lower_bound.py` | `lower_bound(problem, strong=True)` |
| `data/labels/*.jsonl` | CP-SAT labels: test 200, train 1,000, large 30 |
| `data/benchmarks/large_best_known.synthetic.json` | best schedule + bound per large instance (mapping + priority included) |
| `docs/benchmarks/exact_*.md`, `chain_dp_vs_cpsat.md` | the evidence behind every number here |
| `DECISIONS.md` (2026-11-04 to 2026-11-07) | why each choice was made |
