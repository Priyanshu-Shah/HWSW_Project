# HW/SW partitioner: hand-over package

**For:** the person combining the exact, heuristic and ML parts into one partitioner.
This is the full project code: all three parts share one simulator, one problem format and one result format.
Everything here is **synthetic** unless a file says otherwise.

## Set up (Python ≥ 3.11)

```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"          # Windows; .venv/bin/python elsewhere
.venv\Scripts\python -m pytest                             # 195 pass, 3 skip here (TGFF not included; the GNN tests skip without torch)
# optional, for the GNN only (CPU build is enough):
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## The three parts and where they stand

| part | entry point | status | read |
|---|---|---|---|
| Exact (Person 1) | `solvers.exact.solve_exact`, `certify`, `run`; `diagnostics.lower_bound(p, strong=True)` | **done**, tested, benchmarked | `docs/handover/exact_component.md` |
| Heuristics (Person 2) | `solvers.heuristics.plan(problem, name)` for HEFT, PEFT, MAGELLAN, KL, tabu, SA, GA; `nsga2` | implemented and tested; the systematic comparison is Person 2's job | `docs/model.md` (heuristics section), `docs/benchmarks/heuristics_vs_optimum.md` |
| ML (Person 3) | `solvers.gnn` (features, model, decoder), `scripts/train_gnn.py` | plumbing done; **no trained model yet**. First diagnosis: the priority heads learn, the option head does not yet beat "fastest option". | `DECISIONS.md` 2026-10-22 and 2026-10-24, `docs/model.md` (GNN section) |

All three return or accept the same `Plan(mapping, priority, insertion)` (`solvers.heuristics.list_sched.Plan`).
`sim.evaluate(problem, plan.mapping, priority=plan.priority, insertion=plan.insertion)` is the one scorer.

## Data in this package

| path | what |
|---|---|
| `data/labels/tier_a_v0_bus.synthetic.jsonl` | test: 200 graphs (10–40 tasks), CP-SAT labels with bus contention; 176 optimal |
| `data/labels/tier_a_v0_bus_train.synthetic.jsonl` | training: 1,000 graphs; 854 optimal |
| `data/labels/tier_a_v0_bus_large.synthetic.jsonl` | 30 graphs of 60–100 tasks |
| `data/benchmarks/large_best_known.synthetic.json` | best schedule + proven bound per large graph (22 optimal, 1 infeasible) |
| `data/benchmarks/gcps_tgff/` | GCPS-model benchmark graphs (20–1,000 tasks) |
| `data/labels/tier_a_v0.synthetic.jsonl` | the test graphs without bus contention (ablation) |

## Rules worth keeping

1. **Score everything with `sim.evaluate`.** Never report a solver's internal makespan.
2. **Tune on `train` only.** Use `test` and `large` for final numbers.
3. **CP-SAT:** hint the mapping only, never start times (false optimum in OR-Tools 9.15). Treat "infeasible" near a deadline with care (µs rounding). `solve_exact` handles both.
4. **Benchmark with sleep disabled on an idle machine.** Wall-clock budgets count standby time.
5. **`DECISIONS.md`** has the reason for every non-obvious choice. Add to it.

**Not in this package:**
- TGFF source: separate `TGFF_3.5.zip`; only needed to regenerate `gcps_tgff` graphs;
- paper PDFs;
- the QuickEst and HLSyn raw data (the fitted statistics are in `datagen/fits/`);
- virtual environments and build output.
