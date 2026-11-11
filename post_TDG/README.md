# post_TDG — costed TDG → HW/SW partition

Takes the costed TDG from `pre_TDG` (graph + `costs`, format in `../TDG/`) and the board file,
decides for every node whether it runs on the CPU or on the FPGA (and which HW design point),
and returns the schedule, the latency with its uncertainty range, and feedback when the deadline
cannot be met.

## Set up (Python ≥ 3.11)

```
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # for the GNN
.venv/bin/python -m pytest      # about 280 tests; 4 skip (TGFF, E3S, scipy not included)
```

`highspy` is pinned below 1.13: PuLP 3.3 does not detect HiGHS 1.15.

## Run

```
.venv/bin/partition tdg.costed.json --board ../library/boards/kv260.json --out partition.json
```

From `pre_TDG`: `hwsw partition <costed TDG>`, or `hwsw run "<task>"` (whole pipeline). The app
has a **Partition** button after **Cost it**.

Exit codes: 0 every constraint met; 1 answered, but the deadline cannot be met (the result holds
the fastest partition and the reason); 2 bad input; 3 crash.

## How it decides (decisions D7, D8)

1. `post_tdg/convert.py`: costed TDG + board → the partitioner's IR (one implementation per
   way to run a node: software, or one per HW design point; setup nodes dropped; one iteration
   with the deadline, per D3).
2. `post_tdg/partition.py`:
   - up to 40 nodes: exact solver (CP-SAT + relaxation bound), warm-started with PEFT,
     MAGELLAN and the GNN. Proven optimum in under a second on our graphs;
   - larger: GNN-seeded genetic algorithm (`fastheur/`), then certified against the bound.
3. Every plan is scored by one simulator (`sim/`), with shared-bus contention and FPGA
   capacity; Monte Carlo over the cost ranges gives p05–p95.

## Layout

| path | what |
|---|---|
| `post_tdg/` | Glue for this repo: converter, solver choice, `partition` CLI |
| `schemas/ estimate/ sim/ solvers/ diagnostics/ metrics/ datagen/` | The partitioner as handed over (exact, heuristics, GNN; see `docs/handover/`) |
| `fastheur/` | Fast compiled heuristics (SA, tabu, GA, …) and `bridge.py` (GNN sampling); package renamed from `hwsw` |
| `models/gnn_r2_all_base_s1.pt` | GNN chosen on validation data in the Thor run 2 (`docs/thor_run2_summary.md`) |
| `data/` | Test data for the upstream tests (synthetic labels, GCPS graphs) |
| `tests/` | Upstream tests, `tests/fastheur/`, and `tests/post_tdg/` (with costed benchmark fixtures) |
| `docs/` | `contract.md`, `model.md`, handover notes, results of the heuristics and Thor runs |

All training and test graphs of the partitioner are synthetic. The Thor run folder
(`../Partiotiner_THOR_RUN_v2/`, caches and outputs) is kept out of git.
