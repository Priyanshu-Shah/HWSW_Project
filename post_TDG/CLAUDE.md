# post_TDG/ — costed TDG → HW/SW partition

Partitioner team's folder. Input: a costed TDG (`../TDG/` contract) + board file (`../library/`). Output: `partition.result.json`.

- `post_tdg/`: our glue. `convert.py` (D7), `partition.py` (D8), `cli.py` (`partition` command).
- Upstream partitioner code, kept as handed over: `schemas/ estimate/ sim/ solvers/ diagnostics/ metrics/ datagen/` (exact solver, heuristics, GNN) and `fastheur/` (fast SA/tabu/GA, renamed from `hwsw`). Change it only to fix a bug; note why in `docs/partitioner_decisions_log.md`.
- Decisions: `decisions.md` (D7, D8). Score every plan with `sim.evaluate`.
- **Never edit `../TDG/`, `../library/`, `../pre_TDG/`.** Format change needed: stop and tell the user.
- Own `.venv` (torch CPU, ortools, numba). Check: `.venv/bin/python -m pytest` and `-m ruff check post_tdg tests/post_tdg`.
