# post_TDG decisions

Active decisions only; edit in place when one changes. Decision ids are global across the repo
(D1, D3–D5 in `TDG/decisions.md`; D2, D6 in `pre_TDG/decisions.md`). The partitioner's own
history (exact solver, heuristics, GNN) is in `docs/partitioner_decisions_log.md`.

## D7. Costed TDG → partitioner input (2026-11-10)

`post_tdg/convert.py` turns the costed TDG (v0.2) and its board file into the partitioner's IR
(`schemas/`, 0.1.0). The contract files are read as they are; nothing in `TDG/` or `library/`
changes.

- Units: s → ms, B/s → MB/s, setup s → µs. `{value, lo, hi}` → `{expr: value, unc: [lo/value, hi/value]}`
  (`lo` 0 is floored at 1e-3 of the value, because `unc` needs `0 < lo`).
- One implementation per way to run a node: `sw` on a CPU core, and one per HW design point
  (`point` id) on the FPGA. `affinity` keeps only the allowed ones.
- `phase: setup` nodes and their edges are dropped (D3: one iteration only). Parallel edges
  (same src and dst) are merged; their bytes add up.
- **D3 is kept:** every graph is one iteration (`arrival: one_shot`) with `deadline_s` as a
  constraint. The partitioner's streaming (pipelined) model is not used for the decision.
- Board: one exclusive PE per CPU core (`cpu0`…), one spatial PE `fpga0` with `fpga.capacity`,
  the board's default interface between every pair of PEs (shared bus), and DRAM bandwidth
  between CPU cores.
- `costs.edges` overrides are refused for now (custom_hls writes none).
- pre_TDG runs the `partition` command on the JSON files; it never imports this code.

Open: custom_hls HW times may already include the kernel's DRAM traffic, while the partitioner
adds the edge transfer on top. Check against measurements when calibrating.

## D8. Solver choice and result (2026-11-10, portfolio 2026-11-12)

The budget-driven portfolio of the integrated package v2 (`post_tdg/portfolio.py`, ported from
`HWSW_integrated_v2/combo/partition.py`; evidence in `docs/person3_results.md`). Budget = `time_limit_s`
(default 10 s); compilation and model loading are outside it.

- Always first: compiled PEFT (safety net, and an extra start for the searches).
- Budget < 50 ms: SA from HEFT only. ≤ 200 nodes: GA seeded with the GNN's best 8 of 256 samples
  plus PEFT. > 200 nodes: GA from HEFT + PEFT (the GNN stops helping there).
- ≤ 40 nodes, budget ≥ 2 s: the GA for 20 % of the budget (0.1–1 s), then CP-SAT warm-started with
  every candidate. 41–100 nodes, budget ≥ 10 s: SA 1 s + GA/GNN 1 s, then CP-SAT.
- Best feasible candidate by `sim.evaluate` wins; a proven optimum wins a tie. Plans CP-SAT did not
  see are certified against the mapping-relaxation bound (30 % of the budget, ≥ 1 s) for the gap.
- Cost: our 3–15-node graphs still solve to a proven optimum, but take ~1 s more than the old
  exact-only path (the fixed GA step). Same or better plans on 54–140-node graphs.
- The GNN (`models/gnn_r2_all_base_s1.pt`, = v2 `models/gnn.pt`) was trained on synthetic graphs
  only, so it only proposes starting points. Without torch or the model, the portfolio runs
  without it and the result says so.
- No partition meets the deadline (fb1): the portfolio runs again without the deadline and the
  fastest partition is returned (`best_effort: true`), with the deadline feedback and the minimum latency.
- Output `partition.result.json`, SI units: `latency_s` (nominal + Monte Carlo range),
  `all_software_s`, `speedup`, per node `target` (`sw`/`hw`), `pe`, `point`, start/finish,
  transfers, FPGA resources used, `feedback` (`none`/`fb1`/`fb2` with reasons).
- CLI exit codes: 0 constraints met, 1 answered but not met, 2 bad input, 3 crash.
