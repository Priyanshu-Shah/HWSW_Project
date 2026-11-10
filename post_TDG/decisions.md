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

## D8. Solver choice and result (2026-11-10)

- ≤ 40 iteration nodes: the exact component (`solvers.exact.run`), warm-started with PEFT,
  MAGELLAN and the GNN's best of 16 samples. It gives a proven optimum (CP-SAT) or the best
  plan and a valid bound. Our graphs (3–15 nodes) solve in under 0.5 s.
- > 40 nodes: the fast GA (`fastheur`) seeded with 8 GNN samples (best mix in the Thor run 2,
  `docs/thor_run2_summary.md`), PEFT as fallback, then `certify` for the gap.
- The GNN (`models/gnn_r2_all_base_s1.pt`) was trained on synthetic graphs only, so it only
  proposes starting points; the simulator scores every plan. Without torch or the model, the
  solvers run without it and the result says so.
- No partition meets the deadline (fb1): the fastest partition is still returned
  (`best_effort: true`), with the deadline feedback and the minimum latency.
- Output `partition.result.json`, SI units: `latency_s` (nominal + Monte Carlo range),
  `all_software_s`, `speedup`, per node `target` (`sw`/`hw`), `pe`, `point`, start/finish,
  transfers, FPGA resources used, `feedback` (`none`/`fb1`/`fb2` with reasons).
- CLI exit codes: 0 constraints met, 1 answered but not met, 2 bad input, 3 crash.
