# custom_hls

**What it does in one sentence:** for every task (node) in the graph, it guesses *how long it
takes on the CPU*, *how long it takes on the FPGA (and how much FPGA space that costs)*, so the
partitioner can decide where each task should run. It never synthesizes anything: it counts work
and applies simple formulas.

Input: a TDG without costs + a board file. Output: the same TDG with a `costs` section
(the "cost map" in `TDG/schemas/tdg.schema.json`). Entry point: `estimate_costs()` in
`estimate_tdg.py`, or `hwsw cost` on the command line.

## The intuition, in plain words

Every task is described as a **loop nest**: "repeat this small piece of work N times", with the
arrays it reads and writes (format D1 in `TDG/decisions.md`). From that description we can answer
three simple questions:

1. **How much work is there?** Multiply the loop sizes, multiply by the arithmetic inside one
   iteration. That gives the number of *operations*. We also add up how many bytes the arrays
   hold: the *memory traffic*.
2. **How fast could the CPU do it?** The CPU does so many operations per second
   (`frequency x operations per cycle x efficiency`, from the board file). It also has to fetch
   the data from memory. Whichever is slower (computing or fetching) decides the time.
3. **How fast could the FPGA do it, and what does it cost?** An FPGA is fast because we can build
   many copies of the arithmetic and run them *in the same clock cycle*. Call the number of
   copies **P** (parallelism).
   - More copies means fewer cycles: time is about `iterations / P / clock`.
   - But more copies use more of the chip: each copy needs DSP blocks (multipliers) and LUTs
     (logic), and the data needs on-chip memory (BRAM) to be reused.
   - And past some P it stops helping, because the FPGA cannot be fed from memory any faster.

An analogy: the CPU is one very fast cook. The FPGA is a kitchen where you can hire P slow cooks;
each costs counter space, and they all share one delivery door (memory bandwidth). `custom_hls`
asks "how long would it take with 1 cook, 4 cooks, 16 cooks, and how much counter space does each
option use?" The partitioner then picks the option that works best for the whole graph.

Because the FPGA has several options (several P), the cost map lists **design points**
(`p1`, `p2`, `p3`) per node; the CPU has just one time.

## What happens to one node (worked example)

3x3 convolution over a 2048x2048 int16 image, on the `zcu104` test board
(see `tests/custom_hls/` for the checks of these numbers):

| Step | Result |
|---|---|
| **Operations**: 2046 x 2046 outputs x 9 multiply-accumulates | 37,675,044 ops |
| **Memory traffic**: input 8.4 MB + output 8.4 MB + 18 B kernel | 16,760,858 bytes |
| **On-chip buffer**: a sliding window needs 2 full image rows + 3 values | 8,198 bytes |
| **CPU time**: 37.7M ops / (1.2 GHz x 8 ops/cycle x 0.3 efficiency) | 13.1 ms |
| **FPGA, P=9** (the 3x3 window, from the LLM's `unrollable` hint) | 20.9 ms, 9 DSP, 1644 LUT, 4 BRAM |
| **FPGA, P=36** | 8.4 ms (**memory-bound**: 16.8 MB through a 2 GB/s link), 36 DSP |
| **FPGA, P=144** | 8.4 ms (no faster: still memory-bound), 144 DSP |

The last two rows are the model doing its job: a bigger P costs more chip and gains nothing,
so the partitioner will not pick it.

The partitioner later adds the cost of moving data between CPU and FPGA (`setup + bytes /
bandwidth` of the board's interface) wherever two connected tasks run on different sides.

## How the code is organized (follow it top to bottom)

```text
estimate_tdg.py        for every node call node_costing, collect problems, attach `costs`
 └ node_costing.py     one node -> its cost-map entry
    ├ templates/       conv2d, gemm, pool, elementwise, reduce, memcpy -> a loop nest (D1 dict)
    ├ loop_nest.py     the loop nest as typed objects (ArrayDecl, LoopNest)
    ├ loop_nest_analysis.py   count work: ops, memory traffic, on-chip buffers, pipeline rate (II)
    ├ hw_design_points.py    P -> FPGA time and resources, drop points that do not fit the board
    ├ sw_time.py       CPU time
    └ estimate_values.py     wraps a number as {value, lo, hi, source}
constants.py           every tunable number of the models (the things we calibrate)
bounds.py              sanity check on a costed TDG: all-CPU latency vs best possible latency
data_types.py          int8 ... fp32: byte sizes, widths
```

Node kinds: `opaque` nodes (camera capture, file output...) just carry a fixed CPU time that we
copy into the cost map. `template` nodes are expanded into a loop nest first. `loop_nest` nodes
are used as written by the LLM.

## Rules of thumb used (conventions)

- One multiply-accumulate = one op. Loads and stores are not ops.
- Every array crosses memory once. Re-reading is assumed to come from on-chip buffers, which
  is what the BRAM count pays for.
- The LLM's `unrollable` hint sets the smallest P; the options are hint x (1, 4, 16). If none of
  them fit the board, the hint is ignored (P = 1, 4, 16).
- The FPGA reaches memory through the board's default interface; the CPU through DRAM directly.
- The CPU baseline runs one thread (`sw_cores_used = 1`).
- Every number gets a range `[lo, hi]` = value -/+ 30% (CPU/FPGA) or 50% (opaque), and a `source`
  (`roofline` for FPGA, `profile_model` for CPU). **These ranges are currently guesses, not
  measurements.** That is exactly what calibration replaces.

## What the model does NOT capture (known gaps)

| Gap | Effect |
|---|---|
| No real synthesis: clock speed, routing, congestion | big designs will not reach the assumed clock |
| Memory is modelled as one bandwidth number | burst efficiency, many tasks sharing memory, caches |
| Reused arrays are held whole on chip (no tiling) | big matrix multiplies get no FPGA option |
| Loop-carried dependencies (for example a running sum) not modelled | real II may be above 1 |
| CPU: no cache effects, no multithreading, vectorization assumed | CPU time may be off for big/small working sets |
| Random access = a flat traffic multiplier | |
| One set of constants for all boards | 7-series and UltraScale+ DSPs differ (int8 packing) |

## Calibration plan

**Goal:** replace the guessed constants in `constants.py` with fitted ones, and replace the
guessed `[lo, hi]` ranges with measured error. Target (from `../../../implementation.md`): HW time and
resources within **+-50%** of Vitis HLS on kernels the fit has not seen.

### 1. What gets calibrated, and how it is measured

| Constant(s) | Meaning | Measured from | Fit |
|---|---|---|---|
| `dsp_per_multiply`, `dsp_per_float_add` | DSPs per multiplier / float adder, per dtype | HLS report: DSP count at several P | slope of DSP vs P |
| `lut_per_bit`, `floating_point_lut_factor` | logic per arithmetic op | HLS report: LUT count at several P, kernels with different op mixes | non-negative least squares over all kernels |
| `control_lut_overhead`, `ff_per_lut` | fixed logic per node; flip-flops per LUT | intercept of LUT vs P; FF/LUT ratio | intercept; median ratio |
| `bram18k_bytes`, `bram_ports`, `small_buffer_bytes` | how arrays map onto BRAM | HLS report: BRAM for arrays of varied size and partitioning | check formula, adjust |
| `pipeline_depth_cycles` | cycles to fill the pipeline | HLS report: latency minus `trip x II` | median |
| `initiation_interval_per_branch`, `control_heavy_initiation_interval` | slowdown from branches / irregular loops | HLS report: achieved II on branchy and irregular kernels | per kernel class |
| `hw_uncertainty` | the `[lo, hi]` width for FPGA numbers | error after the fit, on kernels held out of the fit | 90th percentile of the error |
| `sw_time_penalty_per_branch`, `sw_control_heavy_factor`, `sw_unsupported_dtype_slowdown` | CPU penalties | run the reference kernels natively on the board's CPU | ratio fits |
| `sw_uncertainty` | the `[lo, hi]` width for CPU numbers | CPU error on held-out kernels | 90th percentile |
| `random_access_traffic_factor` | extra DRAM traffic from random access | **needs on-board runs** (HLS alone cannot show DRAM behaviour) | stage 2 |

Not ours to calibrate: the board numbers themselves (`cpu.efficiency`, `cpu.ops_per_cycle`,
`fpga.fclk_hz`, DRAM and interface bandwidth) belong to the library team. If our CPU runs show a
systematic mismatch, we report it to the user so the library team can correct the board file.

### 2. Reference kernels (14)

One folder per kernel under `calibration/kernels/<name>/` with: `kernel.cpp` (HLS C++),
`nest.json` (the same kernel as a D1 loop nest, which custom_hls costs), and `native.cpp` (the
same code for the CPU run).

| # | Kernel | Why |
|---|---|---|
| 1 | conv2d 3x3, int8 | the main CNN op |
| 2 | conv2d 3x3, int16 | dtype effect on DSP/LUT |
| 3 | conv2d 1x1 (pointwise), int8 | no sliding window |
| 4 | gemm, int16 | reuse buffer, DSP-heavy |
| 5 | gemm, fp32 | floating-point cost |
| 6 | max-pool 2x2, int8 | compare ops, window buffer |
| 7 | elementwise add, int8 | memory-bound extreme |
| 8 | elementwise multiply-add, fp32 | float op mix |
| 9 | sum reduction, int32 | loop-carried dependency |
| 10 | memcpy, 1 MB | pure data movement |
| 11 | clamp / threshold with branches | branch penalty |
| 12 | histogram | random write access |
| 13 | table lookup (S-box style) | random access, const table |
| 14 | insertion sort | data-dependent loops (control-heavy) |

Each at 2-3 sizes where it makes sense, so the fit can separate "per iteration" cost from "fixed" cost.

### 3. Procedure

1. **Pick the parallelism values.** A script runs custom_hls on each `nest.json` and prints the
   P values it would produce for each board, so HLS is run at *exactly* those P (same design
   points, 1:1 comparison).
2. **Synthesize.** For each kernel x P x target part (`xc7z020`, `xczu3eg`, `xczu7ev`), run Vitis
   HLS C-synthesis with `PIPELINE II=1` and `UNROLL`/`ARRAY_PARTITION` matching P, at the board's
   `fclk`. Keep the report (`csynth.xml`). No bitstreams are needed.
3. **Extract** into `calibration/results/hls_reports.csv`: latency (cycles), achieved II,
   LUT, FF, DSP, BRAM18K, estimated clock period.
4. **Predict** with custom_hls for the same kernel/P/board, into `predictions.csv`.
5. **CPU runs.** Compile `native.cpp` for the board's CPU (`-O3`, NEON), run each kernel 20 times,
   record the median wall time into `cpu_times.csv`. Needs the physical boards (or an emulator
   for a first pass).
6. **Fit** (`calibration/fit_constants.py`): fit the constants of the table above on about 2/3
   of the kernels, write them to `custom_hls/constants.json` (which `load_constants()` then picks
   up automatically; no code change).
7. **Validate on held-out kernels**: error = predicted / actual - 1, for time and each
   resource. Rotate the held-out set (leave-one-kernel-out) so every kernel is checked once.
8. **Set the ranges**: `hw_uncertainty` and `sw_uncertainty` = 90th percentile of the absolute
   error. Write the error table to `calibration/REPORT.md`.
9. **Stage 2 (needs boards, optional for the MVP)**: run 3-4 kernels as real accelerators to
   measure the effective FPGA memory bandwidth and the random-access factor.

### 4. Done when

- On held-out kernels, 90% of FPGA time and resource predictions are within +-50% of the HLS
  report (the MVP target), and the shipped `[lo, hi]` ranges cover the measured value for at
  least 80% of them.
- `calibration/REPORT.md` lists the error per kernel and per quantity.
- `constants.json` is committed together with the commit of the data that produced it.

### 5. Order of work

Steps 1-4 and 6-8 for the **FPGA side** need only Vitis HLS (no board) and give the biggest
gain (DSP/LUT/BRAM/II). Do these first. The CPU side (step 5) needs the board.

## TODO

Calibration (above):
- [ ] Write the 14 reference kernels and their `nest.json` files (`calibration/kernels/`)
- [ ] Script: print the P values custom_hls produces per kernel/board
- [ ] Script: run Vitis HLS and extract `csynth.xml` into a CSV
- [ ] `fit_constants.py` and the held-out validation
- [ ] CPU timing runs on the boards; stage 2 memory measurements

Model improvements (after calibration shows where the error is):
- [ ] Separate constants per FPGA family (7-series vs UltraScale+), chosen from `fpga.part`
- [ ] Tiling for `reuse` arrays, so big matrix multiplies get a hardware option
- [ ] Loop-carried dependencies (reductions): II from the adder latency
- [ ] Memory-port limits on II (reads per cycle vs BRAM ports) for non-reuse arrays
- [ ] Clock: lower the achievable clock for large designs (use the HLS estimated period)
- [ ] CPU: multithreading option, cache effects (working set vs cache size)
- [ ] Per-edge cost overrides (`costs.edges`): currently never produced
- [ ] Power estimates (`power_w`): not produced
- [ ] More template ops if the LLM keeps writing the same loop nest by hand
- [ ] Check templates against real network layers (YOLO, MobileNet) once we have them
