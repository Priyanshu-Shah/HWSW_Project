# library/ — board specifications

Owned by the **library team**. This README states what we expect; the files themselves are
the library team's to build and maintain.

## Purpose

One JSON file per board (CPU + FPGA on the same board) with the raw numbers needed to estimate
how fast a task runs on the CPU, how fast and how large it is on the FPGA, and what it costs to
move data between them. Specs only: **no per-task or per-kernel costs** (custom_hls computes
those from these numbers).

## Who reads these files

| Reader | Uses |
|---|---|
| pre_TDG: LLM pipeline | `index.json` (board list shown to the user), board `id` and `summary` |
| pre_TDG: custom_hls | `cpu`, `fpga`, `dram`, `interfaces` to compute HW/SW time and resources |
| pre_TDG: validator | `fpga.capacity` keys and values (every HW design point must fit) |
| post_TDG: partitioner | `fpga.capacity` (resource limit), `interfaces` (communication cost) |

## Boards (initial set)

All are Xilinx/AMD Zynq UltraScale+ or Zynq-7000 boards with an ARM CPU and an FPGA on the
same chip:

| `id` | Board | SoC |
|---|---|---|
| `pynq_z2` | PYNQ-Z2 | Zynq-7020 |
| `ultra96v2` | Ultra96-V2 | ZU3EG |
| `kv260` | Kria KV260 | K26 (ZU5EV class) |
| `zcu104` | ZCU104 | ZU7EV |
| `zcu102` | ZCU102 | ZU9EG |

We expect 5–6 boards in total; propose a sixth if a clearly useful one is missing. The board
`id` is what users and the TDG's `platform` field refer to.

## Files

```text
library/
  README.md            this file
  index.json           [{"id", "name", "summary"}] for every board
  platform.schema.json JSON Schema (draft 2020-12) that every board file must pass
  boards/
    <id>.json          one file per board
```

## Syntax rules

- JSON. SI units only: seconds, bytes, Hz, watts. Bytes/s for bandwidth.
- `snake_case` keys. Numbers are literals (no formulas, no strings like `"1.2 GHz"`).
- `id` matches `^[A-Za-z_][A-Za-z0-9_]*$` and the file name.
- FPGA resource keys are **exactly** `lut`, `ff`, `dsp`, `bram18k`, `uram` (same as the TDG cost
  map). Use `0` if the device has none (for example `uram` on Zynq-7000).
- Data type names are exactly `int8`, `uint8`, `int16`, `int32`, `fp16`, `fp32`.
- Prefer **sustained, measured** numbers over peak datasheet numbers for bandwidth and
  CPU efficiency; say which in `provenance`.

## Fields

`R` = required, `O` = optional.

| Field | Unit | | Meaning |
|---|---|---|---|
| `id` | | R | Board id (see above) |
| `name` | | R | Human-readable board name |
| `summary` | | R | One line for the board list shown to the user (SoC, CPU, FPGA size) |
| `cpu.name` | | R | Core, e.g. `cortex-a53` |
| `cpu.cores` | count | R | Cores usable by the application |
| `cpu.freq_hz` | Hz | R | Operating frequency |
| `cpu.ops_per_cycle` | ops/cycle/core | R | Peak per data type (SIMD lanes × issue rate). `0` if unsupported. State the convention in `provenance.note`; we assume one multiply-accumulate counts as one op |
| `cpu.efficiency` | 0–1 | R | Fraction of peak reached on typical compute kernels (measured if possible) |
| `cpu.power_w` | W | O | Typical active power |
| `fpga.part` | | R | Device part number |
| `fpga.fclk_hz` | Hz | R | Clock we assume for HLS designs (typical achievable, e.g. 200 MHz) |
| `fpga.fclk_max_hz` | Hz | O | Highest practical clock |
| `fpga.capacity.{lut,ff,dsp,bram18k,uram}` | count | R | Total programmable-logic resources |
| `fpga.power_w` | W | O | Typical active power of a loaded design |
| `dram.bytes` | bytes | R | Memory shared by CPU and FPGA |
| `dram.bandwidth_Bps` | B/s | R | Sustained total bandwidth |
| `interfaces.default` | | R | Name of the interface used when nothing else is specified |
| `interfaces.<name>.bandwidth_Bps` | B/s | R | Sustained CPU↔FPGA transfer rate through that interface |
| `interfaces.<name>.setup_s` | s | R | Fixed latency per transfer (driver, DMA setup) |
| `interfaces.<name>.note` | | O | E.g. `AXI HP0, 128-bit, DMA` |
| `provenance` | | R | Per section (`cpu`, `fpga`, `dram`, each interface): `{source, url, note}`; `source` is one of `datasheet`, `vendor_doc`, `measured`, `estimated` |

Communication cost used downstream: `setup_s + bytes / bandwidth_Bps` for the chosen interface.
At least one interface is required; list every interface a user could realistically choose.

## Example (values illustrative, to be replaced with real numbers)

```json
{
  "id": "zcu104",
  "name": "Xilinx ZCU104",
  "summary": "Zynq UltraScale+ ZU7EV: 4x Cortex-A53 @ 1.2 GHz, 230k LUT, 1728 DSP",
  "cpu": {
    "name": "cortex-a53", "cores": 4, "freq_hz": 1.2e9,
    "ops_per_cycle": {"int8": 16, "uint8": 16, "int16": 8, "int32": 4, "fp16": 0, "fp32": 4},
    "efficiency": 0.3,
    "provenance": {"source": "estimated", "url": "", "note": "NEON peak; efficiency from a reference benchmark"}
  },
  "fpga": {
    "part": "xczu7ev-ffvc1156-2-e", "fclk_hz": 2.0e8,
    "capacity": {"lut": 230400, "ff": 460800, "dsp": 1728, "bram18k": 624, "uram": 96},
    "provenance": {"source": "datasheet", "url": "", "note": ""}
  },
  "dram": {
    "bytes": 2147483648, "bandwidth_Bps": 1.0e10,
    "provenance": {"source": "measured", "url": "", "note": "sustained, STREAM-like"}
  },
  "interfaces": {
    "default": "axi_hp0",
    "axi_hp0": {
      "bandwidth_Bps": 2.0e9, "setup_s": 1.0e-5, "note": "AXI HP0, DMA",
      "provenance": {"source": "measured", "url": "", "note": ""}
    }
  }
}
```

## Deliverables

1. `boards/<id>.json` for every board in the initial set, filled with real numbers and
   `provenance`.
2. `index.json`.
3. `platform.schema.json`, and all board files passing it.

## Changing the format

Other teams read these files by key name. Renaming or removing a key, or changing a unit or its
meaning, is a **contract change**: tell the project lead first, so the pre_TDG and post_TDG
teams can be informed. Adding a new optional field is fine.
