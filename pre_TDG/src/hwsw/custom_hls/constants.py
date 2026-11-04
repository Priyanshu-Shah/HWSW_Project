"""Tunable constants of the cost models.

The defaults are uncalibrated first guesses. Calibration against Vitis HLS (calibration/)
produces `constants.json` next to this file, which then overrides the defaults.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

CALIBRATED_CONSTANTS_PATH = Path(__file__).parent / "constants.json"


class HlsConstants(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ---- Hardware (FPGA) model -------------------------------------------------------------
    # A design point unrolls `unroll_unit * factor` loop iterations per cycle.
    parallelism_factors: tuple[int, ...] = (1, 4, 16)
    pipeline_depth_cycles: int = 20
    initiation_interval_per_branch: int = 1  # II = 1 + branches * this
    control_heavy_initiation_interval: int = 8
    random_access_traffic_factor: float = 4.0  # DRAM traffic multiplier for random access
    small_buffer_bytes: int = 256  # buffers up to this size live in registers, not BRAM
    bram18k_bytes: int = 2304  # one 18 Kb block
    bram_ports: int = 2  # independent reads per BRAM per cycle
    control_lut_overhead: int = 1500
    ff_per_lut: float = 1.5
    lut_per_bit: dict[str, float] = {
        "mac": 1.0, "mul": 0.5, "add": 1.0, "div": 6.0, "cmp": 1.0,
        "logic": 0.5, "shift": 1.0, "sqrt": 8.0, "exp": 16.0,
    }  # fmt: skip
    floating_point_lut_factor: float = 4.0
    dsp_per_multiply: dict[str, float] = {
        "int8": 0.5, "uint8": 0.5, "int16": 1.0, "int32": 3.0, "fp16": 1.0, "fp32": 3.0,
    }  # fmt: skip
    dsp_per_float_add: dict[str, float] = {"fp16": 1.0, "fp32": 2.0}
    hw_uncertainty: float = 0.3  # lo/hi = value * (1 -/+ this)

    # ---- Software (CPU) model --------------------------------------------------------------
    sw_cores_used: int = 1  # single-threaded software baseline
    sw_time_penalty_per_branch: float = 0.05  # time * (1 + this * branches per iteration)
    sw_control_heavy_factor: float = 1.5
    sw_unsupported_dtype_slowdown: float = 4.0  # vs fp32 rate, when ops_per_cycle[dtype] == 0
    sw_uncertainty: float = 0.3

    # ---- Fixed-cost (opaque) nodes ----------------------------------------------------------
    opaque_uncertainty: float = 0.5


def load_constants(path: Path | None = None) -> HlsConstants:
    """`path` if given, else the calibrated file if it exists, else the defaults."""
    path = path or (CALIBRATED_CONSTANTS_PATH if CALIBRATED_CONSTANTS_PATH.is_file() else None)
    if path is None:
        return HlsConstants()
    return HlsConstants.model_validate_json(path.read_text(encoding="utf-8"))
