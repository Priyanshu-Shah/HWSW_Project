"""FPGA design points of a loop nest: for each parallelism P, the time and the resources.

A design point executes P loop iterations per cycle (initiation interval permitting):
  compute time = ceil(iterations / P) * II / fclk      (+ pipeline fill for the node's latency)
  memory time  = DRAM traffic / the FPGA's memory bandwidth
  node time    = max(compute, memory)                   (roofline)
Points that do not fit the board are dropped; no points left means "no hardware option".
"""

import math
from dataclasses import dataclass

from hwsw.custom_hls.constants import HlsConstants
from hwsw.custom_hls.data_types import bit_width, is_floating_point
from hwsw.custom_hls.loop_nest import REUSE
from hwsw.custom_hls.loop_nest_analysis import LoopNestAnalysis, OnChipBuffer
from hwsw.platforms import Board

_MULTIPLY_OPS = ("mac", "mul")


@dataclass(frozen=True)
class HardwareDesignPoint:
    point_id: str  # p1, p2, ... in increasing parallelism
    parallelism: int
    time_s: float  # latency of one run of the node
    ii_s: float  # time between consecutive runs when pipelined (excludes pipeline fill)
    resources: dict[str, int]


def hardware_design_points(
    analysis: LoopNestAnalysis, board: Board, constants: HlsConstants
) -> list[HardwareDesignPoint]:
    """Design points that fit `board`. The LLM's unroll hint is advisory: it is tried first, and
    ignored if none of the hinted points fit."""
    for unit in (analysis.unroll_unit, 1):
        fitting = [
            (parallelism, resources)
            for parallelism in _candidate_parallelism(analysis, unit, constants)
            if _fits(resources := _resources(analysis, parallelism, constants), board)
        ]
        if fitting:
            return [
                _design_point(f"p{index}", parallelism, resources, analysis, board, constants)
                for index, (parallelism, resources) in enumerate(fitting, start=1)
            ]
    return []


def _candidate_parallelism(
    analysis: LoopNestAnalysis, unit: int, constants: HlsConstants
) -> list[int]:
    capped = {min(unit * factor, analysis.iterations) for factor in constants.parallelism_factors}
    return sorted(capped)


def _fits(resources: dict[str, int], board: Board) -> bool:
    return all(amount <= board.fpga.capacity.get(key, 0) for key, amount in resources.items())


def _design_point(
    point_id: str,
    parallelism: int,
    resources: dict[str, int],
    analysis: LoopNestAnalysis,
    board: Board,
    constants: HlsConstants,
) -> HardwareDesignPoint:
    cycles = math.ceil(analysis.iterations / parallelism) * analysis.initiation_interval
    compute_s = cycles / board.fpga.fclk_hz
    pipeline_fill_s = constants.pipeline_depth_cycles / board.fpga.fclk_hz
    memory_s = analysis.offchip_traffic_bytes / _fpga_memory_bandwidth(board)
    return HardwareDesignPoint(
        point_id=point_id,
        parallelism=parallelism,
        time_s=max(compute_s + pipeline_fill_s, memory_s),
        ii_s=max(compute_s, memory_s),
        resources=resources,
    )


def _fpga_memory_bandwidth(board: Board) -> float:
    """The FPGA reaches DRAM through the default interface, never faster than DRAM itself."""
    return min(board.dram.bandwidth_Bps, board.default_interface_spec.bandwidth_Bps)


def _resources(
    analysis: LoopNestAnalysis, parallelism: int, constants: HlsConstants
) -> dict[str, int]:
    luts = math.ceil(
        parallelism * _luts_per_iteration(analysis, constants) + constants.control_lut_overhead
    )
    return {
        "lut": luts,
        "ff": math.ceil(luts * constants.ff_per_lut),
        "dsp": math.ceil(parallelism * _dsps_per_iteration(analysis, constants)),
        "bram18k": sum(_bram18k(buffer, parallelism, constants) for buffer in analysis.buffers),
    }


def _luts_per_iteration(analysis: LoopNestAnalysis, constants: HlsConstants) -> float:
    dtype = analysis.compute_dtype
    scale = bit_width(dtype) * (
        constants.floating_point_lut_factor if is_floating_point(dtype) else 1
    )
    return sum(
        count * constants.lut_per_bit[op] * scale
        for op, count in analysis.body_ops.items()
        if op in constants.lut_per_bit
    )


def _dsps_per_iteration(analysis: LoopNestAnalysis, constants: HlsConstants) -> float:
    dtype = analysis.compute_dtype
    multiplies = sum(analysis.body_ops.get(op, 0) for op in _MULTIPLY_OPS)
    float_adds = analysis.body_ops.get("add", 0) if is_floating_point(dtype) else 0
    return multiplies * constants.dsp_per_multiply[
        dtype
    ] + float_adds * constants.dsp_per_float_add.get(dtype, 0)


def _bram18k(buffer: OnChipBuffer, parallelism: int, constants: HlsConstants) -> int:
    if buffer.size_bytes <= constants.small_buffer_bytes:
        return 0  # kept in registers
    capacity_blocks = math.ceil(buffer.size_bytes / constants.bram18k_bytes)
    if buffer.pattern == REUSE:  # P parallel readers need enough ports
        return max(capacity_blocks, math.ceil(parallelism / constants.bram_ports))
    return capacity_blocks
