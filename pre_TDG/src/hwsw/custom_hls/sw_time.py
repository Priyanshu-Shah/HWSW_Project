"""CPU time of a loop nest: the slower of compute and memory, plus a penalty for branches.

compute = ops / (freq * cores_used * ops_per_cycle[dtype] * efficiency)
memory  = DRAM traffic / DRAM bandwidth
"""

from hwsw.custom_hls.constants import HlsConstants
from hwsw.custom_hls.loop_nest_analysis import LoopNestAnalysis
from hwsw.platforms import Board


def software_time_s(analysis: LoopNestAnalysis, board: Board, constants: HlsConstants) -> float:
    ops_per_second = (
        board.cpu.freq_hz
        * constants.sw_cores_used
        * _ops_per_cycle(analysis.compute_dtype, board, constants)
        * board.cpu.efficiency
    )
    compute_s = analysis.arithmetic_ops / ops_per_second
    memory_s = analysis.offchip_traffic_bytes / board.dram.bandwidth_Bps

    time_s = max(compute_s, memory_s)
    time_s *= 1 + constants.sw_time_penalty_per_branch * analysis.branches
    if analysis.control_heavy:
        time_s *= constants.sw_control_heavy_factor
    return time_s


def _ops_per_cycle(dtype: str, board: Board, constants: HlsConstants) -> float:
    """Per-core rate for `dtype`; a type the CPU lacks is emulated, slower than fp32."""
    rate = board.cpu.ops_per_cycle.get(dtype, 0)
    if rate > 0:
        return rate
    return board.cpu.ops_per_cycle["fp32"] / constants.sw_unsupported_dtype_slowdown
