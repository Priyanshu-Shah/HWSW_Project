"""What a loop nest costs in the abstract: ops, memory traffic, on-chip buffers, pipeline rate.

Conventions (shared by the HW and SW models, and by the `ops_claimed` check):
- One multiply-accumulate (`mac`) counts as ONE op. `load` and `store` are not ops.
- Every array crosses the DRAM boundary once (reads once, writes once). Reuse is assumed to be
  served from on-chip buffers, so those buffers are what the FPGA pays for in BRAM.
- A `random`-access array that is not const is re-fetched: its traffic is multiplied by
  `random_access_traffic_factor`.
"""

import math
from dataclasses import dataclass

from hwsw.custom_hls.constants import HlsConstants
from hwsw.custom_hls.data_types import bytes_per_element, widest
from hwsw.custom_hls.loop_nest import (
    ARITHMETIC_OPS,
    RANDOM,
    REUSE,
    SLIDING_WINDOW,
    ArrayDecl,
    LoopNest,
)


@dataclass(frozen=True)
class OnChipBuffer:
    array_name: str
    size_bytes: int
    pattern: str  # SLIDING_WINDOW (line buffer) | REUSE (whole array) | RANDOM (const lookup table)


@dataclass(frozen=True)
class LoopNestAnalysis:
    iterations: int
    arithmetic_ops: int  # all arithmetic body ops over all iterations; a mac counts as 1
    mac_ops: int
    body_ops: dict[str, int]  # per innermost iteration
    offchip_traffic_bytes: float
    buffers: tuple[OnChipBuffer, ...]
    initiation_interval: int  # cycles between consecutive iterations when pipelined
    compute_dtype: str  # the widest dtype among the arrays
    unroll_unit: int  # iterations that the LLM's `unrollable` hint suggests unrolling together
    branches: int
    control_heavy: bool

    def ops_in_unit(self, ops_unit: str) -> int:
        """Op count in the unit of the node's `ops_claimed` (`ops` or `macs`)."""
        return self.mac_ops if ops_unit == "macs" else self.arithmetic_ops


def analyze_loop_nest(nest: LoopNest, constants: HlsConstants) -> LoopNestAnalysis:
    iterations = nest.iterations
    return LoopNestAnalysis(
        iterations=iterations,
        arithmetic_ops=iterations * sum(nest.body_ops.get(op, 0) for op in ARITHMETIC_OPS),
        mac_ops=iterations * nest.body_ops.get("mac", 0),
        body_ops=dict(nest.body_ops),
        offchip_traffic_bytes=sum(
            _traffic_bytes(array, nest, constants) for array in nest.arrays.values()
        ),
        buffers=_on_chip_buffers(nest),
        initiation_interval=_initiation_interval(nest, constants),
        compute_dtype=widest([array.dtype for array in nest.arrays.values()]),
        unroll_unit=_unroll_unit(nest),
        branches=nest.branches,
        control_heavy=nest.control_heavy,
    )


def _traffic_bytes(array: ArrayDecl, nest: LoopNest, constants: HlsConstants) -> float:
    refetched = nest.access[array.name].pattern == RANDOM and not array.is_const
    return array.size_bytes * (constants.random_access_traffic_factor if refetched else 1)


def _on_chip_buffers(nest: LoopNest) -> tuple[OnChipBuffer, ...]:
    buffers = []
    for name, array in nest.arrays.items():
        access = nest.access[name]
        if access.pattern == SLIDING_WINDOW:
            size = min(_window_bytes(array, access.reuse_vars, nest), array.size_bytes)
        elif access.pattern == REUSE or (access.pattern == RANDOM and array.is_const):
            size = array.size_bytes
        else:
            continue
        buffers.append(OnChipBuffer(name, size, access.pattern))
    return tuple(buffers)


def _window_bytes(array: ArrayDecl, reuse_vars: tuple[str, ...], nest: LoopNest) -> int:
    """Line buffer for a sliding window. The first reuse variable is the window height.

    One reuse variable is a 1-D window (a shift register of that many elements). With more, the
    buffer holds (height - 1) full rows plus the window's width.
    """
    trips = [nest.trip_count(var) for var in reuse_vars]
    if not trips:
        window_elements = 1
    elif len(trips) == 1:
        window_elements = trips[0]
    else:
        window_elements = (trips[0] - 1) * _row_elements(array.shape) + math.prod(trips[1:])
    return window_elements * bytes_per_element(array.dtype)


def _row_elements(shape: tuple[int, ...]) -> int:
    """Elements in one row of an [..., H, W, C] array. Leading (batch-like) dims are ignored."""
    image_dims = shape[-3:]
    return math.prod(image_dims[1:])


def _initiation_interval(nest: LoopNest, constants: HlsConstants) -> int:
    interval = 1 + nest.branches * constants.initiation_interval_per_branch
    if nest.control_heavy:
        interval = max(interval, constants.control_heavy_initiation_interval)
    return interval


def _unroll_unit(nest: LoopNest) -> int:
    hinted = {var for var in nest.unrollable_vars if var in dict(nest.loops)}
    return math.prod(nest.trip_count(var) for var in hinted)
