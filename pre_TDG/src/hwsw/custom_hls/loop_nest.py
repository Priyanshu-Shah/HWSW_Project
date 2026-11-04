"""Typed form of a D1 loop-nest description (`$defs/loop_nest` in the TDG schema)."""

import math
from dataclasses import dataclass, field

from hwsw.custom_hls.data_types import bytes_per_element

# Body counts that are arithmetic work. `load` and `store` are not counted as ops.
ARITHMETIC_OPS = ("mac", "add", "mul", "div", "cmp", "logic", "shift", "sqrt", "exp")

STREAMING = "streaming"
SLIDING_WINDOW = "sliding_window"
REUSE = "reuse"
RANDOM = "random"


@dataclass(frozen=True)
class ArrayDecl:
    name: str
    shape: tuple[int, ...]
    dtype: str
    direction: str  # "in" or "out"
    is_const: bool

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    @property
    def size_bytes(self) -> int:
        return self.elements * bytes_per_element(self.dtype)


@dataclass(frozen=True)
class ArrayAccess:
    pattern: str  # STREAMING | SLIDING_WINDOW | REUSE | RANDOM
    reuse_vars: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoopNest:
    arrays: dict[str, ArrayDecl]
    loops: tuple[tuple[str, int], ...]  # (variable, trip count), outermost first
    body_ops: dict[str, int]  # ops per innermost iteration
    access: dict[str, ArrayAccess]  # every array has an entry (default: streaming)
    branches: int = 0
    control_heavy: bool = False
    unrollable_vars: tuple[str, ...] = field(default=())

    @property
    def iterations(self) -> int:
        return math.prod(trip_count for _, trip_count in self.loops)

    def trip_count(self, var: str) -> int:
        return dict(self.loops)[var]


def parse_loop_nest(raw: dict) -> LoopNest:
    """Build a LoopNest from a D1 dict (as written by the LLM or by a template)."""
    arrays = {
        name: ArrayDecl(
            name=name,
            shape=tuple(spec["shape"]),
            dtype=spec["dtype"],
            direction=spec["dir"],
            is_const=spec.get("const", False),
        )
        for name, spec in raw["arrays"].items()
    }
    access = {
        name: ArrayAccess(
            pattern=raw.get("access", {}).get(name, {}).get("pattern", STREAMING),
            reuse_vars=tuple(raw.get("access", {}).get(name, {}).get("reuse", [])),
        )
        for name in arrays
    }
    control = raw.get("control", {})
    return LoopNest(
        arrays=arrays,
        loops=tuple((loop["var"], loop["range"]) for loop in raw["loops"]),
        body_ops=dict(raw["body"]),
        access=access,
        branches=control.get("branches", 0),
        control_heavy=control.get("control_heavy", False),
        unrollable_vars=tuple(raw.get("hints", {}).get("unrollable", [])),
    )
