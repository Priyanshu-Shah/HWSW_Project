from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from estimate import Option, Problem, annotate
from schemas import load_case
from schemas.ir import TaskGraph
from schemas.platform import Platform

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "examples" / "golden"
GOLDEN_IDS = ["toy_forkjoin", "collision_detection", "wifi_tx"]


def golden_path(name: str) -> Path:
    return GOLDEN / f"{name}.synthetic.json"


def golden_problem(name: str) -> Problem:
    return annotate(*load_case(golden_path(name)))


def golden_variant(name: str, edit: Callable[[dict[str, Any]], None]) -> Problem:
    """Load a golden graph as a dict, apply `edit`, and annotate it with its platform."""
    graph, platform = load_case(golden_path(name))
    d = copy.deepcopy(json.loads(golden_path(name).read_text(encoding="utf-8")))
    edit(d)
    return annotate(TaskGraph.model_validate(d), platform)


def platform_variant(name: str, edit: Callable[[dict[str, Any]], None]) -> Problem:
    graph, platform = load_case(golden_path(name))
    d = platform.model_dump(mode="json")
    edit(d)
    return annotate(graph, Platform.model_validate(d))


def m(**assign: str) -> dict[str, Option]:
    """Mapping shorthand: m(A="cpu0/sw", B="fpga0/hw")."""
    return {n: Option(*v.split("/")) for n, v in assign.items()}


@pytest.fixture
def toy() -> Problem:
    return golden_problem("toy_forkjoin")


@pytest.fixture
def collision() -> Problem:
    return golden_problem("collision_detection")


@pytest.fixture
def wifi() -> Problem:
    return golden_problem("wifi_tx")


# Hand-checkable toy mappings (see examples/golden/toy_forkjoin.synthetic.json)
TOY_ALL_SW = m(A="cpu0/sw", B="cpu0/sw", C="cpu0/sw", D="cpu0/sw")
TOY_B_HW = m(A="cpu0/sw", B="fpga0/hw", C="cpu0/sw", D="cpu0/sw")
TOY_C_HW = m(A="cpu0/sw", B="cpu0/sw", C="fpga0/hw", D="cpu0/sw")
TOY_BOTH_HW = m(A="cpu0/sw", B="fpga0/hw", C="fpga0/hw", D="cpu0/sw")
