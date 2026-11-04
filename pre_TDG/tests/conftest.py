"""Shared fixtures: board files, example TDGs, and a way to make small edits to them."""

import copy
from pathlib import Path

import pytest

from hwsw.json_files import read_json
from hwsw.platforms import Board, load_board
from hwsw.repo_paths import TDG_EXAMPLES_DIR

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def boards_dir() -> Path:
    return FIXTURES_DIR / "platforms"


@pytest.fixture
def zcu104(boards_dir: Path) -> Board:
    return load_board("zcu104", boards_dir)


@pytest.fixture
def costed_example() -> dict:
    """The contract's worked example: graph + cost map, on zcu104."""
    return read_json(TDG_EXAMPLES_DIR / "det_pipeline.tdg.json")


@pytest.fixture
def example_graph(costed_example: dict) -> dict:
    """Same, without the cost map."""
    graph = copy.deepcopy(costed_example)
    del graph["costs"]
    return graph


@pytest.fixture(params=["sobel_edge", "matmul_batch"])
def fixture_graph(request) -> dict:
    return read_json(FIXTURES_DIR / "tdgs" / f"{request.param}.tdg.json")
