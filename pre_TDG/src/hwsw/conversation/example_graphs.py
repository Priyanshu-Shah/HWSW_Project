"""Ready-made graph-only TDGs, for the stub agent and for front ends' "load example" button."""

from pathlib import Path

from hwsw.json_files import read_json
from hwsw.repo_paths import FIXTURE_TDGS_DIR, TDG_EXAMPLES_DIR

_EXAMPLE_DIRS = (TDG_EXAMPLES_DIR, FIXTURE_TDGS_DIR)
_SUFFIX = ".tdg.json"


def list_example_names() -> list[str]:
    return sorted({path.name.removesuffix(_SUFFIX) for path in _example_files()})


def load_example_graph(name: str) -> dict:
    """The example named `name` without its cost map, so it can be costed from scratch."""
    for path in _example_files():
        if path.name == name + _SUFFIX:
            return _graph_only(read_json(path))
    raise KeyError(f"no example named '{name}'; available: {', '.join(list_example_names())}")


def _example_files() -> list[Path]:
    return [path for folder in _EXAMPLE_DIRS for path in sorted(folder.glob(f"*{_SUFFIX}"))]


def _graph_only(tdg: dict) -> dict:
    return {key: value for key, value in tdg.items() if key not in ("costs", "$schema")}
