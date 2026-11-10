from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from conftest import GOLDEN, GOLDEN_IDS, ROOT, golden_path
from schemas import load_case, load_graph, load_platform
from schemas.common import Param
from schemas.export import MODELS, OUT_DIR, render
from schemas.ir import TaskGraph

BASE = {
    "id": "g",
    "provenance": {"kind": "synthetic", "source": "test"},
    "platform": {"id": "p"},
    "nodes": [
        {"id": "a", "implementations": [{"id": "sw", "pe_type": "cpu", "cost": {"time_ms": 1}}]},
        {"id": "b", "implementations": [{"id": "sw", "pe_type": "cpu", "cost": {"time_ms": 1}}]},
    ],
    "edges": [{"src": "a", "dst": "b", "bytes": 10}],
}


def graph(**over):
    d = json.loads(json.dumps(BASE))
    d.update(over)
    return d


@pytest.mark.parametrize("name", GOLDEN_IDS)
def test_goldens_load_with_their_platform(name):
    g, p = load_case(golden_path(name))
    assert g.id == name and g.platform.id == p.id


def test_valid_minimal_graph():
    TaskGraph.model_validate(graph())


@pytest.mark.parametrize(
    "over, msg",
    [
        ({"edges": [{"src": "a", "dst": "b", "bytes": 1}, {"src": "b", "dst": "a", "bytes": 1}]}, "cycle"),
        ({"edges": [{"src": "a", "dst": "zz", "bytes": 1}]}, "unknown node"),
        ({"edges": [{"src": "a", "dst": "a", "bytes": 1}]}, "self-loop"),
        ({"nodes": BASE["nodes"] + [BASE["nodes"][0]]}, "duplicate node ids"),
        ({"arrival": {"kind": "streaming"}}, "needs period_ms"),
        ({"surprise": 1}, "Extra inputs"),
    ],
)
def test_invalid_graphs_rejected(over, msg):
    with pytest.raises(ValidationError, match=msg):
        TaskGraph.model_validate(graph(**over))


def test_param_shorthand_and_uncertainty_bounds():
    assert Param.model_validate(3).expr == 3
    assert Param.model_validate("2 * w").expr == "2 * w"
    assert Param.model_validate({"expr": 1, "unc": [0.9, 1.2]}).unc == (0.9, 1.2)
    with pytest.raises(ValidationError):
        Param.model_validate({"expr": 1, "unc": [1.1, 1.2]})  # nominal must lie inside the interval


@pytest.mark.parametrize("name", list(MODELS))
def test_committed_json_schemas_are_up_to_date(name):
    committed = (OUT_DIR / f"{name}.schema.json").read_text(encoding="utf-8")
    assert committed == render(name), "run `python -m schemas.export` and commit the result"


def test_goldens_validate_against_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    ir = json.loads((OUT_DIR / "ir.schema.json").read_text(encoding="utf-8"))
    for name in GOLDEN_IDS:
        jsonschema.validate(json.loads(golden_path(name).read_text(encoding="utf-8")), ir)


def test_synthetic_files_say_so_in_their_name():
    """Project rule: anything synthetic is labelled synthetic in its file name."""
    files = list(GOLDEN.glob("*.json")) + list((ROOT / "platforms").glob("*.json"))
    assert files
    for f in files:
        loader = load_platform if f.parent.name == "platforms" else load_graph
        obj = loader(f)
        if obj.provenance.kind in ("synthetic", "mixed"):
            assert ".synthetic." in f.name, f"{f} has {obj.provenance.kind} provenance but no .synthetic. in its name"
