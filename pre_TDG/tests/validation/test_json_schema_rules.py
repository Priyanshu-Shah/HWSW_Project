"""The contract schema rejects malformed documents, with a message that says what is wrong."""

import pytest

from hwsw.validation import validate_tdg

# (what is wrong, how to break the example graph, text the message must contain)
MALFORMED_GRAPHS = [
    (
        "numeric schema_version",
        lambda g: g["graph"].update(schema_version=0.2),
        "graph.schema_version",
    ),
    (
        "streaming without period",
        lambda g: g["arrival"].pop("period_s"),
        "'period_s' is a required property",
    ),
    ("one_shot with period", lambda g: g["arrival"].update(kind="one_shot"), "arrival"),
    (
        "control edge carrying bytes",
        lambda g: g["edges"].append(
            {"id": "e9", "src": "n1", "dst": "n3", "kind": "control", "bytes": 5}
        ),
        "edges[2]",
    ),
    (
        "opaque node that also has a template",
        lambda g: g["nodes"][0].update(template=g["nodes"][2]["template"]),
        "nodes[0]",
    ),
    (
        "loop_nest without ops_claimed",
        lambda g: g["nodes"][1].pop("ops_claimed"),
        "'ops_claimed' is a required property",
    ),
    ("unknown field on a node", lambda g: g["nodes"][0].update(colour="red"), "colour"),
    (
        "conv2d without C_out",
        lambda g: g["nodes"][2]["template"]["params"].pop("C_out"),
        "nodes[2].template.params: 'C_out' is a required property",
    ),
    (
        "gemm op with conv2d parameters",
        lambda g: g["nodes"][2]["template"].update(op="gemm"),
        "nodes[2].template",
    ),
    (
        "unsupported dtype",
        lambda g: g["nodes"][1]["loop_nest"]["arrays"]["img"].update(dtype="int4"),
        "'int4' is not one of",
    ),
    (
        "formula instead of literal bytes",
        lambda g: g["edges"][0].update(bytes="1280*720*3"),
        "edges[0].bytes",
    ),
    ("unknown body op", lambda g: g["nodes"][1]["loop_nest"]["body"].update(fma=1), "fma"),
    ("node id with a dash", lambda g: g["nodes"][0].update(id="n-1"), "nodes[0].id"),
    ("no nodes", lambda g: g.update(nodes=[]), "nodes"),
]


@pytest.mark.parametrize(
    ("breakage", "mutate", "expected_text"),
    MALFORMED_GRAPHS,
    ids=[breakage for breakage, _, _ in MALFORMED_GRAPHS],
)
def test_malformed_graph_is_rejected(example_graph, breakage, mutate, expected_text):
    mutate(example_graph)
    report = validate_tdg(example_graph)
    assert not report.is_valid
    assert any(issue.rule == "SCHEMA" and expected_text in str(issue) for issue in report.issues), (
        report.format()
    )


@pytest.mark.parametrize(
    ("breakage", "mutate"),
    [
        ("estimate without source", lambda c: c["costs"]["nodes"][1]["sw"]["time_s"].pop("source")),
        (
            "unknown resource key",
            lambda c: c["costs"]["nodes"][1]["hw"][0]["resources"].update(bram=2),
        ),
        ("negative time", lambda c: c["costs"]["nodes"][1]["sw"]["time_s"].update(value=-1)),
        ("edge override with only an id", lambda c: c["costs"]["edges"].append({"id": "e1"})),
    ],
)
def test_malformed_cost_map_is_rejected(costed_example, breakage, mutate):
    mutate(costed_example)
    assert not validate_tdg(costed_example).is_valid
