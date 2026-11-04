import pytest

from helpers import find_node
from hwsw.validation import validate_tdg


@pytest.mark.parametrize(
    ("breakage", "mutate", "expected_text"),
    [
        (
            "repeated loop variable",
            lambda nest: nest["loops"].append({"var": "x", "range": 2}),
            "loop variable 'x' is declared 2 times",
        ),
        (
            "reuse of an unknown loop variable",
            lambda nest: nest["access"]["frame"].update(reuse=["q"]),
            "unknown loop variable 'q'",
        ),
        (
            "unrollable hint on an unknown loop variable",
            lambda nest: nest["hints"].update(unrollable=["q"]),
            "hints.unrollable uses unknown loop variable 'q'",
        ),
        (
            "access for an array that does not exist",
            lambda nest: nest["access"].update(ghost={"pattern": "streaming"}),
            "array 'ghost', which is not in arrays",
        ),
    ],
)
def test_loop_nest_naming_errors(example_graph, breakage, mutate, expected_text):
    mutate(find_node(example_graph, "n2")["loop_nest"])
    report = validate_tdg(example_graph)
    assert [issue.rule for issue in report.errors] == ["R12"]
    assert expected_text in report.errors[0].message
    assert report.errors[0].location == "node n2"
