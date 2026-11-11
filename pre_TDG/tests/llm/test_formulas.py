import pytest

from hwsw.llm.formulas import evaluate_count, evaluate_formula
from hwsw.llm.schema_parts import tdg_ref, with_tdg_defs
from hwsw.validation.json_schema_rules import check_against_schema


@pytest.mark.parametrize(
    ("formula", "value"),
    [
        ("1280*720*3", 2764800),
        ("(1080-2)*(1920-2)", 2067604),
        ("2^10", 1024),
        ("1920 × 1080", 2073600),
        ("8.7e9 / 2", 4.35e9),
        (42, 42),
        (0.5, 0.5),
    ],
)
def test_arithmetic_is_evaluated(formula, value):
    assert evaluate_formula(formula) == value


@pytest.mark.parametrize(
    "formula", ["__import__('os')", "width*height", "1/0", "10**1000", "", True, "[1]"]
)
def test_anything_but_arithmetic_is_rejected(formula):
    with pytest.raises(ValueError):
        evaluate_formula(formula)


def test_counts_must_be_whole_and_non_negative():
    assert evaluate_count("6/2") == 3
    for bad in ("7/2", "-1"):
        with pytest.raises(ValueError, match="whole number"):
            evaluate_count(bad)


def test_sub_schema_keeps_only_the_defs_it_needs():
    schema = with_tdg_defs({"type": "object", "properties": {"opaque": tdg_ref("opaque")}})
    assert set(schema["$defs"]) == {"opaque", "non_negative_number", "estimate_source"}
    assert not check_against_schema({"opaque": {"sw_time_s": 0.001, "source": "llm"}}, schema)
    assert check_against_schema({"opaque": {"sw_time_s": -1, "source": "llm"}}, schema)
