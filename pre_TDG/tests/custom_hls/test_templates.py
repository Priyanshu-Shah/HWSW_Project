"""Each template expands to a schema-valid D1 loop nest with the right operation count."""

import math

import pytest
from jsonschema import Draft202012Validator

from hwsw.custom_hls.constants import HlsConstants
from hwsw.custom_hls.loop_nest import parse_loop_nest
from hwsw.custom_hls.loop_nest_analysis import analyze_loop_nest
from hwsw.custom_hls.templates import TEMPLATE_OPS, TemplateParameterError, expand_template
from hwsw.validation.json_schema_rules import load_tdg_schema

# op -> (parameters, operation count by closed form)
CASES = {
    "conv2d": (
        {
            "H": 32,
            "W": 32,
            "C_in": 3,
            "C_out": 8,
            "K": 3,
            "stride": 2,
            "padding": 1,
            "dtype": "int8",
        },
        16 * 16 * 8 * 3 * 3 * 3,
    ),
    "gemm": ({"M": 8, "N": 16, "K": 32, "dtype": "int16"}, 8 * 16 * 32),
    "pool": (
        {"H": 8, "W": 8, "C": 4, "K": 2, "stride": 2, "mode": "max", "dtype": "int8"},
        4 * 4 * 4 * 2 * 2,
    ),
    "elementwise": (
        {"elements": 1000, "ops_per_element": 3, "inputs": 2, "dtype": "fp32"},
        3000,
    ),
    "reduce": ({"elements": 1000, "mode": "sum", "outputs": 10, "dtype": "int32"}, 1000),
    "memcpy": ({"bytes": 4096}, 0),
}


def test_every_template_has_a_case():
    assert set(CASES) == set(TEMPLATE_OPS)


@pytest.fixture(scope="module")
def loop_nest_validator() -> Draft202012Validator:
    schema = load_tdg_schema()
    return Draft202012Validator(
        {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": "#/$defs/loop_nest"}
    )


@pytest.mark.parametrize("op", TEMPLATE_OPS)
def test_expansion_is_a_valid_loop_nest(op, loop_nest_validator):
    params, _ = CASES[op]
    errors = list(loop_nest_validator.iter_errors(expand_template(op, params)))
    assert not errors, [error.message for error in errors]


@pytest.mark.parametrize("op", TEMPLATE_OPS)
def test_operation_count_matches_closed_form(op):
    params, expected_ops = CASES[op]
    analysis = analyze_loop_nest(parse_loop_nest(expand_template(op, params)), HlsConstants())
    assert analysis.arithmetic_ops == expected_ops


def test_batch_scales_the_work():
    params = {"M": 4, "N": 4, "K": 4, "dtype": "int8", "batch": 5}
    analysis = analyze_loop_nest(parse_loop_nest(expand_template("gemm", params)), HlsConstants())
    assert analysis.arithmetic_ops == 5 * 4 * 4 * 4


def test_depthwise_conv_uses_one_input_channel_per_filter():
    params = {
        "H": 8,
        "W": 8,
        "C_in": 16,
        "C_out": 16,
        "K": 3,
        "stride": 1,
        "groups": 16,
        "dtype": "int8",
    }
    analysis = analyze_loop_nest(parse_loop_nest(expand_template("conv2d", params)), HlsConstants())
    assert analysis.arithmetic_ops == 6 * 6 * 16 * 1 * 3 * 3


def test_memcpy_moves_whole_words():
    nest = parse_loop_nest(expand_template("memcpy", {"bytes": 10}))
    assert nest.iterations == math.ceil(10 / 4)


@pytest.mark.parametrize(
    ("op", "params", "message"),
    [
        (
            "conv2d",
            {
                "H": 8,
                "W": 8,
                "C_in": 3,
                "C_out": 8,
                "K": 3,
                "stride": 1,
                "groups": 2,
                "dtype": "int8",
            },
            "groups=2 must divide",
        ),
        (
            "conv2d",
            {"H": 2, "W": 2, "C_in": 1, "C_out": 1, "K": 3, "stride": 1, "dtype": "int8"},
            "does not fit",
        ),
        (
            "pool",
            {"H": 2, "W": 2, "C": 1, "K": 3, "stride": 1, "mode": "max", "dtype": "int8"},
            "does not fit",
        ),
        (
            "reduce",
            {"elements": 4, "mode": "sum", "outputs": 8, "dtype": "int8"},
            "exceeds elements",
        ),
        ("softmax", {}, "unknown template op"),
    ],
)
def test_inconsistent_parameters_are_explained(op, params, message):
    with pytest.raises(TemplateParameterError, match=message):
        expand_template(op, params)
