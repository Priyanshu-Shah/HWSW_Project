"""The worked example of TDG/decisions.md D1: a 3x3 convolution over a 2048x2048 int16 image."""

import pytest

from helpers import CONV_3X3_2048, analyze


def test_operation_count():
    analysis = analyze(CONV_3X3_2048)
    assert analysis.arithmetic_ops == 2046 * 2046 * 9 == 37_675_044
    assert analysis.ops_in_unit("macs") == analysis.ops_in_unit("ops")


def test_off_chip_traffic_counts_each_array_once():
    assert analyze(CONV_3X3_2048).offchip_traffic_bytes == 2048 * 2048 * 2 + 18 + 2046 * 2046 * 2


def test_sliding_window_needs_a_line_buffer():
    (buffer,) = analyze(CONV_3X3_2048).buffers
    assert buffer.array_name == "img"
    # (3 - 1) rows of 2048 int16 elements, plus the 3-wide window
    assert buffer.size_bytes == (2 * 2048 + 3) * 2


def test_hint_sets_the_unroll_unit():
    assert analyze(CONV_3X3_2048).unroll_unit == 9


def test_macs_unit_counts_only_multiply_accumulates():
    raw = {**CONV_3X3_2048, "body": {"mac": 2, "add": 1}}
    analysis = analyze(raw)
    iterations = 2046 * 2046 * 9
    assert analysis.ops_in_unit("macs") == 2 * iterations
    assert analysis.ops_in_unit("ops") == 3 * iterations


def test_load_and_store_are_not_ops():
    raw = {**CONV_3X3_2048, "body": {"mac": 1, "load": 4, "store": 1}}
    assert analyze(raw).arithmetic_ops == 37_675_044


@pytest.mark.parametrize(
    ("control", "expected_interval"),
    [({"branches": 0}, 1), ({"branches": 2}, 3), ({"control_heavy": True}, 8)],
)
def test_initiation_interval(control, expected_interval):
    assert analyze({**CONV_3X3_2048, "control": control}).initiation_interval == expected_interval


def test_random_access_refetches_but_a_const_table_stays_on_chip():
    lookup = {"shape": [256], "dtype": "uint8", "dir": "in"}
    base = {
        "arrays": {"data": {"shape": [1024], "dtype": "uint8", "dir": "in"}, "table": lookup},
        "loops": [{"var": "i", "range": 1024}],
        "body": {"logic": 1},
    }
    random_data = {**base, "access": {"data": {"pattern": "random"}}}
    assert analyze(random_data).offchip_traffic_bytes == 1024 * 4 + 256

    const_table = {
        **base,
        "arrays": {**base["arrays"], "table": {**lookup, "const": True}},
        "access": {"table": {"pattern": "random"}},
    }
    analysis = analyze(const_table)
    assert analysis.offchip_traffic_bytes == 1024 + 256
    assert [b.array_name for b in analysis.buffers] == ["table"]


def test_compute_dtype_is_the_widest_array_type():
    raw = {
        "arrays": {
            "a": {"shape": [4], "dtype": "int8", "dir": "in"},
            "b": {"shape": [4], "dtype": "fp32", "dir": "out"},
        },
        "loops": [{"var": "i", "range": 4}],
        "body": {"add": 1},
    }
    assert analyze(raw).compute_dtype == "fp32"
