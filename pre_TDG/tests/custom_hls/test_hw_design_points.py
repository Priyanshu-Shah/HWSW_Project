import math

import pytest

from helpers import CONV_3X3_2048, analyze
from hwsw.custom_hls.constants import HlsConstants
from hwsw.custom_hls.hw_design_points import hardware_design_points
from hwsw.custom_hls.templates import expand_template


def test_conv_points_follow_the_unroll_hint(zcu104):
    points = hardware_design_points(analyze(CONV_3X3_2048), zcu104, HlsConstants())
    assert [(point.point_id, point.parallelism) for point in points] == [
        ("p1", 9),
        ("p2", 36),
        ("p3", 144),
    ]


def test_compute_bound_point_follows_the_cycle_count(zcu104):
    p9, *_ = hardware_design_points(analyze(CONV_3X3_2048), zcu104, HlsConstants())
    expected_cycles = math.ceil(2046 * 2046 * 9 / 9)
    assert p9.time_s == pytest.approx((expected_cycles + 20) / 2.0e8)
    assert p9.time_s == pytest.approx(0.0209, abs=1e-4)


def test_more_parallelism_runs_into_the_memory_limit(zcu104):
    """P=36 would compute in 5.2 ms, but moving 16.8 MB over the 2 GB/s interface takes 8.4 ms."""
    p9, p36, p144 = hardware_design_points(analyze(CONV_3X3_2048), zcu104, HlsConstants())
    memory_s = (2048 * 2048 * 2 + 18 + 2046 * 2046 * 2) / 2.0e9
    assert p36.time_s == pytest.approx(memory_s)
    assert p144.time_s == pytest.approx(memory_s)
    assert p36.time_s < p9.time_s


def test_resources_of_the_worked_example(zcu104):
    p9, *_ = hardware_design_points(analyze(CONV_3X3_2048), zcu104, HlsConstants())
    assert p9.resources["dsp"] == 9  # 9 int16 multiplies, one DSP each
    assert (
        p9.resources["bram18k"] == 4
    )  # 8198-byte line buffer / 2304 bytes; the 18 B kernel is in registers


def test_memory_bound_node_stops_speeding_up(zcu104):
    """A tiny per-element computation hits the memory bandwidth, so more parallelism is useless."""
    analysis = analyze(
        {
            "arrays": {
                "a": {"shape": [1 << 24], "dtype": "uint8", "dir": "in"},
                "b": {"shape": [1 << 24], "dtype": "uint8", "dir": "out"},
            },
            "loops": [{"var": "i", "range": 1 << 24}],
            "body": {"add": 1},
        }
    )
    *_, p16 = hardware_design_points(analysis, zcu104, HlsConstants())
    memory_s = analysis.offchip_traffic_bytes / 2.0e9  # the board's default interface
    assert p16.time_s == pytest.approx(memory_s)
    assert p16.ii_s == pytest.approx(memory_s)


def test_parallelism_never_exceeds_the_iteration_count(zcu104):
    tiny = {
        "arrays": {"a": {"shape": [3], "dtype": "int8", "dir": "in"}},
        "loops": [{"var": "i", "range": 3}],
        "body": {"add": 1},
    }
    assert [
        p.parallelism for p in hardware_design_points(analyze(tiny), zcu104, HlsConstants())
    ] == [1, 3]


def test_unfitting_hint_is_ignored(boards_dir):
    from hwsw.platforms import load_board

    pynq = load_board("pynq_z2", boards_dir)  # 220 DSPs; the gemm hint (k=256) needs 256+
    nest = expand_template("gemm", {"M": 256, "N": 256, "K": 256, "dtype": "int16"})
    points = hardware_design_points(analyze(nest), pynq, HlsConstants())
    assert [p.parallelism for p in points] == [1, 4, 16]
    assert all(p.resources["dsp"] <= 220 for p in points)


def test_node_that_never_fits_has_no_hardware_option(boards_dir):
    from hwsw.platforms import load_board

    pynq = load_board("pynq_z2", boards_dir)
    huge_weights = expand_template("gemm", {"M": 1024, "N": 4096, "K": 4096, "dtype": "fp32"})
    assert hardware_design_points(analyze(huge_weights), pynq, HlsConstants()) == []


def test_every_point_fits_the_board(zcu104):
    nest = expand_template(
        "conv2d",
        {
            "H": 64,
            "W": 64,
            "C_in": 16,
            "C_out": 32,
            "K": 3,
            "stride": 1,
            "padding": 1,
            "dtype": "int8",
        },
    )
    for point in hardware_design_points(analyze(nest), zcu104, HlsConstants()):
        assert all(amount <= zcu104.fpga.capacity[key] for key, amount in point.resources.items())
