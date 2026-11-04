import pytest

from helpers import CONV_3X3_2048, analyze
from hwsw.custom_hls.constants import HlsConstants
from hwsw.custom_hls.sw_time import software_time_s


def test_compute_bound_time_for_the_worked_example(zcu104):
    time_s = software_time_s(analyze(CONV_3X3_2048), zcu104, HlsConstants())
    # 37.7M ops / (1.2 GHz * 1 core * 8 int16 ops/cycle * 0.3 efficiency)
    assert time_s == pytest.approx(37_675_044 / 2.88e9)


def test_memory_bound_when_there_is_no_arithmetic(zcu104):
    copy = {
        "arrays": {
            "a": {"shape": [1_000_000], "dtype": "int32", "dir": "in"},
            "b": {"shape": [1_000_000], "dtype": "int32", "dir": "out"},
        },
        "loops": [{"var": "i", "range": 1_000_000}],
        "body": {"load": 1, "store": 1},
    }
    assert software_time_s(analyze(copy), zcu104, HlsConstants()) == pytest.approx(
        8_000_000 / 1.0e10
    )


def test_branches_and_irregular_control_slow_the_cpu_down(zcu104):
    constants = HlsConstants()
    plain = software_time_s(analyze(CONV_3X3_2048), zcu104, constants)
    branchy = software_time_s(
        analyze({**CONV_3X3_2048, "control": {"branches": 2}}), zcu104, constants
    )
    heavy = software_time_s(
        analyze({**CONV_3X3_2048, "control": {"control_heavy": True}}), zcu104, constants
    )
    assert branchy == pytest.approx(plain * 1.1)
    assert heavy == pytest.approx(plain * 1.5)


def test_more_cores_used_means_less_time(zcu104):
    one = software_time_s(analyze(CONV_3X3_2048), zcu104, HlsConstants(sw_cores_used=1))
    four = software_time_s(analyze(CONV_3X3_2048), zcu104, HlsConstants(sw_cores_used=4))
    assert four == pytest.approx(one / 4)


def test_type_the_cpu_lacks_is_emulated_slower_than_fp32(zcu104):
    def vector_add(dtype):
        return analyze(
            {
                "arrays": {
                    "a": {"shape": [1 << 20], "dtype": dtype, "dir": "in"},
                    "b": {"shape": [1 << 20], "dtype": dtype, "dir": "out"},
                },
                "loops": [{"var": "i", "range": 1 << 20}],
                "body": {"add": 64},
            }
        )

    fp32 = software_time_s(vector_add("fp32"), zcu104, HlsConstants())
    fp16 = software_time_s(vector_add("fp16"), zcu104, HlsConstants())  # ops_per_cycle["fp16"] == 0
    assert fp16 == pytest.approx(fp32 * 4)
