import copy

import pytest

from helpers import find_node, find_node_cost
from hwsw.custom_hls import CostEstimationError, estimate_costs
from hwsw.platforms import load_board


def test_every_node_gets_a_cost_entry(example_graph, zcu104):
    costed = estimate_costs(example_graph, zcu104)
    assert [cost["id"] for cost in costed["costs"]["nodes"]] == ["n1", "n2", "n3"]
    assert costed["costs"]["edges"] == []


def test_input_is_not_modified(example_graph, zcu104):
    before = copy.deepcopy(example_graph)
    estimate_costs(example_graph, zcu104)
    assert example_graph == before


def test_opaque_node_copies_its_own_time_into_the_cost_map(example_graph, zcu104):
    cost = find_node_cost(estimate_costs(example_graph, zcu104), "n1")
    assert cost["sw"]["time_s"] == {
        "value": 0.002,
        "lo": 0.001,
        "hi": 0.003,
        "source": "library_default",
    }
    assert "hw" not in cost and "ops_estimated" not in cost


def test_ops_are_reported_in_the_nodes_own_unit(example_graph, zcu104):
    find_node(example_graph, "n2")["ops_unit"] = "macs"
    costed = estimate_costs(example_graph, zcu104)
    assert find_node_cost(costed, "n2")["ops_estimated"] == 320 * 320 * 3 * 4  # mac:4 per iteration


def test_sw_only_node_gets_no_hardware_points(example_graph, zcu104):
    find_node(example_graph, "n3")["affinity"] = "sw_only"
    cost = find_node_cost(estimate_costs(example_graph, zcu104), "n3")
    assert "sw" in cost and "hw" not in cost


def test_hw_only_node_gets_no_software_cost(example_graph, zcu104):
    find_node(example_graph, "n3")["affinity"] = "hw_only"
    cost = find_node_cost(estimate_costs(example_graph, zcu104), "n3")
    assert "hw" in cost and "sw" not in cost


def test_hardware_points_carry_the_fields_the_partitioner_reads(example_graph, zcu104):
    point = find_node_cost(estimate_costs(example_graph, zcu104), "n3")["hw"][0]
    assert set(point) == {"point", "parallelism", "time_s", "ii_s", "resources"}
    assert set(point["time_s"]) == {"value", "lo", "hi", "source"}
    assert set(point["resources"]) == {"lut", "ff", "dsp", "bram18k"}
    assert point["time_s"]["lo"] < point["time_s"]["value"] < point["time_s"]["hi"]


def test_all_problems_are_reported_at_once(example_graph, zcu104):
    find_node(example_graph, "n3")["template"]["params"]["groups"] = 2  # C_in=3 is not divisible
    example_graph["nodes"].append(
        {
            "id": "n4",
            "name": "pool",
            "phase": "iteration",
            "affinity": "any",
            "kind": "template",
            "template": {
                "op": "pool",
                "params": {
                    "H": 2,
                    "W": 2,
                    "C": 1,
                    "K": 3,
                    "stride": 1,
                    "mode": "max",
                    "dtype": "int8",
                },
            },
        }
    )
    with pytest.raises(CostEstimationError) as failure:
        estimate_costs(example_graph, zcu104)
    assert [problem.split(":")[0] for problem in failure.value.problems] == ["node n3", "node n4"]


def test_board_must_match_the_tdg(example_graph, boards_dir):
    with pytest.raises(CostEstimationError, match="targets platform 'zcu104'"):
        estimate_costs(example_graph, load_board("kv260", boards_dir))


def test_a_faster_board_gives_a_faster_software_time(boards_dir, example_graph, zcu104):
    slow = load_board("pynq_z2", boards_dir)
    slow_graph = {**example_graph, "platform": "pynq_z2"}
    fast_time = find_node_cost(estimate_costs(example_graph, zcu104), "n3")["sw"]["time_s"]["value"]
    slow_time = find_node_cost(estimate_costs(slow_graph, slow), "n3")["sw"]["time_s"]["value"]
    assert slow_time > fast_time
