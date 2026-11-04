from helpers import find_node, find_node_cost
from hwsw.validation import validate_tdg


def test_estimate_must_be_ordered_lo_value_hi(costed_example):
    find_node_cost(costed_example, "n2")["hw"][0]["time_s"]["lo"] = 0.5
    (issue,) = validate_tdg(costed_example).errors
    assert issue.rule == "R4"
    assert issue.location == "costs node n2 hw point p1 time_s"


def test_hw_only_node_needs_a_hardware_point(costed_example):
    find_node(costed_example, "n2")["affinity"] = "hw_only"
    find_node_cost(costed_example, "n2")["hw"] = []
    (issue,) = validate_tdg(costed_example).errors
    assert (issue.rule, issue.location) == ("R6", "node n2")


def test_iteration_node_needs_a_software_cost(costed_example):
    del find_node_cost(costed_example, "n3")["sw"]
    (issue,) = validate_tdg(costed_example).errors
    assert (issue.rule, issue.location) == ("R8", "node n3")


def test_setup_node_needs_no_software_cost(costed_example):
    find_node(costed_example, "n3")["phase"] = "setup"
    del find_node_cost(costed_example, "n3")["sw"]
    assert validate_tdg(costed_example).is_valid


def test_hw_only_node_needs_no_software_cost(costed_example):
    find_node(costed_example, "n3")["affinity"] = "hw_only"
    del find_node_cost(costed_example, "n3")["sw"]
    assert validate_tdg(costed_example).is_valid


def test_cost_entry_for_unknown_node(costed_example):
    costed_example["costs"]["nodes"].append(
        {"id": "n9", "sw": costed_example["costs"]["nodes"][0]["sw"]}
    )
    (issue,) = validate_tdg(costed_example).errors
    assert (issue.rule, issue.location) == ("R2", "costs node n9")


def test_cost_entry_for_unknown_edge(costed_example):
    costed_example["costs"]["edges"].append({"id": "e9", "interface": "axi_hp0"})
    (issue,) = validate_tdg(costed_example).errors
    assert (issue.rule, issue.location) == ("R2", "costs edge e9")


def test_node_with_two_cost_entries(costed_example):
    costed_example["costs"]["nodes"].append(costed_example["costs"]["nodes"][0])
    assert [issue.rule for issue in validate_tdg(costed_example).errors] == ["R1"]


def test_repeated_hardware_point_id(costed_example):
    find_node_cost(costed_example, "n2")["hw"][1]["point"] = "p1"
    assert [issue.rule for issue in validate_tdg(costed_example).errors] == ["R1"]


def test_ops_far_from_claim_is_a_warning(costed_example):
    find_node(costed_example, "n2")["ops_claimed"] *= 10
    report = validate_tdg(costed_example)
    assert report.is_valid
    (issue,) = report.warnings
    assert (issue.rule, issue.location) == ("R9", "node n2")


def test_ops_within_factor_two_are_fine(costed_example):
    find_node(costed_example, "n2")["ops_claimed"] = 1.84e6 * 1.9
    assert not validate_tdg(costed_example).issues
