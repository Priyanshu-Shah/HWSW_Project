from helpers import rules_of
from hwsw.validation import validate_tdg


def test_objective_other_than_latency_only_is_rejected(example_graph):
    example_graph["objective"] = {"latency": 0.5, "energy": 0.5}
    report = validate_tdg(example_graph)
    assert rules_of(report) == {"R10"}
    assert not report.is_valid


def test_deadline_longer_than_period_is_a_warning(example_graph):
    example_graph["constraints"]["deadline_s"] = 0.05  # period is 0.0333
    report = validate_tdg(example_graph)
    assert report.is_valid
    assert [issue.rule for issue in report.warnings] == ["R11"]


def test_deadline_equal_to_period_is_fine(example_graph):
    example_graph["constraints"]["deadline_s"] = example_graph["arrival"]["period_s"]
    assert not validate_tdg(example_graph).issues


def test_no_deadline_is_fine(example_graph):
    del example_graph["constraints"]
    assert not validate_tdg(example_graph).issues
