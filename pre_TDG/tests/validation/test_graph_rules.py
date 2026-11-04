from hwsw.validation import validate_tdg


def test_missing_edge_end_names_the_edge_and_lists_real_ids(example_graph):
    example_graph["edges"][0]["dst"] = "n9"
    (issue,) = validate_tdg(example_graph).errors
    assert issue.rule == "R2"
    assert issue.location == "edge e1"
    assert "dst 'n9' does not exist" in issue.message
    assert "existing node ids: n1, n2, n3" in issue.message


def test_duplicate_node_ids_are_rejected(example_graph):
    example_graph["nodes"].append(dict(example_graph["nodes"][1]))  # a second node with id n2
    report = validate_tdg(example_graph)
    assert [issue.rule for issue in report.errors] == ["R1"]
    assert "'n2' is used 2 times" in report.errors[0].message


def test_duplicate_edge_ids_are_rejected(example_graph):
    example_graph["edges"][1]["id"] = "e1"
    assert [issue.rule for issue in validate_tdg(example_graph).errors] == ["R1"]


def test_cycle_is_rejected_and_shown(example_graph):
    example_graph["edges"].append(
        {"id": "e3", "src": "n3", "dst": "n2", "kind": "control", "bytes": 0}
    )
    (issue,) = validate_tdg(example_graph).errors
    assert issue.rule == "R3"
    assert "n2 -> n3 -> n2" in issue.message


def test_control_edges_count_towards_cycles(example_graph):
    example_graph["edges"].append(
        {"id": "e3", "src": "n3", "dst": "n1", "kind": "control", "bytes": 0}
    )
    assert [issue.rule for issue in validate_tdg(example_graph).errors] == ["R3"]
