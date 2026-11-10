import copy

import pytest
from hwsw.repo_paths import PARTITION_COMMAND


@pytest.fixture
def sobel(client) -> dict:
    return client.get("/api/examples/sobel_edge").json()


def test_boards_come_from_the_configured_folder(client):
    body = client.get("/api/boards").json()
    assert body["source"] == "fixtures"
    assert [board["id"] for board in body["boards"]] == ["kv260", "pynq_z2", "zcu104"]
    assert all(board["summary"] for board in body["boards"])


def test_examples_are_listed_and_graph_only(client):
    names = client.get("/api/examples").json()["names"]
    assert {"det_pipeline", "sobel_edge", "matmul_batch"} <= set(names)
    assert "costs" not in client.get("/api/examples/det_pipeline").json()


def test_unknown_example_is_a_404(client):
    response = client.get("/api/examples/nope")
    assert response.status_code == 404
    assert "no example named 'nope'" in response.json()["detail"]


def test_validate_accepts_a_good_tdg(client, sobel):
    body = client.post("/api/tdg/validate", json={"tdg": sobel}).json()
    assert body == {"valid": True, "issues": []}


def test_validate_reports_problems_with_their_rule(client, sobel):
    sobel["edges"][0]["dst"] = "n9"
    body = client.post("/api/tdg/validate", json={"tdg": sobel}).json()
    assert body["valid"] is False
    (issue,) = body["issues"]
    assert issue["rule"] == "R2" and issue["severity"] == "error"
    assert "dst 'n9' does not exist" in issue["message"]


def test_validate_reports_an_unknown_board(client, sobel):
    sobel["platform"] = "no_such_board"
    body = client.post("/api/tdg/validate", json={"tdg": sobel}).json()
    assert body["valid"] is False
    assert "unknown platform 'no_such_board'" in body["issues"][0]["message"]


def test_cost_adds_the_cost_map_and_bounds(client, sobel):
    body = client.post("/api/tdg/cost", json={"tdg": sobel}).json()
    assert body["valid"] is True
    costs = body["costed_tdg"]["costs"]["nodes"]
    assert [cost["id"] for cost in costs] == ["n1", "n2", "n3", "n4", "n5"]
    assert body["bounds"]["best_case_s"] < body["bounds"]["all_software_s"]
    assert body["bounds"]["deadline_s"] == pytest.approx(0.016667)


def test_cost_refuses_an_invalid_tdg_without_costing(client, sobel):
    sobel["objective"] = {"energy": 1.0}
    body = client.post("/api/tdg/cost", json={"tdg": sobel}).json()
    assert body["valid"] is False
    assert body["costed_tdg"] is None and body["bounds"] is None
    assert [issue["rule"] for issue in body["issues"]] == ["R10"]


def test_cost_reports_template_problems_as_issues(client, sobel):
    sobel["nodes"][3]["template"] = {
        "op": "pool",
        "params": {"H": 2, "W": 2, "C": 1, "K": 3, "stride": 1, "mode": "max", "dtype": "int8"},
    }
    body = client.post("/api/tdg/cost", json={"tdg": sobel}).json()
    assert body["valid"] is False
    assert body["issues"][0]["rule"] == "COST"
    assert "node n4" in body["issues"][0]["message"]


def test_cost_can_be_repeated_on_an_already_costed_tdg(client, sobel):
    once = client.post("/api/tdg/cost", json={"tdg": sobel}).json()["costed_tdg"]
    twice = client.post("/api/tdg/cost", json={"tdg": copy.deepcopy(once)}).json()
    assert twice["valid"] is True
    assert twice["costed_tdg"] == once


def test_bounds_of_a_costed_tdg(client, sobel):
    costed = client.post("/api/tdg/cost", json={"tdg": sobel}).json()["costed_tdg"]
    body = client.post("/api/tdg/bounds", json={"tdg": costed}).json()
    assert body["all_software_s"] > body["best_case_s"] > 0


def test_bounds_of_a_tdg_without_costs_is_a_422(client, sobel):
    response = client.post("/api/tdg/bounds", json={"tdg": sobel})
    assert response.status_code == 422
    assert "not a costed TDG" in response.json()["detail"]


@pytest.mark.skipif(not PARTITION_COMMAND.is_file(), reason="post_TDG partitioner not installed")
def test_partition_of_a_costed_tdg(client, sobel):
    costed = client.post("/api/tdg/cost", json={"tdg": sobel}).json()["costed_tdg"]
    body = client.post("/api/tdg/partition", json={"tdg": costed}).json()
    assert body["error"] is None
    assert {n["id"] for n in body["result"]["nodes"]} == {n["id"] for n in sobel["nodes"]}
    assert any(line.startswith("hardware:") for line in body["summary"])


def test_partition_needs_a_costed_tdg(client, sobel):
    body = client.post("/api/tdg/partition", json={"tdg": sobel}).json()
    assert body["result"] is None and "cost the graph first" in body["error"]
