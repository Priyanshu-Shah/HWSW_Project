"""Comparing the partition of a generated graph with the partition of its reference."""

from hwsw.benchmarks.partition_scoring import _compare, score_partition


def partition(latency, sides):
    return {
        "solver": {"status": "optimal"},
        "feasible": True,
        "latency_s": {"nominal": latency},
        "speedup": 2.0,
        "nodes": [{"id": node_id, "target": target} for node_id, target in sides.items()],
    }


def test_latency_ratio_and_same_side_over_matched_nodes():
    generated = partition(0.012, {"a": "sw", "b": "hw", "c": "hw"})
    reference = partition(0.010, {"n1": "sw", "n2": "hw", "n3": "sw"})
    score = _compare(generated, reference, {"n1": "a", "n2": "b", "n3": "c"})
    assert score["latency_ratio"] == 1.2
    assert score["same_side"] == round(2 / 3, 3)
    assert score["hw_nodes"] == 2


def test_no_partitioner_no_score(tmp_path):
    assert score_partition({}, {}, {}, tmp_path, tmp_path, command=tmp_path / "none") is None
