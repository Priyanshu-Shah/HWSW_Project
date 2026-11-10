from __future__ import annotations

from datagen import random_dag


def test_random_dag_is_deterministic_and_labelled_synthetic():
    a, b = random_dag(10, seed=3), random_dag(10, seed=3)
    assert a == b
    assert a.provenance.kind == "synthetic" and "synthetic" in a.id
    assert random_dag(10, seed=4) != a


def test_random_dag_every_node_has_a_software_option():
    g = random_dag(25, seed=0)
    assert all(any(i.id == "sw" for i in n.implementations) for n in g.nodes)
