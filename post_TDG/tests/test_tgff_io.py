"""TGFF parser against the reference output shipped with TGFF 3.5 and against E3S (same file format)."""

from __future__ import annotations

import pytest

from conftest import ROOT
from datagen.tgff_io import parse_tgff, read_tgff

TGFF_EXAMPLE = ROOT / "third_party" / "tgff" / "tgff-3.5" / "examples" / "kseries_parallel.tgff"
E3S_FILE = ROOT / "e3s" / "e3s-0.9" / "auto-indust-cords.tgff"


@pytest.mark.skipif(not TGFF_EXAMPLE.exists(), reason="TGFF not vendored")
def test_reads_shipped_tgff_example():
    f = read_tgff(TGFF_EXAMPLE)
    assert f.hyperperiod == 1275
    assert len(f.graphs) == 4  # tg_cnt 4 in kseries_parallel.tgffopt
    g = f.graphs[0]
    assert g.period == 1275 and len(g.tasks) >= 20  # task_cnt 20 is a lower bound for series-parallel graphs
    names = set(g.tasks)
    assert all(s in names and d in names for _, s, d, _ in g.arcs)


@pytest.mark.skipif(not E3S_FILE.exists(), reason="E3S not present")
def test_reads_e3s_processor_tables():
    f = read_tgff(E3S_FILE)
    procs = f.tables["PROC"]
    assert len(procs) == 17
    p0 = procs[0]
    assert "price" in p0.attrs
    assert p0.columns[:4] == ["type", "version", "valid", "task_time"]
    assert len(f.graphs) == 4  # auto-indust: 4 task graphs


def test_table_header_and_rows():
    text = """@COMMUN 0 {
#        price
       70.1

#------
# type    exec_time
      0     47.4
      1     44.5
}
"""
    t = parse_tgff(text).tables["COMMUN"][0]
    assert t.attrs == {"price": 70.1}
    assert t.column("exec_time") == {0: 47.4, 1: 44.5}
