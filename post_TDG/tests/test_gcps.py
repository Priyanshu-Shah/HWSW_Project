"""GCPS reproduction: LSSP reproduces Table 1 of Zheng et al. 2021 exactly; generator matches Table 2."""

from __future__ import annotations

import numpy as np
import pytest

from solvers.gcps.instance import TGFF_EXE, fig3_example, generate
from solvers.gcps.lssp import greedy_area, lssp

HW = {1, 2, 6, 9, 11}  # Table 1 partition (1-based)
TABLE1 = {  # task: (start, finish)
    1: (0, 2), 2: (0, 3), 3: (0, 4), 4: (5, 10), 5: (10, 13), 6: (14, 24), 7: (20, 23), 8: (16, 20), 9: (16, 23),
    10: (27, 32), 11: (23, 32),
}


def test_lssp_reproduces_table1():
    inst = fig3_example()
    y = np.array([1 if i + 1 in HW else 0 for i in range(inst.n)])
    s = lssp(inst, y, detail=True)
    assert s.t_sl == 32
    for t, (a, b) in TABLE1.items():
        assert (s.start[t - 1], s.finish[t - 1]) == (a, b), t
    bus = {(i + 1, j + 1): (a, b) for i, j, a, b in s.bus}
    assert bus[(4, 6)] == (10, 14) and bus[(5, 9)] == (14, 16) and bus[(5, 8)] == (16, 16)
    assert inst.t_max / s.t_sl == pytest.approx(83 / 32)
    assert inst.area[y.astype(bool)].sum() == 17 <= inst.C


def test_greedy_area_stops_at_first_overflow():
    y = greedy_area(np.array([0.9, 0.8, 0.7]), np.array([5.0, 10.0, 1.0]), cap=12.0)
    assert y.tolist() == [1, 0, 0]  # task 3 would still fit, but the rule stops at the first overflow


@pytest.mark.skipif(not TGFF_EXE.exists(), reason="TGFF not built (scripts/build_tgff.py)")
def test_generator_follows_table2():
    insts = generate(20, "CT2", 1 / 3, seed=0, graphs=3)
    assert len(insts) == 3 and all(i.n == 20 for i in insts)
    i = insts[0]
    assert 2e3 <= i.ts.min() and i.ts.max() <= 1.2e4 and 200 <= i.th.min() and i.th.max() <= 1200
    assert all(100 <= c <= 600 for *_, c in i.edges)
    assert i.C == pytest.approx(i.area.sum() / 3)
    assert all(a < b for a, b, _ in i.edges)  # topological numbering
    assert generate(20, "CT2", 1 / 3, seed=0, graphs=3)[0].ts.tolist() == i.ts.tolist()
