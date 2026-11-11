"""Binary HW/SW instances in the GCPS model (Zheng, Liang & Xiong, DAES 25, 2021) and their TGFF-based generator.

Model (docs/survey/slices/07_gcps_reproduction.md §1): one CPU runs SW tasks sequentially; every HW task has its own
unit (HW tasks run in parallel); a single shared bus carries data between tasks on different sides; total HW area
<= C = AP * sum(area). Costs are in ns.

Generator (§2, Table 2): TGFF 3.5 topology (`task_degree 3 3`, our choice: the paper gives none), costs uniform over
SW 2000–12000, HW 200–1200, area 100–400 and the chosen communication class:
CT1 2–100, CT2 100–600, CT3 800–2000. Only graphs with exactly N tasks are kept. All instances are SYNTHETIC.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datagen.tgff_io import read_tgff

ROOT = Path(__file__).resolve().parents[2]
TGFF_EXE = ROOT / "build" / "tgff" / ("tgff.exe" if sys.platform == "win32" else "tgff")

CT = {"CT1": (2.0, 100.0), "CT2": (100.0, 600.0), "CT3": (800.0, 2000.0)}
SW_RANGE, HW_RANGE, AREA_RANGE = (2e3, 1.2e4), (2e2, 1.2e3), (100.0, 400.0)


@dataclass
class BinInstance:
    """A GCPS-model instance. Tasks are 0..n-1 in a topological order; edges are (i, j, comm_ns)."""

    name: str
    ts: np.ndarray
    th: np.ndarray
    area: np.ndarray
    edges: list[tuple[int, int, float]]
    C: float

    @property
    def n(self) -> int:
        return len(self.ts)

    @property
    def t_max(self) -> float:
        """All-software schedule length (one CPU, no communication): AR's numerator."""
        return float(self.ts.sum())


def fig3_example() -> BinInstance:
    """The 11-task worked example of Fig. 3 / Table 1 (C = 18)."""
    tsa = [(4, 2, 4), (5, 3, 2), (4, 1, 4), (5, 2, 7), (3, 1, 5), (16, 10, 3), (3, 2, 4), (4, 2, 2), (14, 7, 3),
           (5, 2, 5), (20, 9, 5)]
    e = [(1, 4, 1), (2, 4, 2), (2, 5, 2), (3, 5, 1), (4, 6, 4), (4, 7, 1), (5, 8, 1), (5, 9, 2), (6, 10, 3),
         (7, 10, 1), (8, 11, 1), (9, 11, 2)]
    a = np.array(tsa, dtype=float)
    return BinInstance("gcps_fig3", a[:, 0], a[:, 1], a[:, 2], [(i - 1, j - 1, float(c)) for i, j, c in e], 18.0)


def _tgff_graphs(n: int, seed: int, count: int) -> list[tuple[list[str], list[tuple[str, str]]]]:
    if not TGFF_EXE.exists():
        raise FileNotFoundError(f"{TGFF_EXE} not found: run `python scripts/build_tgff.py --check` first")
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "g.tgffopt").write_text(
            f"seed {seed}\ntg_cnt {count}\ntask_cnt {n} 1\ntask_degree 3 3\nperiod_mul 1\ntg_write\n",
            encoding="utf-8")
        subprocess.run([str(TGFF_EXE), "g"], cwd=tmp, check=True, capture_output=True)
        f = read_tgff(Path(tmp, "g.tgff"))
    return [(list(g.tasks), [(s, d) for _, s, d, _ in g.arcs]) for g in f.graphs]


def _topo(names: list[str], arcs: list[tuple[str, str]]) -> list[str]:
    indeg = {x: 0 for x in names}
    succ: dict[str, list[str]] = {x: [] for x in names}
    for s, d in arcs:
        indeg[d] += 1
        succ[s].append(d)
    order, ready = [], [x for x in names if indeg[x] == 0]
    while ready:
        x = ready.pop(0)
        order.append(x)
        for d in succ[x]:
            indeg[d] -= 1
            if indeg[d] == 0:
                ready.append(d)
    if len(order) != len(names):
        raise ValueError("TGFF graph is not acyclic")
    return order


def generate(n: int, ct: str = "CT2", ap: float = 1 / 3, seed: int = 0, graphs: int = 10) -> list[BinInstance]:
    """`graphs` instances with exactly n tasks (TGFF topology, Table 2 cost ranges), deterministic in `seed`."""
    rng = np.random.default_rng([20211030, n, seed])
    lo_c, hi_c = CT[ct]
    out: list[BinInstance] = []
    tseed = seed * 1000 + 1
    while len(out) < graphs:
        for names, arcs in _tgff_graphs(n, tseed, max(3 * graphs, 10)):
            if len(names) != n or len(out) >= graphs:
                continue
            order = _topo(names, arcs)
            idx = {x: i for i, x in enumerate(order)}
            ts = rng.uniform(*SW_RANGE, n)
            th = rng.uniform(*HW_RANGE, n)
            area = rng.uniform(*AREA_RANGE, n)
            edges = sorted({(idx[s], idx[d]) for s, d in arcs})
            comm = rng.uniform(lo_c, hi_c, len(edges))
            out.append(BinInstance(f"gcps_tgff_n{n}_{ct}_s{seed}_{len(out)}", ts, th, area,
                                   [(i, j, float(c)) for (i, j), c in zip(edges, comm, strict=True)],
                                   float(ap * area.sum())))
        tseed += 1
    return out
