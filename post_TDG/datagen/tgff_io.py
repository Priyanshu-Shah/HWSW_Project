"""Parser for TGFF output files (`.tgff`, TGFF 3.x format; also reads E3S, which uses the same format).

A file holds `@HYPERPERIOD`, `@TASK_GRAPH i { PERIOD, TASK, ARC, HARD_/SOFT_DEADLINE }` blocks and attribute
tables `@LABEL i { ... }`. Inside a table, a comment line names the columns of the data lines that follow it: the
first data line holds table-level attributes (e.g. price), later lines are per-type rows (first column `type`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TgffGraph:
    index: int
    period: float | None = None
    tasks: dict[str, int] = field(default_factory=dict)  # task name -> type
    arcs: list[tuple[str, str, str, int]] = field(default_factory=list)  # (name, src, dst, type)
    deadlines: list[tuple[str, str, str, float]] = field(default_factory=list)  # (kind, name, task, at)


@dataclass
class TgffTable:
    label: str
    index: int
    attrs: dict[str, float] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    rows: list[list[float]] = field(default_factory=list)

    def column(self, name: str) -> dict[int, float]:
        """type -> value for one per-type column."""
        i = self.columns.index(name)
        return {int(r[0]): r[i] for r in self.rows if len(r) > i}


@dataclass
class TgffFile:
    hyperperiod: float | None
    graphs: list[TgffGraph]
    tables: dict[str, list[TgffTable]]


_BLOCK = re.compile(r"^@(\w+)\s+(\d+)\s*\{")
_NUM = re.compile(r"^[-+0-9.eE\s]+$")


def _num(x: str) -> float:
    return float(x)


def parse_tgff(text: str) -> TgffFile:
    hyper = None
    graphs: list[TgffGraph] = []
    tables: dict[str, list[TgffTable]] = {}
    lines = text.replace("\r", "").split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("@HYPERPERIOD"):
            hyper = _num(ln.split()[1])
            i += 1
            continue
        m = _BLOCK.match(ln)
        if not m:
            i += 1
            continue
        label, idx = m.group(1), int(m.group(2))
        body = []
        i += 1
        while i < len(lines) and lines[i].strip() != "}":
            body.append(lines[i].strip())
            i += 1
        i += 1
        if label == "TASK_GRAPH":
            g = TgffGraph(idx)
            for b in body:
                t = b.split()
                if not t:
                    continue
                u = [w.upper() for w in t]  # E3S writes some keywords in lower case ("to")

                def at(kw: str, t=t, u=u) -> str:
                    return t[u.index(kw) + 1]

                if u[0] == "PERIOD":
                    g.period = _num(t[1])
                elif u[0] == "TASK":
                    g.tasks[t[1]] = int(at("TYPE"))
                elif u[0] == "ARC":
                    g.arcs.append((t[1], at("FROM"), at("TO"), int(at("TYPE"))))
                elif u[0] in ("HARD_DEADLINE", "SOFT_DEADLINE"):
                    g.deadlines.append((u[0], t[1], at("ON"), _num(at("AT"))))
            graphs.append(g)
        else:
            tab = TgffTable(label, idx)
            cols: list[str] = []
            seen_header = False
            for b in body:
                if not b:
                    continue
                if b.startswith("#"):
                    words = b.lstrip("#").split()
                    # a column header is the "type ..." line, or (before the attribute line) the attribute names;
                    # other comments are descriptions (E3S names each kernel row in a comment)
                    if words and not set(b.lstrip("#").strip()) <= set("-=") and (
                            words[0] == "type" or not seen_header):
                        cols = words
                    continue
                if not _NUM.match(b):
                    continue
                vals = [_num(x) for x in b.split()]
                if cols and cols[0] == "type":
                    tab.columns = cols
                    tab.rows.append(vals)
                elif not seen_header:
                    tab.attrs.update(dict(zip(cols, vals, strict=False)))
                    seen_header = True
            tables.setdefault(label, []).append(tab)
    return TgffFile(hyper, graphs, tables)


def read_tgff(path: str | Path) -> TgffFile:
    return parse_tgff(Path(path).read_text(encoding="utf-8", errors="replace"))
