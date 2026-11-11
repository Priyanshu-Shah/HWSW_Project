"""Compact problem instance built from the group's JSON formats (graph + platform) or a GCPS benchmark file.

Everything the solvers touch is an integer index into flat arrays:

* tasks are re-indexed in topological order, so every predecessor index is smaller than its successor's;
* every legal (PE, implementation) pair is a *global option id* ``k``; ``task_opts[i]`` lists the ids of task ``i``;
* ``ctime[e, a, b]`` is the transfer time of edge ``e`` from PE ``a`` to PE ``b`` over the fastest interface for
  that size (0 on the same PE, ``inf`` when no interface connects them); ``cif[e, a, b]`` is the interface index.

Semantics follow ``format/model.md`` of the data package (nominal costs only; uncertainty intervals are ignored,
as in the exact labels).
"""

from __future__ import annotations

import ast
import json
import math
import operator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

INF = float("inf")

# ---------------------------------------------------------------------------------------------------------
# safe expression evaluation (same grammar as the group's estimate/expr.py)
# ---------------------------------------------------------------------------------------------------------
_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_FUN = {"min": min, "max": max, "ceil": math.ceil, "floor": math.floor, "log2": math.log2, "sqrt": math.sqrt}


def _ev(node: ast.AST, env: dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _ev(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        return float(env[node.id])
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_ev(node.left, env), _ev(node.right, env))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _ev(node.operand, env)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUN and not node.keywords:
        return float(_FUN[node.func.id](*[_ev(a, env) for a in node.args]))
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


def param(p, env: dict[str, float]) -> float | None:
    """Nominal value of a cost parameter: number, expression string, or {expr, unc}."""
    if p is None:
        return None
    if isinstance(p, dict):
        p = p["expr"]
    if isinstance(p, (int, float)):
        return float(p)
    return _ev(ast.parse(str(p), mode="eval"), env)


# ---------------------------------------------------------------------------------------------------------
@dataclass
class Instance:
    name: str
    n: int
    tnames: list[str]
    preds: list[list[tuple[int, int]]]          # (pred task, edge id)
    succs: list[list[tuple[int, int]]]          # (succ task, edge id)
    e_src: np.ndarray
    e_dst: np.ndarray
    e_bytes: np.ndarray
    # platform
    m: int
    pe_ids: list[str]
    pe_kind: list[str]
    pe_excl: np.ndarray                         # bool[m]
    res_names: list[str]
    pe_cap: np.ndarray                          # float[m, R]  (inf = unconstrained)
    iface_ids: list[str]
    iface_shared: np.ndarray                    # bool[n_iface]
    # options
    K: int
    o_task: np.ndarray
    o_pe: np.ndarray
    o_time: np.ndarray
    o_energy: np.ndarray
    o_res: np.ndarray                           # float[K, R]
    o_name: list[str]                           # "<pe>/<impl>"
    task_opts: list[list[int]]
    # communication
    ctime: np.ndarray                           # float[E, m, m]
    cenergy: np.ndarray                         # float[E, m, m]
    cif: np.ndarray                             # int[E, m, m]
    # metadata
    deadline: float | None = None
    lower_bound: float | None = None
    static_power: float = 0.0
    pinned: dict[int, int] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------------------------------------
    @property
    def E(self) -> int:
        return len(self.e_src)

    @property
    def R(self) -> int:
        return len(self.res_names)

    def opt_index(self, task: int, pe_id: str, impl: str) -> int:
        want = f"{pe_id}/{impl}"
        for k in self.task_opts[task]:
            if self.o_name[k] == want:
                return k
        raise KeyError(f"task {task}: no option {want}")

    def pe_class(self, pe: int) -> str:
        """'sw' for sequential CPU-like PEs, 'hw' for everything else (FPGA fabric, accelerators)."""
        return "sw" if self.pe_kind[pe] == "cpu" else "hw"

    def mapping_from_names(self, mapping: dict[str, list[str]]) -> np.ndarray:
        idx = {t: i for i, t in enumerate(self.tnames)}
        out = np.empty(self.n, dtype=np.int64)
        for t, (pe, impl) in mapping.items():
            out[idx[t]] = self.opt_index(idx[t], pe, impl)
        return out

    def mapping_to_names(self, opt: np.ndarray) -> dict[str, list[str]]:
        res = {}
        for i in range(self.n):
            pe, impl = self.o_name[int(opt[i])].split("/", 1)
            res[self.tnames[i]] = [pe, impl]
        return res

    def cheapest_comm_matrix(self) -> np.ndarray:
        return self.ctime


def _finish_comm(m, n_if, iface_conn, iface_setup_ms, iface_ms_per_byte, iface_e_per_byte, e_bytes):
    E = len(e_bytes)
    ctime = np.zeros((E, m, m))
    cenergy = np.zeros((E, m, m))
    cif = np.full((E, m, m), -1, dtype=np.int64)
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            cands = [i for i in range(n_if) if a in iface_conn[i] and b in iface_conn[i]]
            if not cands:
                ctime[:, a, b] = INF
                continue
            t = np.stack([iface_setup_ms[i] + e_bytes * iface_ms_per_byte[i] for i in cands])  # (c, E)
            best = np.argmin(t, axis=0)        # first minimum -> deterministic tie-break by interface order
            ctime[:, a, b] = t[best, np.arange(E)]
            cif[:, a, b] = np.array(cands)[best]
            cenergy[:, a, b] = np.array([iface_e_per_byte[cands[j]] for j in best]) * e_bytes * 1e-6
    return ctime, cenergy, cif


def from_record(rec: dict) -> Instance:
    """Build an instance from a label record (keys ``graph`` and ``platform``) or any dict with those keys."""
    return from_graph_platform(rec["graph"], rec["platform"], name=rec.get("id"), deadline=rec.get("deadline_ms"),
                               lower_bound=rec.get("lower_bound_ms"))


def from_graph_platform(graph: dict, platform: dict, name: str | None = None, deadline: float | None = None,
                        lower_bound: float | None = None, workload: dict | None = None) -> Instance:
    env = {**graph.get("workload", {}), **(workload or {})}
    allowed = graph.get("platform", {}).get("allowed_pes") or [p["id"] for p in platform["pes"]]
    pes = [p for p in platform["pes"] if p["id"] in allowed]
    pe_ids = [p["id"] for p in pes]
    m = len(pes)
    pe_index = {p: i for i, p in enumerate(pe_ids)}
    res_names = sorted({r for p in pes for r in p.get("capacity", {})}
                       | {r for nd in graph["nodes"] for im in nd["implementations"]
                          for r in im["cost"].get("resources", {})})
    pe_cap = np.full((m, len(res_names)), INF)
    for i, p in enumerate(pes):
        for r, c in p.get("capacity", {}).items():
            pe_cap[i, res_names.index(r)] = c
    ifs = platform.get("interfaces", [])
    iface_conn = [{pe_index[c] for c in itf["connects"] if c in pe_index} for itf in ifs]
    ctx = dict(
        m=m, n_if=len(ifs), iface_conn=iface_conn,
        iface_setup_ms=[itf["setup_us"] / 1000.0 for itf in ifs],
        iface_ms_per_byte=[1.0 / (itf["bandwidth_mb_s"] * 1e3) for itf in ifs],
        iface_e_per_byte=[itf.get("energy_nj_per_byte", 0.0) for itf in ifs],
    )

    # topological re-indexing (stable w.r.t. the file order)
    ids = [nd["id"] for nd in graph["nodes"]]
    pos = {t: i for i, t in enumerate(ids)}
    indeg = [0] * len(ids)
    adj: list[list[int]] = [[] for _ in ids]
    for ed in graph["edges"]:
        if ed.get("distance", 0):
            continue                                   # cross-frame edges do not constrain a single iteration
        adj[pos[ed["src"]]].append(pos[ed["dst"]])
        indeg[pos[ed["dst"]]] += 1
    import heapq
    heap = [i for i in range(len(ids)) if indeg[i] == 0]
    heapq.heapify(heap)
    topo: list[int] = []
    while heap:
        u = heapq.heappop(heap)
        topo.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, v)
    if len(topo) != len(ids):
        raise ValueError("task graph has a cycle")
    new_idx = {ids[old]: new for new, old in enumerate(topo)}
    n = len(ids)
    tnames = [ids[old] for old in topo]
    nodes = {nd["id"]: nd for nd in graph["nodes"]}

    e_src, e_dst, e_bytes = [], [], []
    preds = [[] for _ in range(n)]
    succs = [[] for _ in range(n)]
    for ed in graph["edges"]:
        if ed.get("distance", 0):
            continue
        s, d = new_idx[ed["src"]], new_idx[ed["dst"]]
        eid = len(e_src)
        e_src.append(s)
        e_dst.append(d)
        e_bytes.append(param(ed.get("bytes"), env) or 0.0)
        preds[d].append((s, eid))
        succs[s].append((d, eid))
    e_bytes_a = np.array(e_bytes, dtype=float)

    o_task, o_pe, o_time, o_energy, o_res, o_name = [], [], [], [], [], []
    task_opts: list[list[int]] = [[] for _ in range(n)]
    pinned: dict[int, int] = {}
    for i, t in enumerate(tnames):
        nd = nodes[t]
        pin = nd.get("pinned_pe")
        if pin:
            pinned[i] = pe_index[pin]
        for im in nd["implementations"]:
            c = im["cost"]
            tm = param(c["time_ms"], env)
            en = param(c.get("energy_mj"), env)
            pw = param(c.get("power_mw"), env)
            if en is None:
                en = (pw * tm / 1000.0) if pw is not None else 0.0
            rvec = np.zeros(len(res_names))
            for r, v in c.get("resources", {}).items():
                rvec[res_names.index(r)] = param(v, env)
            for pi, pe in enumerate(pes):
                if pe["type"] != im["pe_type"] or (pin and pe_ids[pi] != pin):
                    continue
                task_opts[i].append(len(o_task))
                o_task.append(i); o_pe.append(pi); o_time.append(tm); o_energy.append(en)
                o_res.append(rvec); o_name.append(f"{pe_ids[pi]}/{im['id']}")
        if not task_opts[i]:
            raise ValueError(f"task {t}: no legal option")
    ctime, cenergy, cif = _finish_comm(e_bytes=e_bytes_a, **ctx)
    return Instance(
        name=name or graph.get("id", "instance"), n=n, tnames=tnames, preds=preds, succs=succs,
        e_src=np.array(e_src, dtype=np.int64), e_dst=np.array(e_dst, dtype=np.int64), e_bytes=e_bytes_a,
        m=m, pe_ids=pe_ids, pe_kind=[p["kind"] for p in pes],
        pe_excl=np.array([p["concurrency"] == "exclusive" for p in pes]),
        res_names=res_names, pe_cap=pe_cap,
        iface_ids=[i["id"] for i in ifs], iface_shared=np.array([bool(i.get("shared", True)) for i in ifs], dtype=bool),
        K=len(o_task), o_task=np.array(o_task, dtype=np.int64), o_pe=np.array(o_pe, dtype=np.int64),
        o_time=np.array(o_time), o_energy=np.array(o_energy),
        o_res=np.array(o_res).reshape(len(o_task), len(res_names)), o_name=o_name, task_opts=task_opts,
        ctime=ctime, cenergy=cenergy, cif=cif,
        deadline=deadline if deadline is not None else param(graph.get("constraints", {}).get("deadline_ms"), env),
        lower_bound=lower_bound,
        static_power=sum(p.get("static_power_mw", 0.0) for p in pes),
        pinned=pinned, meta={"provenance": graph.get("provenance", {}).get("notes")},
    )


def from_gcps(path: str | Path) -> Instance:
    """GCPS benchmark file -> instance in the group's model.

    One CPU (exclusive), one FPGA (spatial, resource ``area`` with capacity ``area_budget_C``), one shared bus.
    Times are converted from ns to ms. Edge costs are fixed per edge (``comm_ns``), so they are stored directly in
    ``ctime`` (no bytes/bandwidth round trip). All tasks have exactly one SW and one HW implementation.
    """
    d = json.loads(Path(path).read_text())
    n = d["n"]
    sw = np.array(d["sw_time"]) * 1e-6
    hw = np.array(d["hw_time"]) * 1e-6
    area = np.array(d["area"])
    edges = d["edges"]
    # re-index topologically (TGFF graphs are usually already ordered; do it properly anyway)
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for i, j, _ in edges:
        adj[int(i)].append(int(j)); indeg[int(j)] += 1
    import heapq
    heap = [i for i in range(n) if indeg[i] == 0]; heapq.heapify(heap); topo = []
    while heap:
        u = heapq.heappop(heap); topo.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, v)
    assert len(topo) == n, "cycle in GCPS file"
    new = {old: i for i, old in enumerate(topo)}
    preds = [[] for _ in range(n)]; succs = [[] for _ in range(n)]
    e_src, e_dst, ct = [], [], []
    for i, j, c in edges:
        s, t = new[int(i)], new[int(j)]
        eid = len(e_src); e_src.append(s); e_dst.append(t); ct.append(float(c) * 1e-6)
        preds[t].append((s, eid)); succs[s].append((t, eid))
    E = len(e_src)
    ctime = np.zeros((E, 2, 2)); ctime[:, 0, 1] = ctime[:, 1, 0] = np.array(ct).reshape(E)
    cif = np.full((E, 2, 2), -1, dtype=np.int64); cif[:, 0, 1] = cif[:, 1, 0] = 0
    o_task, o_pe, o_time, o_res, o_name, task_opts = [], [], [], [], [], [[] for _ in range(n)]
    for i, old in enumerate(topo):
        for pe, tm, a, nm in ((0, sw[old], 0.0, "cpu0/sw"), (1, hw[old], area[old], "fpga0/hw")):
            task_opts[i].append(len(o_task)); o_task.append(i); o_pe.append(pe); o_time.append(tm)
            o_res.append([a]); o_name.append(nm)
    return Instance(
        name=d["name"], n=n, tnames=[f"t{old}" for old in topo], preds=preds, succs=succs,
        e_src=np.array(e_src, dtype=np.int64), e_dst=np.array(e_dst, dtype=np.int64), e_bytes=np.zeros(E),
        m=2, pe_ids=["cpu0", "fpga0"], pe_kind=["cpu", "fpga"], pe_excl=np.array([True, False]),
        res_names=["area"], pe_cap=np.array([[INF], [float(d["area_budget_C"])]]),
        iface_ids=["bus"], iface_shared=np.array([True]),
        K=len(o_task), o_task=np.array(o_task, dtype=np.int64), o_pe=np.array(o_pe, dtype=np.int64),
        o_time=np.array(o_time), o_energy=np.zeros(len(o_task)), o_res=np.array(o_res).reshape(len(o_task), 1),
        o_name=o_name, task_opts=task_opts, ctime=ctime, cenergy=np.zeros((E, 2, 2)), cif=cif,
        meta={"gcps_all_sw_ns": d.get("all_sw_time"), "gcps_ar_greedy": d.get("greedy_hgp")},
    )


def from_golden(path: str | Path) -> Instance:
    """Hand-written case study (graph JSON that references a platform file by relative path)."""
    path = Path(path)
    g = json.loads(path.read_text())
    plat_path = (path.parent / g["platform"]["path"]).resolve()
    platform = json.loads(plat_path.read_text())
    return from_graph_platform(g, platform, name=g["id"])
