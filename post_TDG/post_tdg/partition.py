"""Costed TDG + board -> HW/SW partition (`partition.result.json`). Decision D8 in `decisions.md`.

- Up to `EXACT_MAX_TASKS` iteration nodes: the exact component (`solvers.exact`), warm-started with PEFT,
  MAGELLAN and the GNN's greedy plan. It returns a proven optimum, or the best plan and a valid lower bound.
- Larger graphs: the fast GA (`fastheur`) seeded with GNN samples (best mix on the Thor run 2), then PEFT as a
  fallback; the best plan is certified against the mapping-relaxation bound.
- Every plan is scored by the one simulator (`sim.evaluate`); Monte Carlo gives the latency range.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path

from estimate import Option, annotate
from estimate.annotate import Problem
from post_tdg.convert import FPGA_PE, SW_IMPL, Conversion, convert
from schemas.result import Result, SolverInfo
from sim import evaluate
from solvers import exact
from solvers.common import build_result
from solvers.heuristics import Plan, all_sw, plan

RESULT_VERSION = "0.1"
EXACT_MAX_TASKS = 40  # CP-SAT proves optima up to 40 tasks in about a second (handover benchmarks)
DEFAULT_TIME_LIMIT_S = 10.0
MONTE_CARLO_SAMPLES = 1000
DEFAULT_GNN = Path(__file__).resolve().parents[1] / "models" / "gnn_r2_all_base_s1.pt"


@dataclass
class Partition:
    result: dict  # partition.result.json (SI units)
    raw: Result  # the partitioner's own Result (ms), for debugging


def partition(
    costed_tdg: dict,
    board: dict,
    *,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    gnn_path: Path | None = DEFAULT_GNN,
    seed: int = 0,
    samples: int = MONTE_CARLO_SAMPLES,
) -> Partition:
    conversion = convert(costed_tdg, board)
    problem = annotate(conversion.graph, conversion.platform)
    gnn, gnn_note = _load_gnn(gnn_path)
    raw = _solve(problem, conversion, gnn, time_limit_s, seed, samples)
    best_effort = False
    if not raw.mapping and problem.deadline_ms is not None:
        # No partition meets the deadline: still give the fastest one, with the deadline feedback kept.
        fastest = _solve(_without_deadline(problem), conversion, gnn, time_limit_s, seed, samples)
        if fastest.mapping:
            fastest.feasible = False
            fastest.diagnostics = raw.diagnostics
            raw, best_effort = fastest, True
    if gnn_note:
        raw.solver.notes = "; ".join(filter(None, [raw.solver.notes, gnn_note]))
    result = to_result_json(costed_tdg, board, conversion, problem, raw)
    result["best_effort"] = best_effort
    return Partition(result=result, raw=raw)


def _solve(problem: Problem, conversion: Conversion, gnn, time_limit_s: float, seed: int, samples: int) -> Result:
    if len(problem.order) <= EXACT_MAX_TASKS:
        return _solve_exact(problem, gnn, time_limit_s, seed, samples)
    return _solve_large(problem, conversion, gnn, time_limit_s, seed, samples)


def _without_deadline(problem: Problem) -> Problem:
    return dataclasses.replace(problem, deadline_ms=None, _norms=None)


# ------------------------------------------------------------------------------------------- solvers
def _solve_exact(problem: Problem, gnn, time_limit_s: float, seed: int, samples: int) -> Result:
    warm = [plan(problem, "peft"), plan(problem, "magellan")]
    if gnn is not None:
        decoded = gnn.decode(problem, gnn.featurize(problem), samples=16, seed=seed)
        warm.append(Plan(decoded.mapping, decoded.priority))
    return exact.run(problem, time_limit_s=time_limit_s, warm_starts=warm, seed=seed, n_samples=samples)


def _solve_large(problem: Problem, conversion: Conversion, gnn, time_limit_s: float, seed: int, samples: int) -> Result:
    from fastheur import bridge
    from fastheur.algos import METHODS
    from fastheur.fast import Fast, warmup
    from fastheur.instance import from_graph_platform

    t0 = time.perf_counter()
    warmup()
    inst = from_graph_platform(
        conversion.graph.model_dump(mode="json", exclude_none=True),
        conversion.platform.model_dump(mode="json", exclude_none=True),
    )
    fast = Fast(inst)
    seeds = []
    if gnn is not None:
        sampler = bridge.GnnSampler(gnn, problem, gnn.featurize(problem), inst)
        seeds = [(c, k) for c, k, _, _ in sampler.population(fast, m=256, keep=8, seed=seed, time_cap_s=0.25)]
    with _extra_ga_seeds(seeds):
        found = METHODS["ga"](inst, fast, seed, max(0.5, 0.7 * time_limit_s))
    candidates = [plan(problem, "peft")]
    if found.plan is not None:
        candidates.append(
            Plan(
                bridge.from_opt(inst, problem, found.plan.opt),
                bridge.handover_priority(inst, found.plan.order),
                insertion=found.plan.insertion,
            )
        )
    best = min(candidates, key=lambda p: _score(problem, p))
    left = max(1.0, time_limit_s - (time.perf_counter() - t0))
    cert = exact.certify(problem, best, time_limit_s=left)
    status = {"optimal": "optimal", "infeasible": "infeasible"}.get(cert.status, "heuristic")
    info = SolverInfo(
        name="gnn+ga" if seeds else "ga",
        status=status,  # type: ignore[arg-type]
        runtime_s=time.perf_counter() - t0,
        seed=seed,
        objective=cert.makespan_ms,
        params={"time_limit_s": time_limit_s, "bound_ms": cert.bound_ms, "gap": cert.gap},
        notes="; ".join(cert.certificate) or None,
    )
    return build_result(
        problem,
        best.mapping,
        info,
        priority=best.priority,
        insertion=best.insertion,
        n_samples=samples,
        seed=seed,
        strong_bound=True,
    )


def _score(problem: Problem, candidate: Plan) -> tuple[bool, float]:
    ev = evaluate(problem, candidate.mapping, priority=candidate.priority, insertion=candidate.insertion)
    return (not ev.feasible, ev.sim.makespan)


@contextlib.contextmanager
def _extra_ga_seeds(extra: list):
    """The GA seeds itself with HEFT and all-SW; add the GNN samples (as the Thor run 2 did)."""
    from fastheur.algos import evo

    original = evo._seeds

    def seeds(inst, fast, tracker):
        out = original(inst, fast, tracker)
        for opt, keys in extra:
            makespan, excess = fast.evaluate(opt)
            tracker.offer(makespan, excess, opt)
            out.append((opt.copy(), keys.copy()))
        return out

    evo._seeds = seeds
    try:
        yield
    finally:
        evo._seeds = original


def _load_gnn(path: Path | None):
    if path is None:
        return None, None
    try:
        from fastheur.bridge import Gnn

        return Gnn.load(path), None
    except Exception as failure:  # noqa: BLE001 - the GNN only adds warm starts; solve without it
        return None, f"GNN not used ({type(failure).__name__}: {failure})"


# ------------------------------------------------------------------------------------------- output
def to_result_json(costed_tdg: dict, board: dict, conversion: Conversion, problem: Problem, raw: Result) -> dict:
    names = {node["id"]: node["name"] for node in costed_tdg["nodes"]}
    deadline_s = costed_tdg.get("constraints", {}).get("deadline_s")
    params = raw.solver.params
    bound_ms = params.get("bound_ms")
    lower_bound = raw.diagnostics.lower_bound
    floor_ms = None
    if lower_bound is not None:
        floor_ms = max(filter(None, [lower_bound.latency_ms, lower_bound.latency_ms_relaxation]))
    software_ms = _all_software_ms(problem)
    latency = raw.metrics.latency_ms if raw.metrics else None
    schedule = {task.node: task for task in raw.schedule}
    nodes = []
    for node_id, assignment in raw.mapping.items():
        task = schedule[node_id]
        on_fpga = assignment.pe == FPGA_PE
        nodes.append(
            {
                "id": node_id,
                "name": names[node_id],
                "target": "hw" if on_fpga else "sw",
                "pe": assignment.pe,
                "point": assignment.impl if on_fpga else None,
                "start_s": task.start_ms / 1e3,
                "finish_s": task.finish_ms / 1e3,
            }
        )
    nodes.sort(key=lambda item: (item["start_s"], item["id"]))
    transfers = [
        {
            "src": comm.src,
            "dst": comm.dst,
            "interface": comm.interface,
            "bytes": comm.bytes,
            "start_s": comm.start_ms / 1e3,
            "finish_s": comm.finish_ms / 1e3,
        }
        for comm in raw.comms
        if comm.interface is not None
    ]
    used = (raw.metrics.resources.get(FPGA_PE, {}) if raw.metrics else {}) or {}
    return {
        "schema_version": RESULT_VERSION,
        "graph_id": costed_tdg["graph"]["id"],
        "platform": board["id"],
        "solver": {
            "method": raw.solver.name,
            "status": raw.solver.status,
            "runtime_s": raw.solver.runtime_s,
            "gap": params.get("gap"),
            "lower_bound_s": bound_ms / 1e3 if bound_ms is not None else None,
            "notes": raw.solver.notes,
        },
        "feasible": raw.feasible,
        "deadline_s": deadline_s,
        "latency_s": _interval_s(latency) if latency else None,
        "all_software_s": software_ms / 1e3 if software_ms is not None else None,
        "speedup": software_ms / latency.nominal if software_ms and latency and latency.nominal else None,
        "nodes": nodes,
        "setup_nodes": conversion.setup_nodes,
        "transfers": transfers,
        "fpga_resources": {
            "used": {key: used.get(key, 0.0) for key in board["fpga"]["capacity"]},
            "capacity": dict(board["fpga"]["capacity"]),
        },
        "feedback": {
            "loop": raw.diagnostics.loop,
            "reasons": list(raw.diagnostics.reasons),
            "min_latency_s": floor_ms / 1e3 if floor_ms is not None else None,
        },
    }


def _interval_s(interval) -> dict:
    return {key: getattr(interval, key) / 1e3 for key in ("nominal", "lo", "hi", "p05", "p50", "p95")} | {
        "samples": interval.n_samples
    }


def _all_software_ms(problem: Problem) -> float | None:
    """All nodes on CPU cores (list-scheduled across the cores); None if a node cannot run in software."""
    if any(all(problem.pe_kind(o.pe) != "cpu" for o in problem.options[t]) for t in problem.order):
        return None
    mapping: dict[str, Option] = all_sw(problem)
    return evaluate(problem, mapping).sim.makespan


__all__ = ["DEFAULT_GNN", "EXACT_MAX_TASKS", "SW_IMPL", "Partition", "partition"]
