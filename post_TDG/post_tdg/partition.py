"""Costed TDG + board -> HW/SW partition (`partition.result.json`). Decision D8 in `decisions.md`.

The solver is the budget-driven portfolio in `portfolio.py` (from the integrated package v2): compiled PEFT
first, then the GA seeded with GNN samples (or SA for tiny budgets), then CP-SAT warm-started with every candidate
on graphs up to 40 nodes (budget >= 2 s) and 41-100 nodes (budget >= 10 s). Every plan is scored by the one
simulator (`sim.evaluate`); Monte Carlo gives the latency range. Plans the exact phase did not see are certified
against the mapping-relaxation bound.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from estimate import Option, annotate
from estimate.annotate import Problem
from post_tdg import portfolio
from post_tdg.convert import FPGA_PE, SW_IMPL, Conversion, convert
from schemas.result import Result, SolverInfo
from sim import evaluate
from solvers import exact
from solvers.common import build_result
from solvers.heuristics import all_sw

RESULT_VERSION = "0.1"
EXACT_MAX_TASKS = portfolio.RULES["exact_max_tasks_small"]
CERTIFY_SHARE = 0.3  # of the budget, at least 1 s: relaxation bound for plans the exact phase did not see
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
    graph = conversion.graph.model_dump(mode="json", exclude_none=True)
    platform = conversion.platform.model_dump(mode="json", exclude_none=True)
    gnn, gnn_note = _load_gnn(gnn_path)
    outcome = portfolio.solve(problem, graph, platform, gnn, time_limit_s, seed)
    best_effort = False
    if problem.deadline_ms is not None and (outcome.best is None or not outcome.best.feasible):
        # No partition meets the deadline: search again without it for the fastest one. The result is still
        # built against the deadline, so it stays infeasible and keeps the fb1 feedback.
        free = portfolio.solve(_without_deadline(problem), {**graph, "constraints": {}}, platform, gnn,
                               time_limit_s, seed)
        if free.best is not None:
            outcome, best_effort = free, True
    raw = _to_raw(problem, outcome, time_limit_s, seed, samples)
    if gnn_note:
        raw.solver.notes = "; ".join(filter(None, [raw.solver.notes, gnn_note]))
    result = to_result_json(costed_tdg, board, conversion, problem, raw)
    result["best_effort"] = best_effort
    return Partition(result=result, raw=raw)


def _without_deadline(problem: Problem) -> Problem:
    return dataclasses.replace(problem, deadline_ms=None, _norms=None)


def _to_raw(problem: Problem, outcome: portfolio.Outcome, time_limit_s: float, seed: int, samples: int) -> Result:
    """The best candidate as the partitioner's `Result`, with a lower bound and gap (CP-SAT's, else certified)."""
    best = outcome.best
    if best is None:
        info = SolverInfo(name="none", status="infeasible", runtime_s=outcome.runtime_s, seed=seed,
                          params={"time_limit_s": time_limit_s, "steps": outcome.steps})
        return build_result(problem, None, info, n_samples=samples, seed=seed, strong_bound=True)
    bound_ms, gap, notes, status = outcome.bound_ms, outcome.gap, None, best.status
    if not outcome.exact_ran or bound_ms is None:
        cert = exact.certify(problem, best.plan, time_limit_s=max(1.0, CERTIFY_SHARE * time_limit_s))
        bound_ms, gap, notes = cert.bound_ms, cert.gap, "; ".join(cert.certificate) or None
        if cert.status == "optimal":
            status = "optimal"
    elif best.name != "exact" and best.makespan_ms > 0:
        gap = max(0.0, (best.makespan_ms - bound_ms) / best.makespan_ms)  # CP-SAT's gap is for its own plan
    if status == "optimal" and best.name != "exact" and outcome.exact_ran:
        notes = "CP-SAT proved this plan optimal"
    info = SolverInfo(
        name=best.name,
        status={"optimal": "optimal", "feasible": "feasible"}.get(status, "heuristic"),  # type: ignore[arg-type]
        runtime_s=outcome.runtime_s,
        seed=seed,
        objective=best.makespan_ms,
        params={"time_limit_s": time_limit_s, "bound_ms": bound_ms, "gap": gap, "steps": outcome.steps,
                "candidates": {c.name: c.makespan_ms for c in outcome.candidates}},
        notes=notes,
    )
    plan = best.plan
    return build_result(problem, plan.mapping, info, priority=plan.priority, insertion=plan.insertion,
                        n_samples=samples, seed=seed, strong_bound=True)


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
