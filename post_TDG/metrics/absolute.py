"""Absolute metrics: nominal value plus Monte Carlo interval for one mapping."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from estimate.annotate import Problem
from schemas.common import Interval
from schemas.result import Metrics, ThroughputMetrics
from sim import montecarlo
from sim.evaluate import _exceeds, avg_power_mw, evaluate
from sim.scheduler import MappingT, simulate
from sim.throughput import orders_from, throughput

DEFAULT_SAMPLES = 1000


def interval(nominal: float, samples: np.ndarray) -> Interval:
    if samples.size == 0:
        return Interval(nominal=nominal, lo=nominal, hi=nominal, p05=nominal, p50=nominal, p95=nominal, n_samples=0)
    method = "linear" if np.isfinite(samples).all() else "nearest"  # avoid inf - inf when some samples are inf
    p05, p50, p95 = np.percentile(samples, [5, 50, 95], method=method).tolist()
    return Interval(nominal=nominal, lo=float(samples.min()), hi=float(samples.max()), p05=p05, p50=p50, p95=p95,
                    n_samples=int(samples.size))


def throughput_metrics(
    problem: Problem,
    mapping: MappingT,
    factors: list[dict] | None,
    priority: Sequence[str] | None = None,
    insertion: bool = False,
) -> tuple[ThroughputMetrics, np.ndarray]:
    """Exact period and steady latency: nominal, Monte Carlo with the nominal order frozen, and corner bounds.

    Returns the metrics and the per-sample periods (for feasibility accounting)."""
    nom_sim = simulate(problem, mapping, priority=priority, insertion=insertion)
    orders = orders_from(problem, nom_sim)
    nom = throughput(problem, mapping, orders=orders, priority=priority, insertion=insertion, buffers=problem.buffers)
    lo = throughput(problem, mapping, costs=problem.bound_view("lo"), orders=orders, priority=priority,
                    insertion=insertion, buffers=problem.buffers).period_ms
    hi = throughput(problem, mapping, costs=problem.bound_view("hi"), orders=orders, priority=priority,
                    insertion=insertion, buffers=problem.buffers).period_ms
    periods, lats = [], []
    for f in factors or []:
        r = throughput(problem, mapping, costs=problem.view(f), orders=orders, priority=priority,
                       insertion=insertion, buffers=problem.buffers)
        periods.append(r.period_ms)
        if r.steady_latency_ms is not None:
            lats.append(r.steady_latency_ms)
    p = np.array(periods, dtype=float)
    lat = None
    if nom.steady_latency_ms is not None:
        lat = interval(nom.steady_latency_ms, np.array(lats, dtype=float))
    tm = ThroughputMetrics(
        period_ms=interval(nom.period_ms, p),
        period_bounds_ms=(lo, hi),
        load_bound_ms=nom.load_bound_ms,
        steady_latency_ms=lat,
        input_period_ms=nom.input_period_ms,
        buffers=problem.buffers,
        critical_cycle=nom.critical_cycle,
    )
    return tm, p


def metrics_for(
    problem: Problem,
    mapping: MappingT,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
    priority: Sequence[str] | None = None,
    insertion: bool = False,
) -> Metrics:
    nom = evaluate(problem, mapping, priority=priority, insertion=insertion).sim
    factors = montecarlo.draw(problem, n_samples, seed)
    mc = montecarlo.run(problem, mapping, priority=priority, insertion=insertion, factors=factors)
    feasible = mc.feasible
    tm = None
    if problem.arrival == "streaming":
        tm, periods = throughput_metrics(problem, mapping, factors, priority, insertion)
        if problem.period_ms is not None and periods.size:
            feasible = feasible & ~np.array([_exceeds(x, problem.period_ms) for x in periods])
    if n_samples:
        ff = float(feasible.mean())
    else:
        ev = evaluate(problem, mapping, priority=priority, insertion=insertion)
        ff = float(ev.feasible and not (tm and problem.period_ms is not None
                                        and _exceeds(tm.period_ms.nominal, problem.period_ms)))
    return Metrics(
        latency_ms=interval(nom.makespan, mc.latency_ms),
        energy_mj=interval(nom.energy_mj, mc.energy_mj),
        ii_ms=interval(nom.ii_ms, mc.ii_ms),
        avg_power_mw=interval(avg_power_mw(problem, nom), mc.avg_power_mw),
        usd=nom.usd,
        resources=nom.usage,
        feasible_fraction=ff,
        throughput=tm,
        seed=seed,
    )
