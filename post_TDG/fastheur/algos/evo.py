"""Population methods: genetic algorithm (Wang et al. 1997 style) and NSGA-II (Deb et al. 2002).

Chromosome: one global option id per task (``opt``) plus a random-key priority per task (``key``, smaller = earlier).
Decoding = the compiled list scheduler with explicit priorities (default mode, no insertion).
Constraint handling (Deb's constraint-domination): feasible beats infeasible; infeasible solutions compare by capacity
excess. Capacity violations of offspring are repaired and written back into the genome.
"""

from __future__ import annotations

import math

import numpy as np

from ..fast import Fast
from ..instance import Instance
from .base import Result, Tracker, all_sw_opt, hw_area, random_feasible, repair_capacity, usage_of


def _keys_from_order(inst: Instance, order) -> np.ndarray:
    key = np.empty(inst.n)
    for pos, i in enumerate(order):
        key[i] = pos / max(inst.n, 1)
    return key


def _seeds(inst: Instance, fast: Fast, tr: Tracker):
    """Seed genomes: HEFT (mapping + its order), all-SW, and the HEFT mapping with rank priorities."""
    plan, hmk = fast.heft()
    tr.offer_plan(plan, hmk)
    seeds = [(plan.opt.copy(), _keys_from_order(inst, plan.order)),
             (fast.allsw_opt.copy(), np.arange(inst.n) / max(inst.n, 1))]
    return seeds


class _Pop:
    def __init__(self, inst: Instance, fast: Fast, tr: Tracker, rng, mu: int, track=True):
        self.inst, self.fast, self.tr, self.rng, self.mu = inst, fast, tr, rng, mu
        self.n = inst.n
        self.track = track
        self.pm = 1.0 / max(inst.n, 1)

    # -- genetic operators -------------------------------------------------------------------------------
    def mutate(self, opt: np.ndarray, key: np.ndarray):
        inst, rng = self.inst, self.rng
        mask = rng.random(self.n) < self.pm
        if not mask.any():
            mask[int(rng.integers(self.n))] = True
        for i in np.where(mask)[0]:
            opts = inst.task_opts[i]
            if len(opts) > 1:
                opt[i] = opts[int(rng.integers(len(opts)))]
        km = rng.random(self.n) < self.pm
        key[km] = rng.random(int(km.sum()))

    def crossover(self, a, b):
        rng = self.rng
        (oa, ka), (ob, kb) = a, b
        m = rng.random(self.n) < 0.5
        return np.where(m, oa, ob), np.where(m, ka, kb)

    def repair(self, opt: np.ndarray) -> np.ndarray:
        u = usage_of(self.inst, opt)
        if np.all(u <= self.inst.pe_cap * (1 + 1e-9)):
            return opt
        return repair_capacity(self.inst, opt, self.rng)

    def init(self, seeds, size: int):
        inst, rng = self.inst, self.rng
        opts = [s[0].copy() for s in seeds]
        keys = [s[1].copy() for s in seeds]
        j = 0
        while len(opts) < size:
            if j % 2 == 0 or inst.n > 200:          # perturbed seed
                o, k = seeds[j % len(seeds)][0].copy(), seeds[j % len(seeds)][1].copy()
                self.pm, old = min(0.2, 5.0 / max(inst.n, 1) * 3), self.pm
                self.mutate(o, k)
                self.pm = old
                o = self.repair(o)
            else:
                o, k = random_feasible(inst, rng), rng.random(inst.n)
            opts.append(o); keys.append(k)
            j += 1
        return np.array(opts), np.array(keys)

    def evaluate(self, optm, keym):
        mk, ex = self.fast.eval_pop(optm, keym)
        if self.track:
            for q in np.argsort(mk):
                if ex[q] == 0.0:
                    if mk[q] < self.tr.best_mk * (1 - 1e-12):
                        order = np.argsort(keym[q], kind="stable")
                        self.tr.offer(mk[q], 0.0, optm[q], order)
                    break
        return mk, ex


def solve_ga(inst: Instance, fast: Fast, seed: int, budget_s: float, mu: int = 40, pc: float = 0.9) -> Result:
    """(mu + lambda) GA: binary tournaments by constraint-domination, uniform crossover on options and on priorities
    (prob. ``pc``), mutation 1/n per gene, capacity repair written back, elitist survival. Seeded with HEFT and all-SW."""
    tr = Tracker(inst, fast, budget_s)
    rng = np.random.default_rng(seed)
    pop = _Pop(inst, fast, tr, rng, mu)
    seeds = _seeds(inst, fast, tr)
    optm, keym = pop.init(seeds, mu)
    mk, ex = pop.evaluate(optm, keym)
    gens = 0
    while not tr.expired():
        gens += 1
        # binary tournaments (constraint-domination: excess first, then makespan)
        score = np.lexsort((mk, ex))                       # ascending by (ex, mk)
        rankpos = np.empty(len(score), dtype=int)
        rankpos[score] = np.arange(len(score))
        a = rng.integers(len(mk), size=(mu, 2))
        win = np.where(rankpos[a[:, 0]] < rankpos[a[:, 1]], a[:, 0], a[:, 1])
        co, ck = [], []
        for q in range(0, mu, 2):
            p1, p2 = win[q], win[(q + 1) % mu]
            g1, g2 = (optm[p1], keym[p1]), (optm[p2], keym[p2])
            if rng.random() < pc:
                c1 = pop.crossover(g1, g2)
                c2 = pop.crossover(g2, g1)
            else:
                c1, c2 = (g1[0].copy(), g1[1].copy()), (g2[0].copy(), g2[1].copy())
            for o, k in (c1, c2):
                o, k = o.copy(), k.copy()
                pop.mutate(o, k)
                co.append(pop.repair(o)); ck.append(k)
        co, ck = np.array(co[:mu]), np.array(ck[:mu])
        cmk, cex = pop.evaluate(co, ck)
        allo = np.vstack([optm, co]); allk = np.vstack([keym, ck])
        allm = np.concatenate([mk, cmk]); alle = np.concatenate([ex, cex])
        idx = np.lexsort((allm, alle))[:mu]
        optm, keym, mk, ex = allo[idx], allk[idx], allm[idx], alle[idx]
    return tr.result("ga", generations=gens)


# --------------------------------------------------------------------------------------------- NSGA-II
def second_objective(inst: Instance, name: str, opt: np.ndarray, mk: float) -> float:
    if name == "area":
        return hw_area(inst, usage_of(inst, opt))
    if name == "energy":
        pe = inst.o_pe[opt]
        e = float(inst.o_energy[opt].sum())
        if inst.E:
            e += float(inst.cenergy[np.arange(inst.E), pe[inst.e_src], pe[inst.e_dst]].sum())
        return e + inst.static_power * mk / 1000.0
    raise ValueError(name)


def _fronts(objs: np.ndarray, ex: np.ndarray):
    """Non-dominated sorting with constraint-domination (vectorised). Returns a list of index arrays, best front first."""
    feas = ex == 0
    le = np.all(objs[:, None, :] <= objs[None, :, :], axis=2)
    lt = np.any(objs[:, None, :] < objs[None, :, :], axis=2)
    dom = (feas[:, None] & feas[None, :] & le & lt)
    dom |= feas[:, None] & ~feas[None, :]
    dom |= (~feas[:, None]) & (~feas[None, :]) & (ex[:, None] < ex[None, :])
    cnt = dom.sum(axis=0).astype(int)
    fronts = []
    cur = np.where(cnt == 0)[0]
    while len(cur):
        fronts.append(cur)
        cnt = cnt - dom[cur].sum(axis=0)
        cnt[cur] = -1
        cur = np.where(cnt == 0)[0]
    return fronts


def _crowding(objs: np.ndarray) -> np.ndarray:
    N, M = objs.shape
    d = np.zeros(N)
    for m in range(M):
        o = np.argsort(objs[:, m], kind="stable")
        d[o[0]] = d[o[-1]] = np.inf
        span = objs[o[-1], m] - objs[o[0], m]
        if span > 0 and N > 2:
            d[o[1:-1]] += (objs[o[2:], m] - objs[o[:-2], m]) / span
    return d


def pareto_2d(pts: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated rows of a 2-column minimisation set (O(N log N))."""
    o = np.lexsort((pts[:, 1], pts[:, 0]))
    keep, best = [], np.inf
    for i in o:
        if pts[i, 1] < best:
            keep.append(i)
            best = pts[i, 1]
    return np.array(keep, dtype=int)


def hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    """Dominated hypervolume of a 2-objective minimisation set w.r.t. ``ref``."""
    pts = points[np.all(points < ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv, best_y = 0.0, ref[1]
    for x, y in pts:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return float(hv)


def solve_nsga2(inst: Instance, fast: Fast, seed: int, budget_s: float, mu: int = 40, pc: float = 0.9,
                second: str = "area") -> Result:
    """NSGA-II on (makespan, ``second``) with constraint-domination, crowding distance, (mu + mu) elitist survival.
    ``second`` is "area" (default; Tier A has no energy data) or "energy". The single-objective trace records the best
    feasible *makespan* on the way; the final non-dominated feasible set is returned in ``extra['front']`` as rows
    (makespan, second) with the matching option vectors in ``extra['front_opt']``."""
    tr = Tracker(inst, fast, budget_s)
    rng = np.random.default_rng(seed)
    pop = _Pop(inst, fast, tr, rng, mu)
    seeds = _seeds(inst, fast, tr)
    optm, keym = pop.init(seeds, mu)

    def objectives(optm_, mk_):
        return np.array([[m, second_objective(inst, second, o, m)] for o, m in zip(optm_, mk_)])

    mk, ex = pop.evaluate(optm, keym)
    objs = objectives(optm, mk)
    archive: dict[tuple, tuple] = {}

    def update_archive(optm_, objs_, ex_):
        nonlocal archive
        for o, ob, e in zip(optm_, objs_, ex_):
            if e == 0.0 and np.isfinite(ob).all():
                archive[tuple(o.tolist())] = (float(ob[0]), float(ob[1]))

    update_archive(optm, objs, ex)
    gens = 0
    while not tr.expired():
        gens += 1
        fr = _fronts(objs, ex)
        rk = np.empty(len(objs), dtype=int)
        cd = np.zeros(len(objs))
        for r, f in enumerate(fr):
            rk[f] = r
            cd[f] = _crowding(objs[f])
        a = rng.integers(len(objs), size=(mu, 2))
        better = (rk[a[:, 0]] < rk[a[:, 1]]) | ((rk[a[:, 0]] == rk[a[:, 1]]) & (cd[a[:, 0]] >= cd[a[:, 1]]))
        win = np.where(better, a[:, 0], a[:, 1])
        co, ck = [], []
        for q in range(0, mu, 2):
            p1, p2 = win[q], win[(q + 1) % mu]
            g1, g2 = (optm[p1], keym[p1]), (optm[p2], keym[p2])
            if rng.random() < pc:
                c1, c2 = pop.crossover(g1, g2), pop.crossover(g2, g1)
            else:
                c1, c2 = (g1[0].copy(), g1[1].copy()), (g2[0].copy(), g2[1].copy())
            for o, k in (c1, c2):
                o, k = o.copy(), k.copy()
                pop.mutate(o, k)
                co.append(pop.repair(o)); ck.append(k)
        co, ck = np.array(co[:mu]), np.array(ck[:mu])
        cmk, cex = pop.evaluate(co, ck)
        cobj = objectives(co, cmk)
        update_archive(co, cobj, cex)
        if len(archive) > 400:
            ks = list(archive)
            pp = np.array([archive[k] for k in ks])
            archive = {ks[i]: archive[ks[i]] for i in pareto_2d(pp)}
        allo = np.vstack([optm, co]); allk = np.vstack([keym, ck])
        allm = np.concatenate([mk, cmk]); alle = np.concatenate([ex, cex]); allob = np.vstack([objs, cobj])
        fr = _fronts(allob, alle)
        sel: list[int] = []
        for f in fr:
            if len(sel) + len(f) <= mu:
                sel.extend(f.tolist())
            else:
                d = _crowding(allob[f])
                sel.extend(f[np.argsort(-d, kind="stable")][: mu - len(sel)].tolist())
                break
        sel = np.array(sel)
        optm, keym, mk, ex, objs = allo[sel], allk[sel], allm[sel], alle[sel], allob[sel]
    # final non-dominated archive
    if archive:
        keys = list(archive)
        pts = np.array([archive[k] for k in keys])
        nd = pareto_2d(pts)
        front = pts[nd]
        front_opt = np.array([keys[i] for i in nd])
        o = np.argsort(front[:, 0])
        front, front_opt = front[o], front_opt[o]
    else:
        front, front_opt = np.empty((0, 2)), np.empty((0, inst.n), dtype=int)
    return tr.result("nsga2", generations=gens, front=front, front_opt=front_opt, second=second)
