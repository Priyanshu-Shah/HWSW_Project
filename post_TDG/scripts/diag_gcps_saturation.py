"""Diagnostic for the GCPS reproduction: does the GCN output saturate, so that X' ranks tasks like HGP alone?

The GCPS loss Σ(t_h x + t_s (1 − x)) has no area term and t_h < t_s on every task, so its minimum is x = 1 everywhere.
Prints, per size: epochs run, how saturated X is in the first and last evaluation, the Kendall τ between the
blended score X' and the HGP feature, and the AR of GCPS vs greedy HGP. One TGFF graph per size (synthetic).

Usage: python scripts/diag_gcps_saturation.py [--sizes 20 100 1000]
"""

from __future__ import annotations

import argparse

import numpy as np

from metrics.fidelity import kendall_tau
from solvers.gcps import model as gcps_model
from solvers.gcps.baselines import greedy_hgp
from solvers.gcps.instance import generate
from solvers.gcps.lssp import greedy_area, lssp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[20, 100, 1000])
    args = ap.parse_args()
    sigma = 0.3
    for n in args.sizes:
        inst = generate(n, seed=0, graphs=1)[0]
        scores: list[np.ndarray] = []
        gcps_model.greedy_area = lambda s, a, c: (scores.append(s.copy()), greedy_area(s, a, c))[1]
        try:
            r = gcps_model.gcps(inst, seed=0)
        finally:
            gcps_model.greedy_area = greedy_area
        hgp = (inst.ts - inst.th) / inst.area
        f1 = (hgp - hgp.min()) / (np.ptp(hgp) or 1.0)
        first, last = ((s - sigma * f1) / (1 - sigma) for s in (scores[0], scores[-1]))
        ar_greedy = inst.t_max / lssp(inst, greedy_hgp(inst))
        print(f"n={n}: {r.epochs} epochs (best {r.best_epoch}); AR GCPS {r.ar:.3f}, greedy HGP {ar_greedy:.3f}")
        print(f"  X first eval: min {first.min():.3f} mean {first.mean():.3f}; last eval: min {last.min():.4f} "
              f"mean {last.mean():.4f}, fraction > 0.99 = {np.mean(last > 0.99):.2f}")
        print(f"  Kendall tau(X', HGP): first {kendall_tau(list(scores[0]), list(hgp)):.3f}, "
              f"last {kendall_tau(list(scores[-1]), list(hgp)):.3f}")


if __name__ == "__main__":
    main()
