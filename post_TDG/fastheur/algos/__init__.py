"""Registry of all methods: name -> solve(inst, fast, seed, budget_s) -> Result."""

from functools import partial

from .evo import solve_ga, solve_nsga2
from .gclp import solve_gclp, solve_mibs
from .heft import solve_heft, solve_peft
from .local import solve_kl, solve_sa, solve_tabu
from .magellan import solve_magellan

METHODS = {
    "heft": solve_heft,
    "peft": solve_peft,
    "gclp": solve_gclp,
    "mibs": solve_mibs,
    "gclp_lat": partial(solve_gclp, target="lb"),
    "mibs_lat": partial(solve_mibs, target="lb"),
    "magellan": solve_magellan,
    "kl1": partial(solve_kl, variant=1),
    "kl2": partial(solve_kl, variant=2),
    "tabu": solve_tabu,
    "sa": solve_sa,
    "ga": solve_ga,
    "nsga2": solve_nsga2,
}
DETERMINISTIC = {"heft", "peft", "gclp", "mibs", "gclp_lat", "mibs_lat", "magellan"}
