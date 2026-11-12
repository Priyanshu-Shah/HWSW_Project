# Laptop combination run (all data synthetic)

```
date: 2026-11-08 01:33:45.016036
who: soham
platform: Windows-11-10.0.26300-SP0
machine: AMD64  processor: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel
cpu_count: 16
python: 3.13.3
disk free: 97.5 GB
packages: torch 2.14.1, numpy 2.5.4, ortools 9.15.6755, highspy 1.15.1, pulp 3.3.2, pydantic 2.14.1, numba 0.68.1
```

## A. GNN as warm start for SA / tabu / GA

Per instance: median over seeds of makespan / reference, then mean over instances (Thakar's metric). `win/tie/loss` compares each GNN start with the `heft` start of the same method and budget, per instance (tie = within 0.1%). Plans failing re-verification: 0. Final plans where Thakar's simulator and the handover's sim.evaluate disagree (>1e-6 relative): 0 of 4698 rows. Rows left out because the run took > 1.5 x its budget + 1 s (standby or heavy load): 0 of 4698.

### test10 (87 instances)

| method | budget | heft start | gnn start | gnn16 start | gnn vs heft (win/tie/loss) | gnn16 vs heft |
|---|---:|---:|---:|---:|---|---|
| sa | 10 ms | 1.049 | 1.142 | 1.219 | 2/40/45 | 0/34/53 |
| sa | 100 ms | 1.022 | 1.023 | 1.023 | 4/76/7 | 3/76/8 |
| sa | 1000 ms | 1.019 | 1.019 | 1.021 | 2/82/3 | 2/82/3 |
| tabu | 10 ms | 1.068 | 1.083 | 1.219 | 9/49/29 | 0/39/48 |
| tabu | 100 ms | 1.036 | 1.026 | 1.030 | 6/77/4 | 7/74/6 |
| tabu | 1000 ms | 1.022 | 1.019 | 1.019 | 2/85/0 | 2/85/0 |
| ga | 10 ms | 1.187 | 1.174 | 1.225 | 15/49/23 | 0/60/27 |
| ga | 100 ms | 1.030 | 1.012 | 1.025 | 23/53/11 | 15/51/21 |
| ga | 1000 ms | 1.014 | 1.008 | 1.006 | 9/71/7 | 11/73/3 |

- GNN prior time (gnn, inside the budget): median 7.5 ms, p90 10.8 ms
- GNN prior time (gnn16, inside the budget): median 36.0 ms, p90 48.9 ms

## B. Warm starts for solve_exact, 60-100 tasks, T = 10 s (30 of 30 instances)

Ratio = makespan / best known schedule. **Warm-start generation is inside T** (v2): each variant's solve_exact gets T minus the CPU time of generating its warm starts (at least 1 s). Variants ran in an order shuffled per instance. Instances left out because a solve took > 1.5 x its limit + 5 s even after one rerun (standby or heavy load): 0.

| variant | mean ratio | median | p90 | max | best of heft/sa/gnn/all (incl. ties) | proven optimal | median gap | median solver limit (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| heft | 1.066 | 1.007 | 1.092 | 2.161 | 18/30 | 9/30 | 1.9% | 10.0 |
| heft_rep | 1.084 | 1.006 | 1.133 | 2.277 | — | 9/30 | 1.3% | 10.0 |
| sa | 1.040 | 1.009 | 1.094 | 1.317 | 19/30 | 9/30 | 2.3% | 9.0 |
| gnn | 1.071 | 1.009 | 1.100 | 1.862 | 17/30 | 10/30 | 1.4% | 9.8 |
| all | 1.058 | 1.006 | 1.137 | 1.590 | 20/30 | 10/30 | 1.9% | 8.8 |

**Noise floor** (heft vs heft_rep, same settings, another seed): median 0.03%, p90 3.75%, max 27.82%; they differ by more than 0.1% on 12 of 26 instances. Treat smaller differences between variants as noise.

| variant vs heft | win / tie / loss (tie = within 0.1%) | wins larger than the instance's heft-vs-heft_rep spread |
|---|---|---:|
| sa | 7 / 13 / 6 | 2 |
| gnn | 3 / 17 / 6 | 3 |
| all | 7 / 13 / 6 | 2 |

| warm start alone | mean ratio | median | generation CPU s (median, charged to T) | wall s (median) |
|---|---:|---:|---:|---:|
| sa | 1.134 | 1.093 | 0.95 | 1.01 |
| gnn | 1.655 | 1.624 | 0.24 | 0.27 |

