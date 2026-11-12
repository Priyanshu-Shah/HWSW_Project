# Laptop combination run (all data synthetic)

```
date: 2026-11-07 17:54:30.221740
who: thakar
platform: Windows-11-10.0.26200-SP0
machine: AMD64  processor: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD
cpu_count: 12
python: 3.12.4
disk free: 30.1 GB
```

## A. GNN as warm start for SA / tabu / GA

Per instance: median over seeds of makespan / reference, then mean over instances (Thakar's metric). `win/tie/loss` compares each GNN start with the `heft` start of the same method and budget, per instance (tie = within 0.1%). Plans failing re-verification: 0. Final plans where Thakar's simulator and the handover's sim.evaluate disagree (>1e-6 relative): 0 of 15660 rows.

### large (27 instances)

| method | budget | heft start | gnn start | gnn16 start | gnn vs heft (win/tie/loss) | gnn16 vs heft |
|---|---:|---:|---:|---:|---|---|
| sa | 10 ms | 1.284 | 1.346 | 1.348 | 1/17/9 | 0/18/9 |
| sa | 100 ms | 1.162 | 1.162 | 1.287 | 7/11/9 | 1/11/15 |
| sa | 1000 ms | 1.138 | 1.125 | 1.142 | 13/9/5 | 7/13/7 |
| tabu | 10 ms | 1.286 | 1.339 | 1.348 | 0/16/11 | 0/16/11 |
| tabu | 100 ms | 1.188 | 1.178 | 1.286 | 8/12/7 | 3/11/13 |
| tabu | 1000 ms | 1.149 | 1.149 | 1.148 | 5/14/8 | 5/13/9 |
| ga | 10 ms | 1.348 | 1.348 | 1.348 | 1/26/0 | 0/27/0 |
| ga | 100 ms | 1.240 | 1.228 | 1.333 | 6/11/10 | 0/11/16 |
| ga | 1000 ms | 1.123 | 1.111 | 1.107 | 13/8/6 | 13/4/10 |

- GNN prior time (gnn, inside the budget): median 10.1 ms, p90 18.6 ms
- GNN prior time (gnn16, inside the budget): median 101.3 ms, p90 224.3 ms

### test25 (89 instances)

| method | budget | heft start | gnn start | gnn16 start | gnn vs heft (win/tie/loss) | gnn16 vs heft |
|---|---:|---:|---:|---:|---|---|
| sa | 10 ms | 1.117 | 1.140 | 1.383 | 11/39/39 | 0/31/58 |
| sa | 100 ms | 1.084 | 1.081 | 1.077 | 14/65/10 | 14/62/13 |
| sa | 1000 ms | 1.066 | 1.063 | 1.063 | 9/75/5 | 7/75/7 |
| tabu | 10 ms | 1.141 | 1.119 | 1.383 | 15/41/33 | 0/25/64 |
| tabu | 100 ms | 1.116 | 1.084 | 1.091 | 14/53/22 | 13/51/25 |
| tabu | 1000 ms | 1.076 | 1.067 | 1.067 | 5/77/7 | 6/74/9 |
| ga | 10 ms | 1.374 | 1.306 | 1.391 | 22/57/10 | 0/74/15 |
| ga | 100 ms | 1.070 | 1.049 | 1.082 | 32/32/25 | 17/32/40 |
| ga | 1000 ms | 1.050 | 1.031 | 1.034 | 22/59/8 | 22/55/12 |

- GNN prior time (gnn, inside the budget): median 5.9 ms, p90 7.5 ms
- GNN prior time (gnn16, inside the budget): median 40.4 ms, p90 58.6 ms

