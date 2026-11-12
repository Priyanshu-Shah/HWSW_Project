# Laptop combination run (all data synthetic)

```
date: 2026-11-08 15:28:12.815440
who: thakar
platform: Windows-11-10.0.26200-SP0
machine: AMD64  processor: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD
cpu_count: 12
python: 3.12.4
disk free: 31.3 GB
```

## A. GNN as warm start for SA / tabu / GA

Ratio = makespan / reference (label optimum or best known; for unlabelled big graphs the best schedule any run found). Per instance median over seeds, then mean over instances (Thakar's metric). `w/t/l` = per-instance wins / ties / losses of a GNN start against the `heft` start (tie within 0.1%), `p` = two-sided sign test. Plans failing re-verification: 0. Thakar's simulator vs sim.evaluate disagreements: 0 of 28420 rows. Runs dropped as overrun (sleep / heavy load): 0.

### large | model gnn (27 instances)

| constructive / GNN alone | mean ratio | median wall (ms) |
|---|---:|---:|
| gnn_greedy | 1.964 | 20.9 |
| gnn_pop256 | 1.684 | 16.0 |
| heft | 1.348 | 0.5 |
| magellan | 1.144 | 365.4 |
| peft | 1.265 | 1.2 |

| method | budget | heft start | gnn start | gnnpop start | gnn vs heft w/t/l (p) | gnnpop vs heft w/t/l (p) |
|---|---:|---:|---:|---:|---|---|
| sa | 10 ms | 1.289 | 1.347 | 1.310 | 0/16/11 (0.00098) | 1/16/10 (0.012) |
| sa | 100 ms | 1.176 | 1.177 | 1.163 | 13/8/6 (0.17) | 12/12/3 (0.035) |
| sa | 1 s | 1.126 | 1.129 | 1.139 | 6/14/7 (1) | 6/10/11 (0.33) |
| tabu | 10 ms | 1.290 | 1.341 | 1.348 | 0/17/10 (0.002) | 0/17/10 (0.002) |
| tabu | 100 ms | 1.195 | 1.183 | 1.186 | 8/12/7 (1) | 7/11/9 (0.8) |
| tabu | 1 s | 1.153 | 1.125 | 1.141 | 10/13/4 (0.18) | 9/10/8 (1) |
| ga | 10 ms | 1.348 | 1.347 | 1.348 | 2/25/0 (0.5) | 0/27/0 (1) |
| ga | 100 ms | 1.272 | 1.253 | 1.246 | 8/13/6 (0.79) | 9/13/5 (0.42) |
| ga | 1 s | 1.124 | 1.117 | 1.108 | 10/8/9 (1) | 12/10/5 (0.14) |

- GNN prior time (gnn, inside the budget): median 11.8 ms, p90 21.5 ms
- GNN prior time (gnnpop, inside the budget): median 15.3 ms, p90 18.9 ms

### test10 | model gnn (87 instances)

| constructive / GNN alone | mean ratio | median wall (ms) |
|---|---:|---:|
| gnn_greedy | 1.598 | 5.6 |
| gnn_pop256 | 1.086 | 8.6 |
| heft | 1.225 | 0.2 |
| magellan | 1.036 | 13.8 |
| peft | 1.150 | 0.4 |

| method | budget | heft start | gnn start | gnnpop start | gnn vs heft w/t/l (p) | gnnpop vs heft w/t/l (p) |
|---|---:|---:|---:|---:|---|---|
| sa | 10 ms | 1.035 | 1.043 | 1.033 | 2/71/14 (0.0042) | 4/74/9 (0.27) |
| sa | 100 ms | 1.023 | 1.021 | 1.021 | 4/80/3 (1) | 2/83/2 (1) |
| sa | 1 s | 1.017 | 1.018 | 1.017 | 0/84/3 (0.25) | 1/85/1 (1) |
| tabu | 10 ms | 1.066 | 1.024 | 1.104 | 14/68/5 (0.064) | 8/58/21 (0.024) |
| tabu | 100 ms | 1.033 | 1.019 | 1.018 | 8/75/4 (0.39) | 7/78/2 (0.18) |
| tabu | 1 s | 1.022 | 1.016 | 1.016 | 3/84/0 (0.25) | 3/84/0 (0.25) |
| ga | 10 ms | 1.145 | 1.127 | 1.141 | 13/55/19 (0.38) | 21/53/13 (0.23) |
| ga | 100 ms | 1.026 | 1.007 | 1.006 | 17/66/4 (0.0072) | 18/66/3 (0.0015) |
| ga | 1 s | 1.016 | 1.003 | 1.003 | 8/78/1 (0.039) | 5/81/1 (0.22) |

- GNN prior time (gnn, inside the budget): median 5.4 ms, p90 7.1 ms
- GNN prior time (gnnpop, inside the budget): median 8.1 ms, p90 10.7 ms

### test25 | model gnn (89 instances)

| constructive / GNN alone | mean ratio | median wall (ms) |
|---|---:|---:|
| gnn_greedy | 1.957 | 6.7 |
| gnn_pop256 | 1.277 | 9.9 |
| heft | 1.391 | 0.3 |
| magellan | 1.068 | 55.8 |
| peft | 1.212 | 0.5 |

| method | budget | heft start | gnn start | gnnpop start | gnn vs heft w/t/l (p) | gnnpop vs heft w/t/l (p) |
|---|---:|---:|---:|---:|---|---|
| sa | 10 ms | 1.114 | 1.164 | 1.129 | 9/39/41 (5.6e-06) | 10/44/35 (0.00025) |
| sa | 100 ms | 1.081 | 1.074 | 1.079 | 13/64/12 (1) | 12/65/12 (1) |
| sa | 1 s | 1.065 | 1.069 | 1.068 | 8/74/7 (1) | 5/77/7 (0.77) |
| tabu | 10 ms | 1.142 | 1.134 | 1.274 | 17/39/33 (0.033) | 6/30/53 (1.8e-10) |
| tabu | 100 ms | 1.125 | 1.092 | 1.089 | 21/52/16 (0.51) | 16/58/15 (1) |
| tabu | 1 s | 1.076 | 1.065 | 1.067 | 7/75/7 (1) | 6/75/8 (0.79) |
| ga | 10 ms | 1.376 | 1.332 | 1.383 | 18/62/9 (0.12) | 3/75/11 (0.057) |
| ga | 100 ms | 1.073 | 1.047 | 1.043 | 36/38/15 (0.0046) | 46/32/11 (3.3e-06) |
| ga | 1 s | 1.050 | 1.027 | 1.028 | 23/57/9 (0.02) | 28/49/12 (0.017) |

- GNN prior time (gnn, inside the budget): median 6.3 ms, p90 8.0 ms
- GNN prior time (gnnpop, inside the budget): median 9.5 ms, p90 11.1 ms

