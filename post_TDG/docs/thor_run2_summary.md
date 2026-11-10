# Thor run 2 summary (all data synthetic)

```
date: Mon Nov  9 02:23:26 PM IST 2026
Linux roboticsiitj 6.8.12-tegra #1 SMP PREEMPT Tue Dec 30 15:40:41 PST 2025 aarch64 aarch64 aarch64 GNU/Linux
NV Power Mode: MAXN
0
nproc: 14
               total        used        free      shared  buff/cache   available
Mem:           122Gi       9.9Gi       106Gi       120Mi       7.7Gi       112Gi
Swap:             0B          0B          0B
run1: /home/roboticsiitj/Downloads/THOR_RUN_v1
torch 2.10.0+cpu | ortools 9.15.6755 | numba 0.68.1
```

## 1. Training (data scaling, 40-60 tasks, capacity)

Validation = 300 optimal 10-40-task + 100 optimal 40-60-task graphs from run 1's new labels (fixed). Ratio = greedy GNN makespan / optimum, mean. Model kept = best epoch by mean validation ratio. Run 1 reference: 754 training graphs, option accuracy about 0.73.

| model | train graphs | epochs | best epoch | val ratio (best) | 10-40 | 40-60 | val option acc 10-40 / 40-60 (last) | minutes |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| r2_all1040_base_s0 | 17969 | 26.0 | 8 | 1.542 | 1.476 | 1.761 | 0.782 / 0.731 | 60 |
| r2_all_base_s0 | 19028 | 22.9 | 20 | 1.504 | 1.455 | 1.667 | 0.775 / 0.733 | 60 |
| r2_all_base_s1 | 19028 | 22.2 | 22 | 1.500 | 1.452 | 1.659 | 0.776 / 0.728 | 60 |
| r2_all_base_s2 | 19028 | 23.3 | 23 | 1.505 | 1.444 | 1.706 | 0.778 / 0.729 | 60 |
| r2_all_optw_s0 | 19028 | 22.9 | 21 | 1.526 | 1.439 | 1.812 | 0.779 / 0.735 | 60 |
| r2_all_wide_s0 | 19028 | 11.3 | 11 | 1.557 | 1.483 | 1.802 | 0.769 / 0.727 | 60 |
| r2_sub2k_base_s0 | 2000 | 60.0 | 23 | 1.796 | 1.740 | 1.984 | 0.711 / 0.678 | 18 |
| r2_sub6k_base_s0 | 6000 | 60.0 | 46 | 1.606 | 1.556 | 1.773 | 0.765 / 0.715 | 49 |

## 2. GNN alone on unseen test graphs (mean ratio to optimum / best known)

`best64` = best of greedy + 64 samples; `opt map + GNN prio` = optimal mapping scheduled with GNN priorities; `GNN map + HEFT` = GNN mapping scheduled by HEFT. Heuristics do not depend on the model.

### 10-24 tasks

| model | greedy | best64 | opt map + GNN prio | GNN map + HEFT |
|---|---:|---:|---:|---:|
| r2_all1040_base_s0 | 1.377 | 1.072 | 1.070 | 1.350 |
| r2_all_base_s0 | 1.296 | 1.052 | 1.060 | 1.243 |
| r2_all_base_s1 | 1.274 | 1.073 | 1.053 | 1.257 |
| r2_all_base_s2 | 1.357 | 1.070 | 1.058 | 1.336 |
| r2_all_optw_s0 | 1.311 | 1.063 | 1.067 | 1.294 |
| r2_all_wide_s0 | 1.427 | 1.065 | 1.064 | 1.397 |
| r2_sub2k_base_s0 | 1.429 | 1.092 | 1.062 | 1.389 |
| r2_sub6k_base_s0 | 1.575 | 1.131 | 1.067 | 1.544 |
| run1_base_s1 | 1.598 | 1.109 | 1.074 | 1.540 |

Heuristics: heft 1.441, peft 1.172, magellan 1.034

### 25-40 tasks

| model | greedy | best64 | opt map + GNN prio | GNN map + HEFT |
|---|---:|---:|---:|---:|
| r2_all1040_base_s0 | 1.650 | 1.183 | 1.177 | 1.543 |
| r2_all_base_s0 | 1.430 | 1.180 | 1.178 | 1.347 |
| r2_all_base_s1 | 1.434 | 1.167 | 1.166 | 1.354 |
| r2_all_base_s2 | 1.580 | 1.200 | 1.172 | 1.477 |
| r2_all_optw_s0 | 1.461 | 1.189 | 1.189 | 1.390 |
| r2_all_wide_s0 | 1.503 | 1.199 | 1.154 | 1.420 |
| r2_sub2k_base_s0 | 1.698 | 1.232 | 1.201 | 1.570 |
| r2_sub6k_base_s0 | 1.583 | 1.243 | 1.189 | 1.509 |
| run1_base_s1 | 1.957 | 1.344 | 1.200 | 1.872 |

Heuristics: heft 1.609, peft 1.203, magellan 1.082

### 60-100 tasks

| model | greedy | best64 | opt map + GNN prio | GNN map + HEFT |
|---|---:|---:|---:|---:|
| r2_all1040_base_s0 | 1.632 | 1.451 | 1.411 | 1.463 |
| r2_all_base_s0 | 1.489 | 1.362 | 1.413 | 1.335 |
| r2_all_base_s1 | 1.433 | 1.294 | 1.402 | 1.302 |
| r2_all_base_s2 | 1.589 | 1.391 | 1.418 | 1.396 |
| r2_all_optw_s0 | 1.534 | 1.372 | 1.387 | 1.339 |
| r2_all_wide_s0 | 1.595 | 1.424 | 1.403 | 1.379 |
| r2_sub2k_base_s0 | 1.917 | 1.645 | 1.426 | 1.681 |
| r2_sub6k_base_s0 | 1.745 | 1.557 | 1.446 | 1.521 |
| run1_base_s1 | 1.843 | 1.591 | 1.470 | 1.605 |

Heuristics: heft 1.269, peft 1.201

## 3. GNN + search (Thakar's SA / tabu / GA)
Ratio = makespan / reference (label optimum or best known; for unlabelled big graphs the best schedule any run found). Per instance median over seeds, then mean over instances (Thakar's metric). `w/t/l` = per-instance wins / ties / losses of a GNN start against the `heft` start (tie within 0.1%), `p` = two-sided sign test. Plans failing re-verification: 0. Thakar's simulator vs sim.evaluate disagreements: 0 of 12760 rows. Runs dropped as overrun (sleep / heavy load): 0.

### large | model r2_all_base_s1 (27 instances)

| constructive / GNN alone | mean ratio | median wall (ms) |
|---|---:|---:|
| gnn_greedy | 1.503 | 9.4 |
| gnn_pop256 | 1.443 | 9.6 |
| heft | 1.348 | 0.4 |
| magellan | 1.144 | 191.2 |
| peft | 1.265 | 0.5 |

| method | budget | heft start | gnn start | gnnpop start | gnn vs heft w/t/l (p) | gnnpop vs heft w/t/l (p) |
|---|---:|---:|---:|---:|---|---|
| sa | 10 ms | 1.239 | 1.285 | 1.218 | 2/12/13 (0.0074) | 7/11/9 (0.8) |
| sa | 100 ms | 1.158 | 1.155 | 1.153 | 8/11/8 (1) | 8/10/9 (1) |
| sa | 1 s | 1.132 | 1.120 | 1.124 | 5/15/7 (0.77) | 7/14/6 (1) |
| tabu | 10 ms | 1.255 | 1.270 | 1.307 | 4/13/10 (0.18) | 2/13/12 (0.013) |
| tabu | 100 ms | 1.171 | 1.140 | 1.149 | 9/11/7 (0.8) | 8/12/7 (1) |
| tabu | 1 s | 1.134 | 1.117 | 1.118 | 9/15/3 (0.15) | 9/14/4 (0.27) |
| ga | 10 ms | 1.346 | 1.344 | 1.317 | 2/24/1 (1) | 5/22/0 (0.062) |
| ga | 100 ms | 1.155 | 1.120 | 1.093 | 14/8/5 (0.064) | 15/8/4 (0.019) |
| ga | 1 s | 1.106 | 1.086 | 1.073 | 11/8/8 (0.65) | 12/7/8 (0.5) |
| ga | 10 s | 1.088 | 1.076 | 1.065 | 10/10/7 (0.63) | 10/10/7 (0.63) |

- GNN prior time (gnn, inside the budget): median 7.2 ms, p90 10.5 ms
- GNN prior time (gnnpop, inside the budget): median 9.4 ms, p90 11.6 ms

### test25 | model r2_all_base_s1 (89 instances)

| constructive / GNN alone | mean ratio | median wall (ms) |
|---|---:|---:|
| gnn_greedy | 1.434 | 4.7 |
| gnn_pop256 | 1.178 | 5.4 |
| heft | 1.391 | 0.2 |
| magellan | 1.068 | 24.8 |
| peft | 1.212 | 0.3 |

| method | budget | heft start | gnn start | gnnpop start | gnn vs heft w/t/l (p) | gnnpop vs heft w/t/l (p) |
|---|---:|---:|---:|---:|---|---|
| sa | 10 ms | 1.115 | 1.096 | 1.090 | 12/49/28 (0.017) | 14/52/23 (0.19) |
| sa | 100 ms | 1.076 | 1.069 | 1.068 | 13/66/10 (0.68) | 16/65/8 (0.15) |
| sa | 1 s | 1.061 | 1.062 | 1.063 | 6/77/6 (1) | 7/79/3 (0.34) |
| tabu | 10 ms | 1.139 | 1.081 | 1.086 | 23/43/23 (1) | 21/46/22 (1) |
| tabu | 100 ms | 1.095 | 1.068 | 1.070 | 12/68/9 (0.66) | 12/71/6 (0.24) |
| tabu | 1 s | 1.069 | 1.059 | 1.059 | 6/81/2 (0.29) | 6/81/2 (0.29) |
| ga | 10 ms | 1.210 | 1.171 | 1.104 | 24/30/35 (0.19) | 51/23/15 (1e-05) |
| ga | 100 ms | 1.053 | 1.030 | 1.027 | 28/47/14 (0.044) | 30/46/13 (0.014) |
| ga | 1 s | 1.043 | 1.023 | 1.024 | 23/59/7 (0.0052) | 21/55/13 (0.23) |
| ga | 10 s | 1.035 | 1.017 | 1.020 | 13/66/10 (0.68) | 13/64/12 (1) |

- GNN prior time (gnn, inside the budget): median 4.5 ms, p90 5.2 ms
- GNN prior time (gnnpop, inside the budget): median 5.9 ms, p90 6.6 ms

### big100 | model r2_all_base_s1 (5 instances)

| constructive / GNN alone | mean ratio | median wall (ms) |
|---|---:|---:|
| gnn_greedy | 1.854 | 9.5 |
| gnn_pop256 | 2.047 | 11.6 |
| heft | 2.072 | 0.4 |
| magellan | 1.231 | 664.5 |
| peft | 1.484 | 0.7 |

| method | budget | heft start | gnn start | gnnpop start | gnn vs heft w/t/l (p) | gnnpop vs heft w/t/l (p) |
|---|---:|---:|---:|---:|---|---|
| sa | 100 ms | 1.396 | 1.352 | 1.268 | 2/1/2 (1) | 2/1/2 (1) |
| sa | 1 s | 1.186 | 1.135 | 1.196 | 3/1/1 (0.62) | 2/0/3 (1) |
| sa | 10 s | 1.086 | 1.081 | 1.077 | 2/1/2 (1) | 1/3/1 (1) |
| tabu | 100 ms | 1.456 | 1.206 | 1.243 | 5/0/0 (0.062) | 4/1/0 (0.12) |
| tabu | 1 s | 1.247 | 1.161 | 1.150 | 4/1/0 (0.12) | 3/1/1 (0.62) |
| tabu | 10 s | 1.180 | 1.080 | 1.078 | 3/1/1 (0.62) | 4/1/0 (0.12) |
| ga | 100 ms | 1.433 | 1.225 | 1.124 | 5/0/0 (0.062) | 5/0/0 (0.062) |
| ga | 1 s | 1.306 | 1.107 | 1.041 | 5/0/0 (0.062) | 5/0/0 (0.062) |
| ga | 10 s | 1.224 | 1.093 | 1.029 | 5/0/0 (0.062) | 5/0/0 (0.062) |

- GNN prior time (gnn, inside the budget): median 9.7 ms, p90 13.3 ms
- GNN prior time (gnnpop, inside the budget): median 12.2 ms, p90 15.6 ms

### big100 | model run1_base_s1 (5 instances)

| constructive / GNN alone | mean ratio | median wall (ms) |
|---|---:|---:|
| gnn_greedy | 3.190 | 11.2 |
| gnn_pop256 | 2.552 | 12.6 |
| heft | 2.087 | 0.5 |
| magellan | 1.227 | 662.0 |
| peft | 1.488 | 0.8 |

| method | budget | heft start | gnn start | gnnpop start | gnn vs heft w/t/l (p) | gnnpop vs heft w/t/l (p) |
|---|---:|---:|---:|---:|---|---|
| sa | 100 ms | 1.269 | 1.213 | 1.249 | 3/0/2 (1) | 3/0/2 (1) |
| sa | 1 s | 1.142 | 1.124 | 1.158 | 1/0/4 (0.38) | 1/0/4 (0.38) |
| sa | 10 s | 1.103 | 1.088 | 1.101 | 3/1/1 (0.62) | 3/1/1 (0.62) |
| tabu | 100 ms | 1.458 | 1.567 | 1.354 | 2/1/2 (1) | 3/2/0 (0.25) |
| tabu | 1 s | 1.251 | 1.268 | 1.179 | 1/1/3 (0.62) | 3/1/1 (0.62) |
| tabu | 10 s | 1.196 | 1.070 | 1.076 | 2/1/2 (1) | 3/1/1 (0.62) |
| ga | 100 ms | 1.438 | 1.455 | 1.326 | 2/0/3 (1) | 3/0/2 (1) |
| ga | 1 s | 1.305 | 1.312 | 1.126 | 3/0/2 (1) | 4/0/1 (0.38) |
| ga | 10 s | 1.222 | 1.253 | 1.113 | 2/0/3 (1) | 4/0/1 (0.38) |

- GNN prior time (gnn, inside the budget): median 10.7 ms, p90 13.8 ms
- GNN prior time (gnnpop, inside the budget): median 12.9 ms, p90 16.7 ms

### big200 | model r2_all_base_s1 (5 instances)

| constructive / GNN alone | mean ratio | median wall (ms) |
|---|---:|---:|
| heft | 1.213 | 1.0 |
| magellan | 1.070 | 3610.2 |
| peft | 1.241 | 1.4 |

| method | budget | heft start | gnnpop start | gnnpop vs heft w/t/l (p) |
|---|---:|---:|---:|---|
| sa | 100 ms | 1.147 | 1.169 | 0/2/3 (0.25) |
| sa | 1 s | 1.115 | 1.122 | 2/1/2 (1) |
| sa | 10 s | 1.057 | 1.046 | 3/2/0 (0.25) |
| tabu | 100 ms | 1.202 | 1.154 | 2/3/0 (0.5) |
| tabu | 1 s | 1.141 | 1.101 | 1/2/2 (1) |
| tabu | 10 s | 1.045 | 1.069 | 0/1/4 (0.12) |
| ga | 100 ms | 1.181 | 1.164 | 2/3/0 (0.5) |
| ga | 1 s | 1.071 | 1.043 | 4/1/0 (0.12) |
| ga | 10 s | 1.060 | 1.033 | 4/1/0 (0.12) |

- GNN prior time (gnnpop, inside the budget): median 22.0 ms, p90 46.9 ms

### big200 | model run1_base_s1 (5 instances)

| constructive / GNN alone | mean ratio | median wall (ms) |
|---|---:|---:|
| heft | 1.219 | 0.9 |
| magellan | 1.075 | 3632.2 |
| peft | 1.247 | 1.5 |

| method | budget | heft start | gnnpop start | gnnpop vs heft w/t/l (p) |
|---|---:|---:|---:|---|
| sa | 100 ms | 1.161 | 1.168 | 2/1/2 (1) |
| sa | 1 s | 1.096 | 1.098 | 3/1/1 (0.62) |
| sa | 10 s | 1.060 | 1.056 | 2/2/1 (1) |
| tabu | 100 ms | 1.208 | 1.203 | 1/4/0 (1) |
| tabu | 1 s | 1.148 | 1.080 | 3/2/0 (0.25) |
| tabu | 10 s | 1.050 | 1.056 | 1/2/2 (1) |
| ga | 100 ms | 1.189 | 1.193 | 0/3/2 (0.5) |
| ga | 1 s | 1.076 | 1.041 | 3/1/1 (0.62) |
| ga | 10 s | 1.066 | 1.023 | 4/1/0 (0.12) |

- GNN prior time (gnnpop, inside the budget): median 21.8 ms, p90 50.2 ms

### big500 | model r2_all_base_s1 (5 instances)

| constructive / GNN alone | mean ratio | median wall (ms) |
|---|---:|---:|
| heft | 1.306 | 2.8 |
| peft | 1.249 | 4.1 |

| method | budget | heft start | gnnpop start | gnnpop vs heft w/t/l (p) |
|---|---:|---:|---:|---|
| sa | 100 ms | 1.269 | 1.303 | 0/3/2 (0.5) |
| sa | 1 s | 1.106 | 1.170 | 0/1/4 (0.12) |
| sa | 10 s | 1.051 | 1.100 | 1/0/4 (0.38) |
| tabu | 100 ms | 1.306 | 1.306 | 0/5/0 (1) |
| tabu | 1 s | 1.178 | 1.257 | 0/2/3 (0.25) |
| tabu | 10 s | 1.108 | 1.135 | 2/1/2 (1) |
| ga | 100 ms | 1.306 | 1.306 | 0/5/0 (1) |
| ga | 1 s | 1.182 | 1.243 | 0/2/3 (0.25) |
| ga | 10 s | 1.074 | 1.047 | 5/0/0 (0.062) |

- GNN prior time (gnnpop, inside the budget): median 46.9 ms, p90 50.4 ms

### big500 | model run1_base_s1 (5 instances)

| constructive / GNN alone | mean ratio | median wall (ms) |
|---|---:|---:|
| heft | 1.320 | 2.9 |
| peft | 1.258 | 4.2 |

| method | budget | heft start | gnnpop start | gnnpop vs heft w/t/l (p) |
|---|---:|---:|---:|---|
| sa | 100 ms | 1.277 | 1.320 | 0/3/2 (0.5) |
| sa | 1 s | 1.132 | 1.148 | 1/1/3 (0.62) |
| sa | 10 s | 1.054 | 1.039 | 4/0/1 (0.38) |
| tabu | 100 ms | 1.320 | 1.320 | 0/5/0 (1) |
| tabu | 1 s | 1.187 | 1.249 | 1/2/2 (1) |
| tabu | 10 s | 1.114 | 1.138 | 2/1/2 (1) |
| ga | 100 ms | 1.320 | 1.320 | 0/5/0 (1) |
| ga | 1 s | 1.193 | 1.187 | 1/2/2 (1) |
| ga | 10 s | 1.082 | 1.062 | 4/0/1 (0.38) |

- GNN prior time (gnnpop, inside the budget): median 47.9 ms, p90 53.6 ms

### big1000 | model r2_all_base_s1 (5 instances)

| constructive / GNN alone | mean ratio | median wall (ms) |
|---|---:|---:|
| heft | 1.636 | 1165.9 |
| peft | 1.465 | 1013.5 |

| method | budget | heft start | gnnpop start | gnnpop vs heft w/t/l (p) |
|---|---:|---:|---:|---|
| sa | 100 ms | 1.900 | 1.900 | 0/5/0 (1) |
| sa | 1 s | 1.900 | 1.900 | 0/5/0 (1) |
| sa | 10 s | 1.135 | 1.166 | 0/3/2 (0.5) |
| tabu | 100 ms | 1.900 | 1.900 | 0/5/0 (1) |
| tabu | 1 s | 1.900 | 1.900 | 0/5/0 (1) |
| tabu | 10 s | 1.562 | 1.592 | 0/4/1 (1) |
| ga | 100 ms | 1.900 | 1.900 | 0/5/0 (1) |
| ga | 1 s | 1.900 | 1.900 | 0/5/0 (1) |
| ga | 10 s | 1.258 | 1.168 | 1/0/4 (0.38) |

- GNN prior time (gnnpop, inside the budget): median 256.1 ms, p90 426.8 ms

### big1000 | model run1_base_s1 (5 instances)

| constructive / GNN alone | mean ratio | median wall (ms) |
|---|---:|---:|
| heft | 2.792 | 1172.2 |
| peft | 2.285 | 1020.6 |

| method | budget | heft start | gnnpop start | gnnpop vs heft w/t/l (p) |
|---|---:|---:|---:|---|
| sa | 100 ms | 3.473 | 3.473 | 0/5/0 (1) |
| sa | 1 s | 3.473 | 3.473 | 0/5/0 (1) |
| sa | 10 s | 1.754 | 1.529 | 1/3/1 (1) |
| tabu | 100 ms | 3.473 | 3.473 | 0/5/0 (1) |
| tabu | 1 s | 3.473 | 3.473 | 0/5/0 (1) |
| tabu | 10 s | 2.623 | 1.109 | 2/3/0 (0.5) |
| ga | 100 ms | 3.473 | 3.473 | 0/5/0 (1) |
| ga | 1 s | 3.473 | 3.473 | 0/5/0 (1) |
| ga | 10 s | 2.061 | 1.029 | 2/0/3 (1) |

- GNN prior time (gnnpop, inside the budget): median 256.4 ms, p90 433.8 ms

