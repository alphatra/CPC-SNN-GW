# Baseline Report v0

Date: `2026-02-10T13:49:36Z`

## Frozen Baselines
- ID baseline: `C (phase2_tf2d_tail_hn_v1)`
- OOD baseline: `C3 ensemble (5 checkpoints, drop43_add46)`

## Metrics (requested set)

| Scope | TPR@1e-4 | pAUC(norm,1e-4) | ECE | Brier |
|---|---:|---:|---:|---:|
| ID | 0.839608 | 0.919800 | 0.007287 | 0.004910 |
| OOD | 0.787451 | 0.885934 | 0.002443 | 0.002463 |

## Latency

| Scope | batch p50 [ms] | batch p95 [ms] | sample p50 [ms] | sample p95 [ms] |
|---|---:|---:|---:|---:|
| ID | 226.685 | 235.321 | 14.1678 | 14.7076 |
| OOD | 1158.009 | 1179.147 | 72.3755 | 73.6967 |

## Protocol
- ID: standard noise/signal pairs on default indices.
- OOD: noise via `swapped_pairs` on `test_late` split, signal from `test_late` split.
- No calibration applied (uncalibrated scores).
