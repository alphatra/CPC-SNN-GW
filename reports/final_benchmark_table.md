# Final Benchmark Table (Phase 2)

Generated: 2026-02-09 21:34:37Z
Protocol: `evaluate_background --method swapped_pairs --swaps 30`

## Main Comparison (A/B/C/C+calib/C2/C2+calib/C3/C3+calib)

| Variant | n | TPR@1e-3 | TPR@1e-4 | TPR@1e-5 | EVT TPR@1e-4 |
|---|---:|---:|---:|---:|---:|
| A: TF2D base | 1 | 0.3703 ± 0.0000 | 0.0379 ± 0.0000 | 0.0011 ± 0.0000 | 0.0276 ± 0.0000 |
| B: TF2D + tail-aware | 1 | 0.7111 ± 0.0000 | 0.0381 ± 0.0000 | 0.0018 ± 0.0000 | 0.0024 ± 0.0000 |
| C: TF2D + tail-aware + hard-negatives | 5 | 0.9694 ± 0.0024 | 0.8401 ± 0.0147 | 0.6962 ± 0.0450 | 0.8615 ± 0.0200 |
| C + temperature calibration | 5 | 0.9695 ± 0.0007 | 0.8373 ± 0.0144 | 0.7236 ± 0.0244 | 0.8292 ± 0.0108 |
| C2 (ablation): TF2D + tail-aware + hard-negatives (soft mining) | 1 | 0.9574 ± 0.0000 | 0.4883 ± 0.0000 | 0.2347 ± 0.0000 | 0.0427 ± 0.0000 |
| C2 + temperature calibration | 1 | 0.9530 ± 0.0000 | 0.5384 ± 0.0000 | 0.3039 ± 0.0000 | 0.0798 ± 0.0000 |
| C3 (ablation): TF2D + tail-aware + hard-negatives (boost=5) | 5 | 0.9634 ± 0.0011 | 0.7755 ± 0.0141 | 0.4565 ± 0.0661 | 0.5070 ± 0.0155 |
| C3 + temperature calibration | 5 | 0.9607 ± 0.0019 | 0.7640 ± 0.0222 | 0.4670 ± 0.0991 | 0.6463 ± 0.0314 |

Notes:
- `n` is the number of report files found for each variant.
- Repeats directory: `reports/repeats`

## NOHN Check (Generalization Control)

| Variant | TPR@1e-4 | TPR@1e-5 |
|---|---:|---:|
| C (NOHN) | 0.9212 | 0.7359 |
| C + temperature (NOHN) | 0.9154 | 0.8333 |

## KPI Policy

- **Primary KPI**: `TPR@1e-4`
- **Secondary KPI**: `TPR@1e-5`
- Operationally:
  - near `FAR=1e-4` prefer uncalibrated scores,
  - near `FAR=1e-5` prefer temperature-calibrated scores.

## Strict OOD Ensemble Check (Train-Early / Test-Late)

Protocol: `evaluate_background --method swapped_pairs --swaps 30` on `test_late` split.

| Variant | n | TPR@1e-3 | TPR@1e-4 | TPR@1e-5 | EVT TPR@1e-4 |
|---|---:|---:|---:|---:|---:|
| B: TF2D + tail-aware (single) | 1 | 0.9457 ± 0.0000 | 0.7422 ± 0.0000 | n/a | 0.8416 ± 0.0000 |
| C3 single checkpoints (seed mean) | 5 | 0.7790 ± 0.2764 | 0.6159 ± 0.2923 | n/a | 0.3162 ± 0.3863 |
| C3 ensemble (5 checkpoints, score-avg) | 1 | 0.9106 ± 0.0000 | 0.8165 ± 0.0000 | n/a | 0.4706 ± 0.0000 |

## Strict OOD Head-To-Head (Repeated, Final)

Protocol: `test_late`, `swapped_pairs`, `swaps=30`, `repeats=5`.
Ensemble composition: seeds `41/42/44/45/46` (drop43_add46).

| Variant | n | TPR@1e-4 |
|---|---:|---:|
| B: TF2D + tail-aware (single checkpoint, repeated eval) | 5 | 0.2682 ± 0.0204 |
| C3 ensemble (5 checkpoints, drop43_add46, repeated eval) | 5 | 0.7947 ± 0.0146 |

Delta (`C3 ensemble - B`) on primary KPI `TPR@1e-4`: **+0.5264**.
