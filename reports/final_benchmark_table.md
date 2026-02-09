# Final Benchmark Table (Phase 2)

Generated: 2026-02-09 12:23:25Z
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
- Repeats directory: `/Users/gracjanziemianski/Documents/CPC-SNN-GravitationalWavesDetection/reports/repeats`

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
