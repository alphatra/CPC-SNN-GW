# Strict OOD Benchmark (Train-Early / Test-Late)

Split mode: `time`
Counts: noise_train=3570, noise_test=1276, signal_train=3570, signal_test=1275

Protocol: `evaluate_background --method swapped_pairs --swaps 30` on `test_late` split.

| Variant | TPR@1e-3 | TPR@1e-4 | TPR@1e-5 | EVT TPR@1e-4 |
|---|---:|---:|---:|---:|
| A: TF2D base | 0.5771 | 0.2685 | n/a | 0.0588 |
| B: TF2D + tail-aware | 0.9457 | 0.7422 | n/a | 0.8416 |
| C1: TF2D + hard-negatives (boost=10) | 0.9306 | 0.4700 | n/a | 0.6565 |
| C1 + temperature calibration | 0.9327 | 0.6221 | n/a | 0.3937 |
| C2: TF2D + hard-negatives (boost=3) | 0.8383 | 0.5487 | n/a | 0.0000 |
| C3: TF2D + hard-negatives (boost=5) | 0.9477 | 0.8619 | n/a | 0.0055 |
| C3 + temperature calibration | 0.9469 | 0.8479 | n/a | 0.0133 |
| C3 (5-seed mean, independent) | 0.7790 | 0.6159 | n/a | 0.3162 |
| C3 ensemble (5 checkpoints) | 0.9106 | 0.8165 | n/a | 0.4706 |

Primary KPI: `TPR@1e-4`
Secondary KPI: `TPR@1e-5`

Interpretation rule (primary KPI):
- Prefer variant with highest `TPR@1e-4` on strict OOD.
- Use calibration primarily for probability quality and deeper-tail operation.

## Head-to-Head Repeats (Final OOD Decision)

Protocol: `test_late`, `swapped_pairs`, `swaps=30`, `repeats=5`.
Ensemble composition: seeds `41/42/44/45/46` (drop43_add46).

| Variant | TPR@1e-4 (mean ± std) |
|---|---:|
| B: TF2D + tail-aware | 0.2682 ± 0.0204 |
| C3 ensemble (5 checkpoints, drop43_add46) | 0.7947 ± 0.0146 |

Primary-KPI delta (`C3 ensemble - B`): **+0.5264**.
Decision: **C3 ensemble is OOD-primary**.
