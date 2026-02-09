# Strict OOD Benchmark (Train-Early / Test-Late)

Split mode: `time`
Counts: noise_train=3570, noise_test=1276, signal_train=3570, signal_test=1275

Protocol: `evaluate_background --method swapped_pairs --swaps 30` on `test_late` split.

| Variant | TPR@1e-3 | TPR@1e-4 | TPR@1e-5 | EVT TPR@1e-4 |
|---|---:|---:|---:|---:|
| B: TF2D + tail-aware | 0.6993 | 0.2340 | n/a | 0.0157 |
| C1: TF2D + hard-negatives (boost=10) | 0.9427 | 0.7629 | n/a | 0.6745 |
| C2: TF2D + hard-negatives (boost=3) | 0.8383 | 0.5487 | n/a | 0.0000 |
| C3: TF2D + hard-negatives (boost=5) | 0.9477 | 0.8619 | n/a | 0.0055 |
| C3 + temperature calibration | 0.9469 | 0.8479 | n/a | 0.0133 |

Primary KPI: `TPR@1e-4`
Secondary KPI: `TPR@1e-5`

Interpretation rule (primary KPI):
- Prefer variant with highest `TPR@1e-4` on strict OOD.
- Use calibration primarily for probability quality and deeper-tail operation.
