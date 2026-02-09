# Freeze Note: OOD Candidate (Phase 2)

Date: 2026-02-09

Primary KPI: `TPR@1e-4` (strict OOD, train-early/test-late, swapped-pairs)

Decision:
- Frozen OOD-primary candidate: `C3 ensemble` (`phase2_tf2d_tail_hn_v3_boost5`, seeds `41/42/44/45/46`)
- Artifact path: `reports/ood_compare_b_vs_c3ens_drop43_add46/`

Rationale:
- Head-to-head repeated strict OOD comparison (`repeats=5`) gives:
  - `B`: `TPR@1e-4 = 0.2682 ± 0.0204`
  - `C3 ensemble (drop43_add46)`: `TPR@1e-4 = 0.7947 ± 0.0146`
  - Delta: `+0.5264` on primary KPI.
- This provides a materially stronger and more stable strict-OOD operating point than B.
- Calibration remains optional for probability reporting; primary ranking is uncalibrated `TPR@1e-4`.

Operational guidance:
- Use uncalibrated **ensemble-averaged** C3 score for thresholding near `FAR≈1e-4`.
- Use calibrated score only for probability reporting / deeper-tail operating points when needed.
