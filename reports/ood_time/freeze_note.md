# Freeze Note: OOD Candidate (Phase 2)

Date: 2026-02-09

Primary KPI: `TPR@1e-4` (strict OOD, train-early/test-late, swapped-pairs)

Decision:
- Frozen OOD-robust candidate: `C3` (`phase2_tf2d_tail_hn_v3_boost5`)
- Artifact path: `artifacts/final_candidate_ood_time_tf2d_tail_hn_v3_boost5/`

Rationale:
- On strict OOD (`reports/ood_time/summary.md`), C3 is best on primary KPI (`TPR@1e-4`).
- C3 outperforms B and C2 on OOD while keeping strong ID performance in `reports/final_benchmark_table.md`.
- Temperature calibration for C3 improves calibration quality and can help deeper-tail operation, but primary ranking is based on uncalibrated `TPR@1e-4`.

Operational guidance:
- Use uncalibrated C3 score as default for thresholding near `FAR≈1e-4`.
- Use calibrated C3 score for probability reporting and deeper-tail operating points when needed.

