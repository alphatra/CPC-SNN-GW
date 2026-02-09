# CPC-SNN Gravitational Waves Detection

## Phase-2 Benchmark Snapshot (2026-02-09)

Protocol: `evaluate_background --method swapped_pairs --swaps 30` on full noise pool (`indices_noise.json`) and full signal pool (`indices_signal.json`).

| Variant | n (repeats) | TPR@1e-3 | TPR@1e-4 | TPR@1e-5 | EVT TPR@1e-4 |
|---|---:|---:|---:|---:|---:|
| A: TF2D base | 1 | 0.3703 ± 0.0000 | 0.0379 ± 0.0000 | 0.0011 ± 0.0000 | 0.0276 ± 0.0000 |
| B: TF2D + tail-aware | 1 | 0.7111 ± 0.0000 | 0.0381 ± 0.0000 | 0.0018 ± 0.0000 | 0.0024 ± 0.0000 |
| C: TF2D + tail-aware + hard-negatives | 5 | 0.9694 ± 0.0024 | 0.8401 ± 0.0147 | 0.6962 ± 0.0450 | 0.8615 ± 0.0200 |
| C + temperature calibration | 5 | 0.9695 ± 0.0007 | 0.8373 ± 0.0144 | 0.7236 ± 0.0244 | 0.8292 ± 0.0108 |
| C2 (ablation): soft mining (`boost=3`) | 1 | 0.9574 ± 0.0000 | 0.4883 ± 0.0000 | 0.2347 ± 0.0000 | 0.0427 ± 0.0000 |
| C2 + temperature calibration | 1 | 0.9530 ± 0.0000 | 0.5384 ± 0.0000 | 0.3039 ± 0.0000 | 0.0798 ± 0.0000 |
| C3 (ablation): balanced mining (`boost=5`) | 5 | 0.9634 ± 0.0011 | 0.7755 ± 0.0141 | 0.4565 ± 0.0661 | 0.5070 ± 0.0155 |
| C3 + temperature calibration | 5 | 0.9607 ± 0.0019 | 0.7640 ± 0.0222 | 0.4670 ± 0.0991 | 0.6463 ± 0.0314 |

Notes:
- For A/B currently only one full swapped run is available (`n=1`), so reported std is `0.0000`.
- C/C+calibration and C3/C3+calibration are averaged over 5 independent swapped-pairs runs from `reports/repeats/`.

## Results (Reports)

This benchmark snapshot is documented in:
- `reports/final_benchmark_table.md`

Primary raw report files used for the table:
- `reports/bg_phase2_tf2d_base_swapped30.txt`
- `reports/bg_phase2_tf2d_tail_swapped30.txt`
- `reports/bg_phase2_tf2d_tail_hn_v1_swapped30.txt`
- `reports/bg_phase2_tf2d_tail_hn_v1_swapped30_cal_temp.txt`
- `reports/bg_phase2_tf2d_tail_hn_v1_swapped30_nohn.txt`
- `reports/bg_phase2_tf2d_tail_hn_v1_swapped30_nohn_cal_temp.txt`
- `reports/calib_phase2_tf2d_tail_hn_v1_temp.json`
- `reports/bg_phase2_tf2d_tail_hn_v2_soft_swapped30.txt`
- `reports/bg_phase2_tf2d_tail_hn_v2_soft_swapped30_cal_temp.txt`
- `reports/calib_phase2_tf2d_tail_hn_v2_soft_temp.json`
- `reports/bg_phase2_tf2d_tail_hn_v3_boost5_swapped30.txt`
- `reports/bg_phase2_tf2d_tail_hn_v3_boost5_swapped30_cal_temp.txt`
- `reports/calib_phase2_tf2d_tail_hn_v3_boost5_temp.json`

Repeated-run artifacts (for `mean ± std` on C and C+calibration):
- `reports/repeats/c_tail_hn_r1.txt`
- `reports/repeats/c_tail_hn_r2.txt`
- `reports/repeats/c_tail_hn_r3.txt`
- `reports/repeats/c_tail_hn_r4.txt`
- `reports/repeats/c_tail_hn_r5.txt`
- `reports/repeats/c_tail_hn_caltemp_r1.txt`
- `reports/repeats/c_tail_hn_caltemp_r2.txt`
- `reports/repeats/c_tail_hn_caltemp_r3.txt`
- `reports/repeats/c_tail_hn_caltemp_r4.txt`
- `reports/repeats/c_tail_hn_caltemp_r5.txt`
- `reports/repeats/c3_tail_hn_boost5_r{1..5}.txt`
- `reports/repeats/c3_tail_hn_boost5_caltemp_r{1..5}.txt`

Canonical reproduction entrypoint:
- `./scripts/reproduce_phase2_abccalib.sh`
- retrain + reproduce: `./scripts/reproduce_phase2_abccalib.sh --with-train`

Strict OOD (train-early / test-late) entrypoint:
- `./scripts/run_ood_time_protocol.sh`
- run-based OOD split: `./scripts/run_ood_time_protocol.sh --mode run --run-map <id_to_run.json> --train-runs <runs> --test-runs <runs>`

## Strict OOD Snapshot (Latest B/C1/C2/C3 Compare)

Protocol: `evaluate_background --method swapped_pairs --swaps 30` on `reports/ood_time/splits/indices_*_test_late.json`.

| Variant | TPR@1e-3 | TPR@1e-4 | EVT TPR@1e-4 |
|---|---:|---:|---:|
| B: TF2D + tail-aware | 0.6993 | 0.2340 | 0.0157 |
| C1: TF2D + hard-negatives (`boost=10`) | 0.9427 | 0.7629 | 0.6745 |
| C2: TF2D + hard-negatives (`boost=3`) | 0.8383 | 0.5487 | 0.0000 |
| C3: TF2D + hard-negatives (`boost=5`) | 0.9477 | 0.8619 | 0.0055 |
| C3 + temperature calibration | 0.9469 | 0.8479 | 0.0133 |

## Primary KPI

Primary KPI is set to **TPR@FPR=1e-4**.

Reason:
- it is low-FAR enough to be meaningful for detection,
- it remains numerically stable in this evaluation setup,
- it best separates candidate quality without overfitting to ultra-rare tail artifacts.

Secondary KPI: `TPR@1e-5` (used for tie-break and operations tuning).

## Current Candidate (Frozen)

Frozen model candidate (ID benchmark winner):
- checkpoint: `artifacts/final_candidate_phase2_tf2d_tail_hn_v1/best.pt`
- calibration: `artifacts/final_candidate_phase2_tf2d_tail_hn_v1/calib_phase2_tf2d_tail_hn_v1_temp.json`
- commit pin: `artifacts/final_candidate_phase2_tf2d_tail_hn_v1/git_commit.txt`

Frozen OOD-robust candidate (train-early/test-late winner):
- checkpoint: `artifacts/final_candidate_ood_time_tf2d_tail_hn_v3_boost5/best.pt`
- OOD report: `artifacts/final_candidate_ood_time_tf2d_tail_hn_v3_boost5/ood_eval_swapped30.txt`
- OOD summary: `artifacts/final_candidate_ood_time_tf2d_tail_hn_v3_boost5/ood_summary.md`
- manifest: `artifacts/final_candidate_ood_time_tf2d_tail_hn_v3_boost5/manifest.json`
- freeze decision note: `reports/ood_time/freeze_note.md`

Operational policy:
- for FAR near `1e-4`: prefer **uncalibrated** scores,
- for FAR near `1e-5` and below: prefer **temperature-calibrated** scores.

## 1D Track Status

The 1D path (SpikingCNN time-series branch) is treated as a **separate bugfix stream** and does not block TF2D publication.
