# CPC-SNN Gravitational Waves Detection

Research codebase for low-FAR gravitational-wave event detection using CPC/SNN-style training with a TF2D frontend, tail-aware losses, hard-negative mining, post-hoc calibration, and strict OOD evaluation protocols.

## What This Repository Contains

- End-to-end training for TF2D and SNN/CPC variants.
- Background evaluation at very low false alarm rates (`TPR@1e-4`, `TPR@1e-5`).
- Hard-negative mining and ablation support (`boost=3/5/10`).
- Temperature/isotonic calibration utilities.
- OOD protocols: train-early/test-late and run-based splits.
- Reproducibility scripts for final benchmark tables and decision artifacts.

## Current Decision Snapshot (2026-02-09)

- Primary KPI: **`TPR@1e-4`**.
- ID benchmark reference candidate: **C (phase2_tf2d_tail_hn_v1)**.
- OOD-primary candidate: **C3 ensemble (seeds 41/42/44/45/46)**.
- Strict OOD head-to-head (`repeats=5`):
  - B (single): `0.2682 ± 0.0204`
  - C3 ensemble: `0.7947 ± 0.0146`
  - Delta: `+0.5264`

Canonical decision artifact:
- `reports/final_decision.json`

Full table:
- `reports/final_benchmark_table.md`

## Repository Structure

```text
.
├── src/
│   ├── train/           # training entrypoints
│   ├── evaluation/      # background eval, calibration, mining
│   ├── models/          # model definitions and adapters
│   ├── inference/       # inference helpers
│   └── data_handling/   # datasets and loading utilities
├── scripts/             # reproducibility and OOD protocols
├── tests/               # smoke/integration-style tests
├── reports/             # curated benchmark outputs and decisions
└── data/                # local datasets/indices (ignored in git)
```

## Environment Setup

Python `>=3.12` is required.

### Option A: `uv` (recommended)

```bash
uv sync
```

### Option B: `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data Prerequisites

The main pipeline expects:

- `data/cpc_snn_train.h5`
- `data/indices_noise.json`
- `data/indices_signal.json`

Typical runtime variables:

```bash
ROOT=/path/to/CPC-SNN-GravitationalWavesDetection
H5=$ROOT/data/cpc_snn_train.h5
N=$ROOT/data/indices_noise.json
S=$ROOT/data/indices_signal.json
```

## Quick Start

### 1) Smoke check

```bash
bash scripts/smoke_entrypoints.sh
```

### 2) Train (example: TF2D + tail-aware + hard-negatives)

```bash
python -m src.train.train_cpc \
  --run_name phase2_tf2d_tail_hn_v3_boost5_seed41 \
  --h5_path "$H5" --noise_indices "$N" --signal_indices "$S" \
  --epochs 20 --batch_size 64 --lr 3e-4 --split_strategy time \
  --workers 0 --amp --use_tf2d --no_mask \
  --loss_type focal --focal_gamma 2.0 \
  --lambda_infonce 0.1 --prediction_steps 6 \
  --tail_penalty 2.0 --tail_threshold 0.5 \
  --hard_neg_bce_weight 0.25 \
  --tail_ranking_weight 0.5 --tail_ranking_margin 0.1 \
  --tail_hard_frac 0.2 --tail_hard_min 8 --tail_max_pairs 4096 \
  --hard_negatives_json "$ROOT/data/hard_negatives_union.json" \
  --hard_negative_boost 5 --hard_negative_max 500
```

### 3) Evaluate background (swapped pairs)

```bash
python -m src.evaluation.evaluate_background \
  --checkpoint "$ROOT/checkpoints/phase2_tf2d_tail_hn_v3_boost5_seed41/best.pt" \
  --noise_h5 "$H5" --indices_noise "$N" --indices_signal "$S" \
  --method swapped_pairs --swaps 30 --batch_size 128 --no_mask
```

### 4) Fit temperature calibration

```bash
python -m src.evaluation.fit_calibration \
  --checkpoint_path "$ROOT/checkpoints/phase2_tf2d_tail_hn_v3_boost5_seed41/best.pt" \
  --h5_path "$H5" --noise_indices "$N" --signal_indices "$S" \
  --method temperature --holdout_frac 0.2 \
  --output_json "$ROOT/reports/calib_temp.json"
```

## Reproducibility Scripts

- Full phase-2 A/B/C/C+calib benchmark:
  - `scripts/reproduce_phase2_abccalib.sh`
- Strict OOD protocol (time/run split):
  - `scripts/run_ood_time_protocol.sh`
- B vs C3-ensemble repeated OOD comparison:
  - `scripts/compare_b_vs_c3ens_ood.sh`
- Regenerate benchmark markdown table:
  - `scripts/generate_phase2_results_table.py`

## Key Outputs

- Final benchmark table: `reports/final_benchmark_table.md`
- Final model decision: `reports/final_decision.json`
- OOD freeze note: `reports/ood_time/freeze_note.md`
- B vs C3-ensemble summary:
  - `reports/ood_compare_b_vs_c3ens_drop43_add46/summary.md`

## Notes

- For operating near `FAR ≈ 1e-4`, uncalibrated scores are currently preferred.
- For probability reporting or deeper tail analysis, use calibrated scores.
- The 1D track remains a separate repair stream and does not block TF2D conclusions.
