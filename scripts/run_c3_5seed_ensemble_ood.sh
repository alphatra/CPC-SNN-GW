#!/usr/bin/env bash
set -euo pipefail

# Train/evaluate true 5-seed C3 and compare single vs ensemble on strict OOD.
#
# Usage:
#   bash scripts/run_c3_5seed_ensemble_ood.sh
#   SKIP_TRAIN=1 bash scripts/run_c3_5seed_ensemble_ood.sh
#
# Optional env:
#   ROOT, H5, N, S, N_OOD, S_OOD, REPORTS_DIR
#   SEEDS="41 42 43 44 45"

ROOT="${ROOT:-$(pwd)}"
H5="${H5:-$ROOT/data/cpc_snn_train.h5}"
N="${N:-$ROOT/data/indices_noise.json}"
S="${S:-$ROOT/data/indices_signal.json}"
N_OOD="${N_OOD:-$ROOT/reports/ood_time/splits/indices_noise_test_late.json}"
S_OOD="${S_OOD:-$ROOT/reports/ood_time/splits/indices_signal_test_late.json}"
REPORTS_DIR="${REPORTS_DIR:-$ROOT/reports/ood_ensemble_c3}"
SEEDS="${SEEDS:-41 42 43 44 45}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

mkdir -p "$REPORTS_DIR"

declare -a CKPTS=()

for seed in $SEEDS; do
  run_name="phase2_tf2d_tail_hn_v3_boost5_seed${seed}"
  ckpt="$ROOT/checkpoints/${run_name}/best.pt"
  CKPTS+=("$ckpt")

  if [[ "$SKIP_TRAIN" != "1" ]]; then
    python -m src.train.train_cpc \
      --run_name "$run_name" \
      --h5_path "$H5" --noise_indices "$N" --signal_indices "$S" \
      --epochs 20 --batch_size 64 --lr 3e-4 --split_strategy time \
      --workers 0 --amp \
      --use_tf2d --no_mask \
      --loss_type focal --focal_gamma 2.0 \
      --lambda_infonce 0.1 --prediction_steps 6 \
      --tail_penalty 2.0 --tail_threshold 0.5 \
      --hard_neg_bce_weight 0.25 \
      --tail_ranking_weight 0.5 --tail_ranking_margin 0.1 \
      --tail_hard_frac 0.2 --tail_hard_min 8 --tail_max_pairs 4096 \
      --hard_negatives_json "$ROOT/data/hard_negatives_union.json" \
      --hard_negative_boost 5 --hard_negative_max 500 \
      --seed "$seed" \
      --no_wandb
  fi

  python -m src.evaluation.evaluate_background \
    --checkpoint "$ckpt" \
    --noise_h5 "$H5" \
    --indices_noise "$N_OOD" --indices_signal "$S_OOD" \
    --method swapped_pairs --swaps 30 --batch_size 128 --no_mask \
    > "$REPORTS_DIR/c3_seed${seed}_ood_swapped30.txt"
done

first_ckpt="${CKPTS[0]}"
rest_ckpts=("${CKPTS[@]:1}")

python -m src.evaluation.evaluate_background \
  --checkpoint "$first_ckpt" \
  --ensemble_checkpoints "${rest_ckpts[@]}" \
  --noise_h5 "$H5" \
  --indices_noise "$N_OOD" --indices_signal "$S_OOD" \
  --method swapped_pairs --swaps 30 --batch_size 128 --no_mask \
  > "$REPORTS_DIR/c3_ensemble5_ood_swapped30.txt"

python - <<'PY'
import os
import pathlib
import re

reports_dir = pathlib.Path(os.environ.get("REPORTS_DIR", "reports/ood_ensemble_c3"))
pat = re.compile(r"FPR=1e-04:\s+([0-9.]+)")

single_vals = []
for p in sorted(reports_dir.glob("c3_seed*_ood_swapped30.txt")):
    txt = p.read_text(encoding="utf-8", errors="ignore")
    m = pat.search(txt)
    if m:
        single_vals.append(float(m.group(1)))

ens_txt = (reports_dir / "c3_ensemble5_ood_swapped30.txt").read_text(
    encoding="utf-8", errors="ignore"
)
ens_match = pat.search(ens_txt)
ens = float(ens_match.group(1)) if ens_match else float("nan")

if single_vals:
    mean_single = sum(single_vals) / len(single_vals)
    best_single = max(single_vals)
    print(f"[single] n={len(single_vals)} mean TPR@1e-4={mean_single:.4f} best={best_single:.4f}")
print(f"[ensemble] TPR@1e-4={ens:.4f}")
PY

echo "[OK] Reports: $REPORTS_DIR"
