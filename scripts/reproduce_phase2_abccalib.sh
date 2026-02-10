#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ---------------------------------------------------------------------------
# Phase-2 canonical reproduction script:
#   A: TF2D base
#   B: TF2D tail-aware
#   C: TF2D tail-aware + hard-negatives (phase2_tf2d_tail_hn_v1)
#   C+calib: C with temperature calibration
#
# Default mode:
#   - uses existing checkpoints
#   - re-runs background evaluation repeats
#   - fits calibration for C
#   - generates final markdown table
#   - freezes candidate artifact
#
# Optional:
#   --with-train   retrains A/B/C checkpoints before evaluation
# ---------------------------------------------------------------------------

WITH_TRAIN=0
for arg in "$@"; do
  case "$arg" in
    --with-train) WITH_TRAIN=1 ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [--with-train]"
      exit 1
      ;;
  esac
done

H5="${H5:-$ROOT_DIR/data/cpc_snn_train.h5}"
NOISE_INDICES="${NOISE_INDICES:-$ROOT_DIR/data/indices_noise.json}"
SIGNAL_INDICES="${SIGNAL_INDICES:-$ROOT_DIR/data/indices_signal.json}"
REPEATS="${REPEATS:-5}"
SWAPS="${SWAPS:-30}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-3e-4}"
WORKERS="${WORKERS:-0}"

CKPT_A="$ROOT_DIR/checkpoints/phase2_tf2d_base/best.pt"
CKPT_B="$ROOT_DIR/checkpoints/phase2_tf2d_tail/best.pt"
CKPT_C="$ROOT_DIR/checkpoints/phase2_tf2d_tail_hn_v1/best.pt"

REPORTS_DIR="$ROOT_DIR/reports"
REPEATS_DIR="$REPORTS_DIR/repeats"
CALIB_C="$REPORTS_DIR/calib_phase2_tf2d_tail_hn_v1_temp.json"
TABLE_MD="$REPORTS_DIR/final_benchmark_table.md"

FREEZE_DIR="$ROOT_DIR/artifacts/final_candidate_phase2_tf2d_tail_hn_v1"

HARD_NEG_UNION="${HARD_NEG_UNION:-$ROOT_DIR/data/hard_negatives_union.json}"
HARD_NEG_SWAPPED="${HARD_NEG_SWAPPED:-$ROOT_DIR/data/hard_negatives_from_swapped_top50.json}"
HARD_NEG_MINED="${HARD_NEG_MINED:-$ROOT_DIR/data/hard_negatives_tf2d_tail_ids.json}"

mkdir -p "$REPORTS_DIR" "$REPEATS_DIR" "$FREEZE_DIR"

prefer_ckpt() {
  local p="$1"
  local d
  d="$(dirname "$p")"
  if [[ -f "$d/best_kpi.pt" ]]; then
    echo "$d/best_kpi.pt"
  elif [[ -f "$p" ]]; then
    echo "$p"
  elif [[ -f "$d/best.pt" ]]; then
    echo "$d/best.pt"
  elif [[ -f "$d/latest.pt" ]]; then
    echo "$d/latest.pt"
  else
    echo "$p"
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path"
    exit 1
  fi
}

build_hard_negative_union_if_needed() {
  if [[ -f "$HARD_NEG_UNION" ]]; then
    return
  fi

  if [[ ! -f "$HARD_NEG_SWAPPED" && ! -f "$HARD_NEG_MINED" ]]; then
    echo "Missing hard-negative sources and union file:"
    echo "  $HARD_NEG_UNION"
    echo "  $HARD_NEG_SWAPPED"
    echo "  $HARD_NEG_MINED"
    exit 1
  fi

  python - <<PY
import json
from pathlib import Path

swapped = Path("$HARD_NEG_SWAPPED")
mined = Path("$HARD_NEG_MINED")
out = Path("$HARD_NEG_UNION")

sa = set(json.loads(swapped.read_text())) if swapped.exists() else set()
sb = set(json.loads(mined.read_text())) if mined.exists() else set()
u = sorted(sa | sb)
out.write_text(json.dumps(u, indent=2))
print(f"[HN] saved {out} count={len(u)}")
PY
}

train_a() {
  python -m src.train.train_cpc \
    --run_name phase2_tf2d_base \
    --h5_path "$H5" --noise_indices "$NOISE_INDICES" --signal_indices "$SIGNAL_INDICES" \
    --epochs "$EPOCHS" --batch_size "$TRAIN_BATCH_SIZE" --lr "$LR" --split_strategy time \
    --workers "$WORKERS" --amp \
    --use_tf2d --no_mask \
    --loss_type focal --focal_gamma 2.0 \
    --lambda_infonce 0.1 --prediction_steps 6 \
    --tail_penalty 0.0 --hard_neg_bce_weight 0.0 --tail_ranking_weight 0.0
}

train_b() {
  python -m src.train.train_cpc \
    --run_name phase2_tf2d_tail \
    --h5_path "$H5" --noise_indices "$NOISE_INDICES" --signal_indices "$SIGNAL_INDICES" \
    --epochs "$EPOCHS" --batch_size "$TRAIN_BATCH_SIZE" --lr "$LR" --split_strategy time \
    --workers "$WORKERS" --amp \
    --use_tf2d --no_mask \
    --loss_type focal --focal_gamma 2.0 \
    --lambda_infonce 0.1 --prediction_steps 6 \
    --tail_penalty 2.0 --tail_threshold 0.5 \
    --hard_neg_bce_weight 0.25 \
    --tail_ranking_weight 0.5 --tail_ranking_margin 0.1 \
    --tail_hard_frac 0.2 --tail_hard_min 8 --tail_max_pairs 4096
}

train_c() {
  build_hard_negative_union_if_needed

  python -m src.train.train_cpc \
    --run_name phase2_tf2d_tail_hn_v1 \
    --h5_path "$H5" --noise_indices "$NOISE_INDICES" --signal_indices "$SIGNAL_INDICES" \
    --epochs "$EPOCHS" --batch_size "$TRAIN_BATCH_SIZE" --lr "$LR" --split_strategy time \
    --workers "$WORKERS" --amp \
    --use_tf2d --no_mask \
    --loss_type focal --focal_gamma 2.0 \
    --lambda_infonce 0.1 --prediction_steps 6 \
    --tail_penalty 2.0 --tail_threshold 0.5 \
    --hard_neg_bce_weight 0.25 \
    --tail_ranking_weight 0.5 --tail_ranking_margin 0.1 \
    --tail_hard_frac 0.2 --tail_hard_min 8 --tail_max_pairs 4096 \
    --hard_negatives_json "$HARD_NEG_UNION" \
    --hard_negative_boost 10 --hard_negative_max 3000
}

run_bg_eval() {
  local ckpt="$1"
  local out_txt="$2"
  local calib_json="${3:-}"

  if [[ -n "$calib_json" ]]; then
    python -m src.evaluation.evaluate_background \
      --checkpoint "$ckpt" \
      --noise_h5 "$H5" \
      --indices_noise "$NOISE_INDICES" \
      --indices_signal "$SIGNAL_INDICES" \
      --method swapped_pairs --swaps "$SWAPS" \
      --batch_size "$EVAL_BATCH_SIZE" --no_mask \
      --calibration_json "$calib_json" \
      > "$out_txt"
  else
    python -m src.evaluation.evaluate_background \
      --checkpoint "$ckpt" \
      --noise_h5 "$H5" \
      --indices_noise "$NOISE_INDICES" \
      --indices_signal "$SIGNAL_INDICES" \
      --method swapped_pairs --swaps "$SWAPS" \
      --batch_size "$EVAL_BATCH_SIZE" --no_mask \
      > "$out_txt"
  fi
}

fit_c_temperature_calibration() {
  python -m src.evaluation.fit_calibration \
    --checkpoint_path "$CKPT_C" \
    --h5_path "$H5" \
    --noise_indices "$NOISE_INDICES" \
    --signal_indices "$SIGNAL_INDICES" \
    --method temperature \
    --holdout_frac 0.2 \
    --output_json "$CALIB_C"
}

freeze_candidate() {
  local now_utc
  now_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  cp "$CKPT_C" "$FREEZE_DIR/best.pt"
  cp "$CALIB_C" "$FREEZE_DIR/calib_phase2_tf2d_tail_hn_v1_temp.json"
  git rev-parse HEAD > "$FREEZE_DIR/git_commit.txt" || echo "unknown" > "$FREEZE_DIR/git_commit.txt"

  cat > "$FREEZE_DIR/manifest.json" <<JSON
{
  "candidate_id": "phase2_tf2d_tail_hn_v1",
  "frozen_at_utc": "$now_utc",
  "source_checkpoint": "$CKPT_C",
  "source_calibration": "$CALIB_C",
  "primary_kpi": "TPR@FPR=1e-4",
  "secondary_kpi": "TPR@FPR=1e-5",
  "benchmark_protocol": "evaluate_background --method swapped_pairs --swaps $SWAPS",
  "repeats": $REPEATS
}
JSON
}

require_file "$H5"
require_file "$NOISE_INDICES"
require_file "$SIGNAL_INDICES"

CKPT_A="$(prefer_ckpt "$CKPT_A")"
CKPT_B="$(prefer_ckpt "$CKPT_B")"
CKPT_C="$(prefer_ckpt "$CKPT_C")"

require_file "$CKPT_A"
require_file "$CKPT_B"
require_file "$CKPT_C"

if [[ "$WITH_TRAIN" -eq 1 ]]; then
  echo "[Phase2] Training A (TF2D base)..."
  train_a
  echo "[Phase2] Training B (TF2D tail-aware)..."
  train_b
  echo "[Phase2] Training C (TF2D tail-aware + hard-negatives)..."
  train_c
fi

require_file "$CKPT_A"
require_file "$CKPT_B"
require_file "$CKPT_C"

echo "[Phase2] Running A/B/C swapped-pairs repeats (n=$REPEATS)..."
for i in $(seq 1 "$REPEATS"); do
  echo "  - repeat $i/$REPEATS"
  run_bg_eval "$CKPT_A" "$REPEATS_DIR/a_base_r${i}.txt"
  run_bg_eval "$CKPT_B" "$REPEATS_DIR/b_tail_r${i}.txt"
  run_bg_eval "$CKPT_C" "$REPEATS_DIR/c_tail_hn_r${i}.txt"
done

cp "$REPEATS_DIR/a_base_r1.txt" "$REPORTS_DIR/bg_phase2_tf2d_base_swapped30.txt"
cp "$REPEATS_DIR/b_tail_r1.txt" "$REPORTS_DIR/bg_phase2_tf2d_tail_swapped30.txt"
cp "$REPEATS_DIR/c_tail_hn_r1.txt" "$REPORTS_DIR/bg_phase2_tf2d_tail_hn_v1_swapped30.txt"

echo "[Phase2] Fitting temperature calibration for C..."
fit_c_temperature_calibration
require_file "$CALIB_C"

echo "[Phase2] Running C+temperature swapped-pairs repeats (n=$REPEATS)..."
for i in $(seq 1 "$REPEATS"); do
  echo "  - cal repeat $i/$REPEATS"
  run_bg_eval "$CKPT_C" "$REPEATS_DIR/c_tail_hn_caltemp_r${i}.txt" "$CALIB_C"
done

cp "$REPEATS_DIR/c_tail_hn_caltemp_r1.txt" "$REPORTS_DIR/bg_phase2_tf2d_tail_hn_v1_swapped30_cal_temp.txt"

echo "[Phase2] Generating benchmark table..."
python "$ROOT_DIR/scripts/generate_phase2_results_table.py" \
  --reports_dir "$REPORTS_DIR" \
  --output_md "$TABLE_MD"

echo "[Phase2] Freezing candidate artifact..."
freeze_candidate

echo
echo "Done."
echo "Table: $TABLE_MD"
echo "Frozen candidate: $FREEZE_DIR"
