#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Strict OOD protocol:
# 1) Create split: train_early / test_late
# 2) Train A/B/C on train_early (optional)
# 3) Evaluate all variants on test_late only
# 4) Fit temperature on C (train_early) and evaluate C+cal on test_late
# 5) Generate OOD summary markdown
#
# Usage:
#   ./scripts/run_ood_time_protocol.sh
#   ./scripts/run_ood_time_protocol.sh --skip-train
#   ./scripts/run_ood_time_protocol.sh --mode run --run-map data/id_to_run.json --train-runs O2,O3a --test-runs O3b

MODE="time"
SKIP_TRAIN=0
RUN_MAP=""
TRAIN_RUNS=""
TEST_RUNS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"; shift 2 ;;
    --skip-train)
      SKIP_TRAIN=1; shift ;;
    --run-map)
      RUN_MAP="$2"; shift 2 ;;
    --train-runs)
      TRAIN_RUNS="$2"; shift 2 ;;
    --test-runs)
      TEST_RUNS="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"
      exit 1 ;;
  esac
done

H5="${H5:-$ROOT_DIR/data/cpc_snn_train.h5}"
NOISE_ALL="${NOISE_ALL:-$ROOT_DIR/data/indices_noise.json}"
SIGNAL_ALL="${SIGNAL_ALL:-$ROOT_DIR/data/indices_signal.json}"
OOD_DIR="${OOD_DIR:-$ROOT_DIR/reports/ood_time}"
SPLIT_DIR="$OOD_DIR/splits"

TRAIN_FRAC="${TRAIN_FRAC:-0.7}"
GAP_FRAC="${GAP_FRAC:-0.05}"
MIN_PER_CLASS="${MIN_PER_CLASS:-300}"

EPOCHS="${EPOCHS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
LR="${LR:-3e-4}"
WORKERS="${WORKERS:-0}"
SWAPS="${SWAPS:-30}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"

HARD_NEG_UNION="${HARD_NEG_UNION:-$ROOT_DIR/data/hard_negatives_union.json}"
HARD_NEG_SWAPPED="${HARD_NEG_SWAPPED:-$ROOT_DIR/data/hard_negatives_from_swapped_top50.json}"
HARD_NEG_MINED="${HARD_NEG_MINED:-$ROOT_DIR/data/hard_negatives_tf2d_tail_ids.json}"

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

CKPT_A="$ROOT_DIR/checkpoints/ood_time_tf2d_base/best.pt"
CKPT_B="$ROOT_DIR/checkpoints/ood_time_tf2d_tail/best.pt"
CKPT_C="$ROOT_DIR/checkpoints/ood_time_tf2d_tail_hn/best.pt"
CAL_C="$OOD_DIR/calib_ood_time_c_temp.json"

mkdir -p "$OOD_DIR" "$SPLIT_DIR"

require_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "Missing required file: $p"
    exit 1
  fi
}

build_hn_union_if_missing() {
  if [[ -f "$HARD_NEG_UNION" ]]; then
    return
  fi
  python - <<PY
import json
from pathlib import Path
swapped=Path("$HARD_NEG_SWAPPED")
mined=Path("$HARD_NEG_MINED")
out=Path("$HARD_NEG_UNION")
sa=set(json.loads(swapped.read_text())) if swapped.exists() else set()
sb=set(json.loads(mined.read_text())) if mined.exists() else set()
u=sorted(sa|sb)
if not u:
    raise SystemExit("No hard negatives found to build union JSON")
out.write_text(json.dumps(u, indent=2))
print(f"[HN] saved {out} count={len(u)}")
PY
}

echo "[OOD] Creating strict split ($MODE)..."
if [[ "$MODE" == "time" ]]; then
  python "$ROOT_DIR/scripts/create_ood_splits.py" \
    --noise_indices "$NOISE_ALL" \
    --signal_indices "$SIGNAL_ALL" \
    --output_dir "$SPLIT_DIR" \
    --mode time \
    --train_frac "$TRAIN_FRAC" \
    --gap_frac "$GAP_FRAC" \
    --min_per_class "$MIN_PER_CLASS"
elif [[ "$MODE" == "run" ]]; then
  if [[ -z "$RUN_MAP" || -z "$TRAIN_RUNS" || -z "$TEST_RUNS" ]]; then
    echo "For --mode run you must provide --run-map, --train-runs, --test-runs"
    exit 1
  fi
  python "$ROOT_DIR/scripts/create_ood_splits.py" \
    --noise_indices "$NOISE_ALL" \
    --signal_indices "$SIGNAL_ALL" \
    --output_dir "$SPLIT_DIR" \
    --mode run \
    --run_map "$RUN_MAP" \
    --train_runs "$TRAIN_RUNS" \
    --test_runs "$TEST_RUNS" \
    --min_per_class "$MIN_PER_CLASS"
else
  echo "Unsupported mode: $MODE"
  exit 1
fi

N_TRAIN="$SPLIT_DIR/indices_noise_train_early.json"
S_TRAIN="$SPLIT_DIR/indices_signal_train_early.json"
N_TEST="$SPLIT_DIR/indices_noise_test_late.json"
S_TEST="$SPLIT_DIR/indices_signal_test_late.json"

require_file "$H5"
require_file "$N_TRAIN"
require_file "$S_TRAIN"
require_file "$N_TEST"
require_file "$S_TEST"

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  echo "[OOD] Training A on early split..."
  python -m src.train.train_cpc \
    --run_name ood_time_tf2d_base \
    --h5_path "$H5" --noise_indices "$N_TRAIN" --signal_indices "$S_TRAIN" \
    --epochs "$EPOCHS" --batch_size "$TRAIN_BATCH_SIZE" --lr "$LR" --split_strategy time \
    --workers "$WORKERS" --amp \
    --use_tf2d --no_mask \
    --loss_type focal --focal_gamma 2.0 \
    --lambda_infonce 0.1 --prediction_steps 6 \
    --tail_penalty 0.0 --hard_neg_bce_weight 0.0 --tail_ranking_weight 0.0

  echo "[OOD] Training B on early split..."
  python -m src.train.train_cpc \
    --run_name ood_time_tf2d_tail \
    --h5_path "$H5" --noise_indices "$N_TRAIN" --signal_indices "$S_TRAIN" \
    --epochs "$EPOCHS" --batch_size "$TRAIN_BATCH_SIZE" --lr "$LR" --split_strategy time \
    --workers "$WORKERS" --amp \
    --use_tf2d --no_mask \
    --loss_type focal --focal_gamma 2.0 \
    --lambda_infonce 0.1 --prediction_steps 6 \
    --tail_penalty 2.0 --tail_threshold 0.5 \
    --hard_neg_bce_weight 0.25 \
    --tail_ranking_weight 0.5 --tail_ranking_margin 0.1 \
    --tail_hard_frac 0.2 --tail_hard_min 8 --tail_max_pairs 4096

  build_hn_union_if_missing

  echo "[OOD] Training C on early split..."
  python -m src.train.train_cpc \
    --run_name ood_time_tf2d_tail_hn \
    --h5_path "$H5" --noise_indices "$N_TRAIN" --signal_indices "$S_TRAIN" \
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
fi

CKPT_A="$(prefer_ckpt "$CKPT_A")"
CKPT_B="$(prefer_ckpt "$CKPT_B")"
CKPT_C="$(prefer_ckpt "$CKPT_C")"

require_file "$CKPT_A"
require_file "$CKPT_B"
require_file "$CKPT_C"

echo "[OOD] Evaluating A/B/C on test-late split..."
python -m src.evaluation.evaluate_background \
  --checkpoint "$CKPT_A" --noise_h5 "$H5" \
  --indices_noise "$N_TEST" --indices_signal "$S_TEST" \
  --method swapped_pairs --swaps "$SWAPS" \
  --batch_size "$EVAL_BATCH_SIZE" --no_mask \
  > "$OOD_DIR/ood_a_base_test_late_swapped${SWAPS}.txt"

python -m src.evaluation.evaluate_background \
  --checkpoint "$CKPT_B" --noise_h5 "$H5" \
  --indices_noise "$N_TEST" --indices_signal "$S_TEST" \
  --method swapped_pairs --swaps "$SWAPS" \
  --batch_size "$EVAL_BATCH_SIZE" --no_mask \
  > "$OOD_DIR/ood_b_tail_test_late_swapped${SWAPS}.txt"

python -m src.evaluation.evaluate_background \
  --checkpoint "$CKPT_C" --noise_h5 "$H5" \
  --indices_noise "$N_TEST" --indices_signal "$S_TEST" \
  --method swapped_pairs --swaps "$SWAPS" \
  --batch_size "$EVAL_BATCH_SIZE" --no_mask \
  > "$OOD_DIR/ood_c_tail_hn_test_late_swapped${SWAPS}.txt"

echo "[OOD] Fitting temperature on train-early split (C)..."
python -m src.evaluation.fit_calibration \
  --checkpoint_path "$CKPT_C" \
  --h5_path "$H5" \
  --noise_indices "$N_TRAIN" \
  --signal_indices "$S_TRAIN" \
  --method temperature \
  --holdout_frac 0.2 \
  --output_json "$CAL_C"

echo "[OOD] Evaluating C+temperature on test-late split..."
python -m src.evaluation.evaluate_background \
  --checkpoint "$CKPT_C" --noise_h5 "$H5" \
  --indices_noise "$N_TEST" --indices_signal "$S_TEST" \
  --method swapped_pairs --swaps "$SWAPS" \
  --batch_size "$EVAL_BATCH_SIZE" --no_mask \
  --calibration_json "$CAL_C" \
  > "$OOD_DIR/ood_c_tail_hn_cal_temp_test_late_swapped${SWAPS}.txt"

echo "[OOD] Generating summary markdown..."
python "$ROOT_DIR/scripts/generate_ood_summary.py" \
  --ood_dir "$OOD_DIR" \
  --swaps "$SWAPS" \
  --output_md "$OOD_DIR/summary.md"

echo
echo "Done."
echo "OOD summary: $OOD_DIR/summary.md"
