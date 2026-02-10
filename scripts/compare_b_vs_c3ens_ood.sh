#!/usr/bin/env bash
set -euo pipefail

# Compare B (single checkpoint) vs C3 ensemble (5 checkpoints) on strict OOD.
# Runs repeated swapped-pairs evaluations and prints TPR@1e-4 mean/std.
#
# Usage:
#   bash scripts/compare_b_vs_c3ens_ood.sh
#
# Optional env vars:
#   ROOT, H5, N_OOD, S_OOD, REPEATS, SWAPS, BATCH_SIZE, OUT_DIR, DEVICE

ROOT="${ROOT:-$(pwd)}"
H5="${H5:-$ROOT/data/cpc_snn_train.h5}"
N_OOD="${N_OOD:-$ROOT/reports/ood_time/splits/indices_noise_test_late.json}"
S_OOD="${S_OOD:-$ROOT/reports/ood_time/splits/indices_signal_test_late.json}"
REPEATS="${REPEATS:-5}"
SWAPS="${SWAPS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OUT_DIR="${OUT_DIR:-$ROOT/reports/ood_compare_b_vs_c3ens}"
DEVICE="${DEVICE:-mps}"

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

CKPT_B="${CKPT_B:-$ROOT/checkpoints/phase2_tf2d_tail/best.pt}"
CKPT_C3_S1="${CKPT_C3_S1:-$ROOT/checkpoints/phase2_tf2d_tail_hn_v3_boost5_seed41/best.pt}"
CKPT_C3_S2="${CKPT_C3_S2:-$ROOT/checkpoints/phase2_tf2d_tail_hn_v3_boost5_seed42/best.pt}"
CKPT_C3_S3="${CKPT_C3_S3:-$ROOT/checkpoints/phase2_tf2d_tail_hn_v3_boost5_seed43/best.pt}"
CKPT_C3_S4="${CKPT_C3_S4:-$ROOT/checkpoints/phase2_tf2d_tail_hn_v3_boost5_seed44/best.pt}"
CKPT_C3_S5="${CKPT_C3_S5:-$ROOT/checkpoints/phase2_tf2d_tail_hn_v3_boost5_seed45/best.pt}"

CKPT_B="$(prefer_ckpt "$CKPT_B")"
CKPT_C3_S1="$(prefer_ckpt "$CKPT_C3_S1")"
CKPT_C3_S2="$(prefer_ckpt "$CKPT_C3_S2")"
CKPT_C3_S3="$(prefer_ckpt "$CKPT_C3_S3")"
CKPT_C3_S4="$(prefer_ckpt "$CKPT_C3_S4")"
CKPT_C3_S5="$(prefer_ckpt "$CKPT_C3_S5")"

mkdir -p "$OUT_DIR"
export OUT_DIR

for i in $(seq 1 "$REPEATS"); do
  python -m src.evaluation.evaluate_background \
    --checkpoint "$CKPT_B" \
    --device "$DEVICE" \
    --noise_h5 "$H5" \
    --indices_noise "$N_OOD" --indices_signal "$S_OOD" \
    --method swapped_pairs --swaps "$SWAPS" \
    --batch_size "$BATCH_SIZE" --no_mask \
    > "$OUT_DIR/b_r${i}.txt"

  python -m src.evaluation.evaluate_background \
    --checkpoint "$CKPT_C3_S1" \
    --device "$DEVICE" \
    --ensemble_checkpoints "$CKPT_C3_S2" "$CKPT_C3_S3" "$CKPT_C3_S4" "$CKPT_C3_S5" \
    --noise_h5 "$H5" \
    --indices_noise "$N_OOD" --indices_signal "$S_OOD" \
    --method swapped_pairs --swaps "$SWAPS" \
    --batch_size "$BATCH_SIZE" --no_mask \
    > "$OUT_DIR/c3ens_r${i}.txt"
done

python - <<'PY'
import math
import os
import pathlib
import re

out = pathlib.Path(os.environ.get("OUT_DIR", "reports/ood_compare_b_vs_c3ens"))
pat = re.compile(r"(?:FPR=1e-0?4:\s+([0-9.]+)|TPR\s*@\s*1e-4:\s*([0-9.]+))")

def collect(pattern: str):
    vals = []
    for p in sorted(out.glob(pattern)):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        m = pat.search(txt)
        if m:
            vals.append(float(m.group(1) or m.group(2)))
    return vals

def stats(vals):
    if not vals:
        return float("nan"), float("nan")
    n = len(vals)
    mu = sum(vals) / n
    var = sum((x - mu) ** 2 for x in vals) / n
    return mu, math.sqrt(var)

b = collect("b_r*.txt")
c3e = collect("c3ens_r*.txt")
if not b or not c3e:
    raise SystemExit(
        f"[ERROR] Missing inputs in {out}. "
        f"Found B={len(b)} files, C3={len(c3e)} files."
    )
mu_b, sd_b = stats(b)
mu_e, sd_e = stats(c3e)

print(f"[B] repeats={len(b)} TPR@1e-4={mu_b:.4f}±{sd_b:.4f}")
print(f"[C3-ensemble] repeats={len(c3e)} TPR@1e-4={mu_e:.4f}±{sd_e:.4f}")
print(f"[delta ensemble-B] {(mu_e - mu_b):+.4f}")
PY

echo "[OK] outputs -> $OUT_DIR"
