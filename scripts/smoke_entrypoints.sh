#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}"

python -m src.train.train_cpc --help >/dev/null
python -m src.evaluation.evaluate_snn --help >/dev/null
python -m src.evaluation.evaluate_background --help >/dev/null
python -m src.evaluation.fit_calibration --help >/dev/null
python -m src.evaluation.mine_hard_negatives --help >/dev/null
python -m src.inference_anomaly --help >/dev/null

echo "Smoke OK: entrypointy CLI są uruchamialne."
