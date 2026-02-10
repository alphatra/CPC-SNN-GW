# Baseline Freeze Runbook

This runbook defines the only accepted promotion path:
`candidate -> regression gate -> promote -> lock`.

## 1. Generate candidate report
```bash
PYTHONPATH=. python -u scripts/generate_baseline_report_v0.py \
  --device mps \
  --use-metal \
  --batch-size 16 \
  --ood-swaps 30 \
  --seed 123 \
  --fit-temp-id \
  --fit-temp-ood \
  --protocol-tag candidate_exp_X \
  --out-json reports/baseline_report_candidate.json \
  --out-md reports/baseline_report_candidate.md
```

Opcjonalnie, jeśli chcesz trzymać kandydata jako artefakt śledzony przez Git (pod CI gate):
```bash
mkdir -p configs/baselines
cp reports/baseline_report_candidate.json configs/baselines/baseline_report_candidate.json
```

## 2. Gate candidate against frozen lock
```bash
PYTHONPATH=. python scripts/check_baseline_regression.py \
  --current-report reports/baseline_report_candidate.json \
  --lock configs/baselines/baseline_lock_v1_full_mps.json \
  --max-tpr-drop-abs 0.02 \
  --max-pauc-drop-abs 0.02 \
  --max-tpr-drop-abs-ood 0.005 \
  --max-pauc-drop-abs-ood 0.01 \
  --max-ece-increase-abs 0.01 \
  --max-brier-increase-abs 0.01 \
  --max-latency-increase-rel 0.30 \
  --require-nondecreasing-scopes ood_baseline \
  --nondecreasing-eps-tpr 0.003 \
  --nondecreasing-eps-pauc 0.002 \
  --enforce-same-device
```

Stop here if gate fails.

## 2b. Recommended atomic flow (gate + promote + lock)
Instead of manual steps 2-4, use one atomic command that updates lock only when gate passes:
```bash
PYTHONPATH=. python scripts/gate_and_promote_baseline.py \
  --current-report reports/baseline_report_candidate.json \
  --current-md reports/baseline_report_candidate.md \
  --promoted-report reports/baseline_report_v1_full.json \
  --promoted-md reports/baseline_report_v1_full.md \
  --lock configs/baselines/baseline_lock_v1_full_mps.json \
  --decision reports/final_decision.json \
  --tag v1_full_mps \
  --notes "Candidate promoted after passing regression gate." \
  --max-tpr-drop-abs 0.02 \
  --max-pauc-drop-abs 0.02 \
  --max-tpr-drop-abs-ood 0.005 \
  --max-pauc-drop-abs-ood 0.01 \
  --max-ece-increase-abs 0.01 \
  --max-brier-increase-abs 0.01 \
  --max-latency-increase-rel 0.30 \
  --require-nondecreasing-scopes ood_baseline \
  --nondecreasing-eps-tpr 0.003 \
  --nondecreasing-eps-pauc 0.002 \
  --enforce-same-device
```

## 3. Promote candidate to v1_full
```bash
cp reports/baseline_report_candidate.json reports/baseline_report_v1_full.json
cp reports/baseline_report_candidate.md reports/baseline_report_v1_full.md
```

## 4. Rebuild lock from promoted report
```bash
PYTHONPATH=. python scripts/create_baseline_lock.py \
  --report reports/baseline_report_v1_full.json \
  --decision reports/final_decision.json \
  --out configs/baselines/baseline_lock_v1_full_mps.json \
  --tag v1_full_mps \
  --notes "Candidate promoted after passing regression gate."
```

## 4b. Build fast/verify decision policy from lock
```bash
PYTHONPATH=. python scripts/build_fast_verify_policy.py \
  --lock configs/baselines/baseline_lock_v1_full_mps.json \
  --scope ood_baseline \
  --confidence-band 0.003 \
  --uncertainty-verify-threshold 0.10 \
  --out configs/policy/decision_policy_mps.json
```

## 4c. Mine hard-negatives v2 from OOD swapped false positives
```bash
PYTHONPATH=. python scripts/mine_hard_negatives_v2.py \
  --checkpoint checkpoints/phase2_tf2d_tail_hn_v3_boost5_seed41/best.pt \
  --ensemble-checkpoints \
    checkpoints/phase2_tf2d_tail_hn_v3_boost5_seed42/best.pt \
    checkpoints/phase2_tf2d_tail_hn_v3_boost5_seed44/best.pt \
    checkpoints/phase2_tf2d_tail_hn_v3_boost5_seed45/best.pt \
    checkpoints/phase2_tf2d_tail_hn_v3_boost5_seed46/best.pt \
  --noise-h5 data/cpc_snn_train.h5 \
  --indices-noise reports/ood_time/splits/indices_noise_test_late.json \
  --swaps 30 \
  --batch-size 128 \
  --device mps \
  --seed 123 \
  --lock configs/baselines/baseline_lock_v1_full_mps.json \
  --top-k 2000 \
  --out-json reports/hardneg_v2/hardneg_v2_ranked.json \
  --out-indices reports/hardneg_v2/indices_hardneg_v2.json
```

## 4d. Bootstrap CI-like comparison for B vs C3 ensemble
```bash
PYTHONPATH=. python scripts/bootstrap_compare_b_vs_c3ens.py \
  --out-dir reports/ood_compare_b_vs_c3ens_drop43_add46 \
  --n-boot 20000 \
  --seed 123 \
  --min-delta 0.0 \
  --min-p-superiority 0.95
```

## 5. Commit permanent artifacts
```bash
git add \
  scripts/generate_baseline_report_v0.py \
  scripts/create_baseline_lock.py \
  scripts/check_baseline_regression.py \
  scripts/build_fast_verify_policy.py \
  scripts/mine_hard_negatives_v2.py \
  scripts/bootstrap_compare_b_vs_c3ens.py \
  configs/baselines/baseline_lock_v1_full_mps.json \
  configs/policy/decision_policy_mps.json \
  docs/baseline_freeze_runbook.md

git commit -m "Baseline freeze workflow: candidate gate and lock update"
git push origin <branch>
```

## Guardrails
- Never overwrite `v1_full` before the regression gate passes.
- Keep lock files in `configs/baselines/` (tracked), not in ignored `reports/`.
- For this branch/workflow we use `mps-only` lock: `baseline_lock_v1_full_mps.json`.
- Treat generated reports in `reports/` as ephemeral runtime artifacts.
- Evaluation should prefer `best_kpi.pt` and fall back to `best.pt` automatically.
