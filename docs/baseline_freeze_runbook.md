# Baseline Freeze Runbook

This runbook defines the only accepted promotion path:
`candidate -> regression gate -> promote -> lock`.

## 1. Generate candidate report
```bash
PYTHONPATH=. python -u scripts/generate_baseline_report_v0.py \
  --device cpu \
  --batch-size 16 \
  --ood-swaps 30 \
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
  --lock configs/baselines/baseline_lock_v1_full.json \
  --max-tpr-drop-abs 0.02 \
  --max-pauc-drop-abs 0.02 \
  --max-ece-increase-abs 0.01 \
  --max-brier-increase-abs 0.01 \
  --max-latency-increase-rel 0.30
```

Stop here if gate fails.

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
  --out configs/baselines/baseline_lock_v1_full.json \
  --tag v1_full_updated \
  --notes "Candidate promoted after passing regression gate."
```

## 5. Commit permanent artifacts
```bash
git add \
  scripts/generate_baseline_report_v0.py \
  scripts/create_baseline_lock.py \
  scripts/check_baseline_regression.py \
  configs/baselines/baseline_lock_v1_full.json \
  docs/baseline_freeze_runbook.md

git commit -m "Baseline freeze workflow: candidate gate and lock update"
git push origin <branch>
```

## Guardrails
- Never overwrite `v1_full` before the regression gate passes.
- Keep lock files in `configs/baselines/` (tracked), not in ignored `reports/`.
- Treat generated reports in `reports/` as ephemeral runtime artifacts.
- Evaluation should prefer `best_kpi.pt` and fall back to `best.pt` automatically.
