#!/usr/bin/env python3
"""Run regression gate and promote baseline only on success.

This script makes the freeze flow atomic:
1) check candidate vs lock
2) if gate passes, promote candidate report
3) rebuild lock from promoted report
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate candidate and promote baseline on pass.")
    ap.add_argument(
        "--current-report",
        type=Path,
        required=True,
        help="Candidate report JSON (e.g. reports/baseline_report_candidate.json).",
    )
    ap.add_argument(
        "--current-md",
        type=Path,
        default=None,
        help="Candidate markdown report (optional).",
    )
    ap.add_argument(
        "--promoted-report",
        type=Path,
        default=Path("reports/baseline_report_v1_full.json"),
        help="Destination promoted JSON report path.",
    )
    ap.add_argument(
        "--promoted-md",
        type=Path,
        default=Path("reports/baseline_report_v1_full.md"),
        help="Destination promoted MD report path.",
    )
    ap.add_argument(
        "--lock",
        type=Path,
        required=True,
        help="Frozen baseline lock path (tracked, e.g. configs/baselines/baseline_lock_v1_full.json).",
    )
    ap.add_argument(
        "--decision",
        type=Path,
        default=Path("reports/final_decision.json"),
        help="Decision snapshot JSON used in lock.",
    )
    ap.add_argument("--tag", type=str, default="v1_full_updated")
    ap.add_argument("--notes", type=str, default="Candidate promoted after passing regression gate.")
    ap.add_argument("--max-tpr-drop-abs", type=float, default=0.02)
    ap.add_argument("--max-pauc-drop-abs", type=float, default=0.02)
    ap.add_argument("--max-tpr-drop-abs-ood", type=float, default=0.005)
    ap.add_argument("--max-pauc-drop-abs-ood", type=float, default=0.01)
    ap.add_argument("--max-ece-increase-abs", type=float, default=0.01)
    ap.add_argument("--max-brier-increase-abs", type=float, default=0.01)
    ap.add_argument("--max-latency-increase-rel", type=float, default=0.30)
    ap.add_argument("--nondecreasing-eps-tpr", type=float, default=0.0)
    ap.add_argument("--nondecreasing-eps-pauc", type=float, default=0.0)
    ap.add_argument("--enforce-same-device", action="store_true")
    ap.add_argument(
        "--require-nondecreasing-scopes",
        type=str,
        default="ood_baseline",
        help="Comma-separated scopes with no allowed TPR/pAUC drop.",
    )
    args = ap.parse_args()

    if not args.current_report.exists():
        raise SystemExit(f"[error] candidate report not found: {args.current_report}")
    if args.current_md is not None and not args.current_md.exists():
        raise SystemExit(f"[error] candidate markdown not found: {args.current_md}")
    if not args.lock.exists():
        raise SystemExit(f"[error] lock not found: {args.lock}")

    scripts_dir = Path(__file__).resolve().parent
    check_script = scripts_dir / "check_baseline_regression.py"
    lock_script = scripts_dir / "create_baseline_lock.py"

    # 1) Gate
    gate_cmd = [
        sys.executable,
        str(check_script),
        "--current-report",
        str(args.current_report),
        "--lock",
        str(args.lock),
        "--max-tpr-drop-abs",
        str(args.max_tpr_drop_abs),
        "--max-pauc-drop-abs",
        str(args.max_pauc_drop_abs),
        "--max-tpr-drop-abs-ood",
        str(args.max_tpr_drop_abs_ood),
        "--max-pauc-drop-abs-ood",
        str(args.max_pauc_drop_abs_ood),
        "--max-ece-increase-abs",
        str(args.max_ece_increase_abs),
        "--max-brier-increase-abs",
        str(args.max_brier_increase_abs),
        "--max-latency-increase-rel",
        str(args.max_latency_increase_rel),
        "--nondecreasing-eps-tpr",
        str(args.nondecreasing_eps_tpr),
        "--nondecreasing-eps-pauc",
        str(args.nondecreasing_eps_pauc),
        "--require-nondecreasing-scopes",
        args.require_nondecreasing_scopes,
    ]
    if args.enforce_same_device:
        gate_cmd.append("--enforce-same-device")
    _run(gate_cmd)

    # 2) Promote candidate report(s)
    args.promoted_report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.current_report, args.promoted_report)
    print(f"[ok] promoted report -> {args.promoted_report}")

    if args.current_md is not None:
        args.promoted_md.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.current_md, args.promoted_md)
        print(f"[ok] promoted md -> {args.promoted_md}")

    # 3) Rebuild lock from promoted report
    lock_cmd = [
        sys.executable,
        str(lock_script),
        "--report",
        str(args.promoted_report),
        "--decision",
        str(args.decision),
        "--out",
        str(args.lock),
        "--tag",
        args.tag,
        "--notes",
        args.notes,
    ]
    _run(lock_cmd)
    print("[ok] gate+promote workflow completed")


if __name__ == "__main__":
    main()
