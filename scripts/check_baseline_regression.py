#!/usr/bin/env python3
"""Regression gate checker against a frozen baseline lock.

Compares a current baseline report against lock snapshot and fails on
metric degradation beyond allowed tolerances.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _get(d: Dict[str, Any], path: List[str], default: float = float("nan")) -> float:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    try:
        return float(cur)
    except Exception:
        return default


def check_scope(
    scope_name: str,
    cur: Dict[str, Any],
    base: Dict[str, Any],
    max_tpr_drop_abs: float,
    max_pauc_drop_abs: float,
    max_ece_increase_abs: float,
    max_brier_increase_abs: float,
    max_latency_increase_rel: float,
) -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    ok = True

    c_tpr = _get(cur, [scope_name, "metrics", "tpr_at_1e-4"])
    b_tpr = _get(base, [scope_name, "tpr_at_1e-4"])
    if not math.isnan(c_tpr) and not math.isnan(b_tpr):
        if c_tpr < b_tpr - max_tpr_drop_abs:
            ok = False
            msgs.append(
                f"[FAIL] {scope_name}: TPR@1e-4 dropped {b_tpr:.6f} -> {c_tpr:.6f} "
                f"(allowed drop {max_tpr_drop_abs:.6f})"
            )
        else:
            msgs.append(f"[OK]   {scope_name}: TPR@1e-4 {c_tpr:.6f} (baseline {b_tpr:.6f})")

    c_pauc = _get(cur, [scope_name, "metrics", "pauc_norm_max_fpr_1e-4"])
    b_pauc = _get(base, [scope_name, "pauc_norm_max_fpr_1e-4"])
    if not math.isnan(c_pauc) and not math.isnan(b_pauc):
        if c_pauc < b_pauc - max_pauc_drop_abs:
            ok = False
            msgs.append(
                f"[FAIL] {scope_name}: pAUC dropped {b_pauc:.6f} -> {c_pauc:.6f} "
                f"(allowed drop {max_pauc_drop_abs:.6f})"
            )
        else:
            msgs.append(f"[OK]   {scope_name}: pAUC {c_pauc:.6f} (baseline {b_pauc:.6f})")

    c_ece = _get(cur, [scope_name, "metrics", "ece_15bins"])
    b_ece = _get(base, [scope_name, "ece_15bins"])
    if not math.isnan(c_ece) and not math.isnan(b_ece):
        if c_ece > b_ece + max_ece_increase_abs:
            ok = False
            msgs.append(
                f"[FAIL] {scope_name}: ECE increased {b_ece:.6f} -> {c_ece:.6f} "
                f"(allowed increase {max_ece_increase_abs:.6f})"
            )
        else:
            msgs.append(f"[OK]   {scope_name}: ECE {c_ece:.6f} (baseline {b_ece:.6f})")

    c_brier = _get(cur, [scope_name, "metrics", "brier"])
    b_brier = _get(base, [scope_name, "brier"])
    if not math.isnan(c_brier) and not math.isnan(b_brier):
        if c_brier > b_brier + max_brier_increase_abs:
            ok = False
            msgs.append(
                f"[FAIL] {scope_name}: Brier increased {b_brier:.6f} -> {c_brier:.6f} "
                f"(allowed increase {max_brier_increase_abs:.6f})"
            )
        else:
            msgs.append(f"[OK]   {scope_name}: Brier {c_brier:.6f} (baseline {b_brier:.6f})")

    c_lat = _get(cur, [scope_name, "latency", "sample_latency_ms_p95"])
    b_lat = _get(base, [scope_name, "sample_latency_ms_p95"])
    if not math.isnan(c_lat) and not math.isnan(b_lat):
        limit = b_lat * (1.0 + max_latency_increase_rel)
        if c_lat > limit:
            ok = False
            msgs.append(
                f"[FAIL] {scope_name}: latency p95 increased {b_lat:.4f} -> {c_lat:.4f} ms "
                f"(limit {limit:.4f} ms)"
            )
        else:
            msgs.append(
                f"[OK]   {scope_name}: latency p95 {c_lat:.4f} ms "
                f"(baseline {b_lat:.4f} ms, limit {limit:.4f} ms)"
            )

    return ok, msgs


def _resolve_scope_threshold(
    scope: str,
    base_value: float,
    id_value: float | None,
    ood_value: float | None,
) -> float:
    if scope == "id_baseline" and id_value is not None:
        return float(id_value)
    if scope == "ood_baseline" and ood_value is not None:
        return float(ood_value)
    return float(base_value)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check baseline regression against lock.")
    ap.add_argument(
        "--current-report",
        type=Path,
        default=Path("reports/baseline_report_v1_full.json"),
        help=(
            "Current baseline report to validate. "
            "Default: reports/baseline_report_v1_full.json"
        ),
    )
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--max-tpr-drop-abs", type=float, default=0.02)
    ap.add_argument("--max-pauc-drop-abs", type=float, default=0.02)
    ap.add_argument("--max-ece-increase-abs", type=float, default=0.01)
    ap.add_argument("--max-brier-increase-abs", type=float, default=0.01)
    ap.add_argument("--max-latency-increase-rel", type=float, default=0.30)
    ap.add_argument("--max-tpr-drop-abs-id", type=float, default=None)
    ap.add_argument("--max-tpr-drop-abs-ood", type=float, default=None)
    ap.add_argument("--max-pauc-drop-abs-id", type=float, default=None)
    ap.add_argument("--max-pauc-drop-abs-ood", type=float, default=None)
    ap.add_argument("--require-nondecreasing-scopes", type=str, default="")
    ap.add_argument(
        "--nondecreasing-eps-tpr",
        type=float,
        default=0.0,
        help="Allowed tiny TPR drop for scopes in --require-nondecreasing-scopes.",
    )
    ap.add_argument(
        "--nondecreasing-eps-pauc",
        type=float,
        default=0.0,
        help="Allowed tiny pAUC drop for scopes in --require-nondecreasing-scopes.",
    )
    ap.add_argument(
        "--enforce-same-device",
        action="store_true",
        help="Fail if current report runtime.device differs from lock runtime device.",
    )
    args = ap.parse_args()

    if not args.current_report.exists():
        suggestions = [
            Path("reports/baseline_report_v1_full.json"),
            Path("reports/baseline_report_v0.json"),
            Path("reports/baseline_report_v0_frozen_now.json"),
            Path("reports/baseline_report_candidate.json"),
        ]
        existing = [str(p) for p in suggestions if p.exists()]
        hint = f" Existing candidates: {existing}" if existing else ""
        raise SystemExit(
            f"[error] current report not found: {args.current_report}.{hint}\n"
            "Tip: pass --current-report reports/baseline_report_v1_full.json"
        )
    if not args.lock.exists():
        raise SystemExit(f"[error] lock file not found: {args.lock}")

    current = json.loads(args.current_report.read_text())
    lock = json.loads(args.lock.read_text())
    baseline = lock.get("primary_metrics_snapshot", {})

    all_ok = True
    lines: List[str] = []
    lines.append(f"[info] current report: {args.current_report}")
    lines.append(f"[info] lock: {args.lock}")
    lines.append(f"[info] lock tag: {lock.get('lock_tag', '')}")
    cur_dev = str(current.get("runtime", {}).get("device", "")).lower()
    lock_dev = str(lock.get("source_report_runtime", {}).get("device", "")).lower()
    if cur_dev:
        lines.append(f"[info] current runtime.device: {cur_dev}")
    if lock_dev:
        lines.append(f"[info] lock runtime.device: {lock_dev}")
    if args.enforce_same_device and cur_dev and lock_dev and cur_dev != lock_dev:
        lines.append(
            f"[FAIL] runtime.device mismatch: current={cur_dev} vs lock={lock_dev} "
            "(use backend-specific lock)"
        )
        all_ok = False

    scopes_no_drop = {
        s.strip() for s in args.require_nondecreasing_scopes.split(",") if s.strip()
    }

    for scope in ("id_baseline", "ood_baseline"):
        tpr_drop = _resolve_scope_threshold(
            scope,
            args.max_tpr_drop_abs,
            args.max_tpr_drop_abs_id,
            args.max_tpr_drop_abs_ood,
        )
        pauc_drop = _resolve_scope_threshold(
            scope,
            args.max_pauc_drop_abs,
            args.max_pauc_drop_abs_id,
            args.max_pauc_drop_abs_ood,
        )
        if scope in scopes_no_drop:
            tpr_drop = max(0.0, float(args.nondecreasing_eps_tpr))
            pauc_drop = max(0.0, float(args.nondecreasing_eps_pauc))
        ok, msgs = check_scope(
            scope_name=scope,
            cur=current,
            base=baseline,
            max_tpr_drop_abs=tpr_drop,
            max_pauc_drop_abs=pauc_drop,
            max_ece_increase_abs=args.max_ece_increase_abs,
            max_brier_increase_abs=args.max_brier_increase_abs,
            max_latency_increase_rel=args.max_latency_increase_rel,
        )
        all_ok = all_ok and ok
        lines.extend(msgs)

    for l in lines:
        print(l)

    if not all_ok:
        raise SystemExit(2)
    print("[ok] regression check passed")


if __name__ == "__main__":
    main()
