#!/usr/bin/env python3
"""Build fast-path vs verify-path decision policy from a frozen lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build serving decision policy from baseline lock.")
    ap.add_argument("--lock", type=Path, required=True, help="Path to baseline lock JSON.")
    ap.add_argument(
        "--scope",
        type=str,
        default="ood_baseline",
        choices=["id_baseline", "ood_baseline"],
        help="Scope used for threshold extraction.",
    )
    ap.add_argument("--fpr-target", type=str, default="1e-4")
    ap.add_argument(
        "--confidence-band",
        type=float,
        default=0.003,
        help="Absolute probability band around threshold for verify path.",
    )
    ap.add_argument(
        "--uncertainty-verify-threshold",
        type=float,
        default=0.10,
        help="Optional uncertainty score threshold for forcing verify path.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("configs/policy/decision_policy_mps.json"),
        help="Output decision policy JSON.",
    )
    args = ap.parse_args()

    lock = json.loads(args.lock.read_text())
    snapshot = lock.get("primary_metrics_snapshot", {})
    scope = snapshot.get(args.scope, {})
    threshold = float(scope.get("threshold_at_1e-4", float("nan")))
    if threshold != threshold:  # NaN check
        raise SystemExit(
            f"[error] lock missing threshold_at_1e-4 for scope={args.scope}. "
            "Rebuild lock from a report containing threshold metrics."
        )

    band = max(0.0, float(args.confidence_band))
    verify_min = max(0.0, threshold - band)
    fast_accept_min = min(1.0, threshold + band)

    runtime_device = str(lock.get("source_report_runtime", {}).get("device", "unknown")).lower()

    policy = {
        "schema_version": 1,
        "policy_name": f"fast_verify_{runtime_device}_{args.scope}",
        "backend": runtime_device,
        "source_lock": {
            "path": str(args.lock),
            "lock_tag": lock.get("lock_tag", ""),
            "created_utc": lock.get("created_utc", ""),
        },
        "scope": args.scope,
        "fpr_target": args.fpr_target,
        "threshold": {
            "score_at_target_fpr": threshold,
            "actual_fpr_at_target": float(scope.get("actual_fpr_at_1e-4", float("nan"))),
        },
        "bands": {
            "confidence_band_abs": band,
            "verify_min_score": verify_min,
            "fast_accept_min_score": fast_accept_min,
        },
        "routing_rules": [
            {
                "name": "fast_accept",
                "if": f"score >= {fast_accept_min:.10f} and uncertainty < {args.uncertainty_verify_threshold:.6f}",
                "action": "fast_path_accept",
            },
            {
                "name": "verify_band",
                "if": f"{verify_min:.10f} <= score < {fast_accept_min:.10f} or uncertainty >= {args.uncertainty_verify_threshold:.6f}",
                "action": "verify_path",
            },
            {"name": "reject", "if": f"score < {verify_min:.10f}", "action": "reject"},
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(f"[ok] wrote policy: {args.out}")


if __name__ == "__main__":
    main()

