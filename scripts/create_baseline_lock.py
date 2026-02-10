#!/usr/bin/env python3
"""Create immutable baseline lock artifact from a baseline report.

Example:
  PYTHONPATH=. python scripts/create_baseline_lock.py \
    --report reports/baseline_report_v0.json \
    --decision reports/final_decision.json \
    --out reports/baseline_lock_v0.json \
    --tag v0_fast
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, List


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def maybe_hash(path_str: str) -> Dict[str, Any]:
    p = Path(path_str)
    if p.exists() and p.is_file():
        return {
            "path": str(p),
            "exists": True,
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
    return {"path": str(p), "exists": False}


def extract_primary_metrics(report: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for scope in ("id_baseline", "ood_baseline"):
        m = report.get(scope, {}).get("metrics", {})
        lat = report.get(scope, {}).get("latency", {})
        out[scope] = {
            "tpr_at_1e-4": float(m.get("tpr_at_1e-4", float("nan"))),
            "threshold_at_1e-4": float(m.get("threshold_at_1e-4", float("nan"))),
            "actual_fpr_at_1e-4": float(m.get("actual_fpr_at_1e-4", float("nan"))),
            "pauc_norm_max_fpr_1e-4": float(m.get("pauc_norm_max_fpr_1e-4", float("nan"))),
            "ece_15bins": float(m.get("ece_15bins", float("nan"))),
            "brier": float(m.get("brier", float("nan"))),
            "sample_latency_ms_p95": float(lat.get("sample_latency_ms_p95", float("nan"))),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Create immutable baseline lock artifact.")
    ap.add_argument("--report", type=Path, required=True, help="Baseline report JSON path.")
    ap.add_argument("--decision", type=Path, default=Path("reports/final_decision.json"))
    ap.add_argument("--out", type=Path, required=True, help="Output lock JSON path.")
    ap.add_argument("--tag", type=str, default="baseline_lock")
    ap.add_argument("--notes", type=str, default="")
    ap.add_argument("--command", type=str, default="", help="CLI command used to produce baseline report.")
    args = ap.parse_args()

    report = json.loads(args.report.read_text())
    decision = json.loads(args.decision.read_text()) if args.decision.exists() else {}

    # Collect checkpoint paths from report first (more reliable than decision drift).
    ckpts: List[str] = []
    ckpts.extend(report.get("id_baseline", {}).get("checkpoints", []))
    ckpts.extend(report.get("ood_baseline", {}).get("checkpoints", []))
    ckpts = list(dict.fromkeys(ckpts))

    lock = {
        "schema_version": 1,
        "lock_tag": args.tag,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "user": os.environ.get("USER", ""),
        "source_report": maybe_hash(str(args.report)),
        "source_decision": maybe_hash(str(args.decision)),
        "source_report_runtime": report.get("runtime", {}),
        "source_report_protocol_tag": report.get("protocol_tag", ""),
        "source_report_task": report.get("task", ""),
        "primary_kpi": report.get("primary_kpi", "TPR@1e-4"),
        "primary_metrics_snapshot": extract_primary_metrics(report),
        "checkpoints": [maybe_hash(c) for c in ckpts],
        "decision_snapshot": decision,
        "reproduce_command": args.command,
        "notes": args.notes,
    }

    args.out.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"[ok] wrote lock: {args.out}")


if __name__ == "__main__":
    main()
