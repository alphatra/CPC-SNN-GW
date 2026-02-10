#!/usr/bin/env python3
"""Bootstrap comparison for B vs C3 ensemble OOD repeats."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


PAT = re.compile(r"(?:FPR=1e-0?4:\s+([0-9.]+)|TPR\s*@\s*1e-4:\s*([0-9.]+))")


def _collect_scores(out_dir: Path, prefix: str) -> dict[int, float]:
    scores: dict[int, float] = {}
    for p in sorted(out_dir.glob(f"{prefix}_r*.txt")):
        m_idx = re.search(r"_r(\d+)\.txt$", p.name)
        if not m_idx:
            continue
        idx = int(m_idx.group(1))
        txt = p.read_text(encoding="utf-8", errors="ignore")
        m = PAT.search(txt)
        if not m:
            continue
        scores[idx] = float(m.group(1) or m.group(2))
    return scores


def _bootstrap_ci(delta: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(delta)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(delta[idx].mean())
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    p_superior = float((means > 0.0).mean())
    return lo, hi, p_superior


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap compare B vs C3 ensemble from repeat logs.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--min-delta", type=float, default=0.0, help="Required mean delta (C3-B).")
    ap.add_argument(
        "--min-p-superiority",
        type=float,
        default=0.95,
        help="Required bootstrap probability that delta > 0.",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional output JSON path. Default: <out-dir>/bootstrap_compare_summary.json",
    )
    args = ap.parse_args()

    out_dir = args.out_dir
    b = _collect_scores(out_dir, "b")
    c3 = _collect_scores(out_dir, "c3ens")
    common = sorted(set(b).intersection(c3))
    if not common:
        raise SystemExit(
            f"[error] no paired repeats found in {out_dir} (need b_r*.txt and c3ens_r*.txt)."
        )

    b_arr = np.array([b[i] for i in common], dtype=np.float64)
    c_arr = np.array([c3[i] for i in common], dtype=np.float64)
    delta = c_arr - b_arr

    mu_b = float(b_arr.mean())
    sd_b = float(b_arr.std())
    mu_c = float(c_arr.mean())
    sd_c = float(c_arr.std())
    mu_d = float(delta.mean())
    sd_d = float(delta.std())
    lo, hi, p_sup = _bootstrap_ci(delta, n_boot=args.n_boot, seed=args.seed)

    summary = {
        "repeats": len(common),
        "paired_repeat_ids": common,
        "b": {"mean_tpr_at_1e4": mu_b, "std_tpr_at_1e4": sd_b},
        "c3_ensemble": {"mean_tpr_at_1e4": mu_c, "std_tpr_at_1e4": sd_c},
        "delta_c3_minus_b": {
            "mean": mu_d,
            "std": sd_d,
            "ci95_low": lo,
            "ci95_high": hi,
            "p_superiority": p_sup,
        },
        "gate": {
            "min_delta": float(args.min_delta),
            "min_p_superiority": float(args.min_p_superiority),
            "pass": bool(mu_d >= args.min_delta and p_sup >= args.min_p_superiority),
        },
    }

    out_json = args.out_json or (out_dir / "bootstrap_compare_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[B] repeats={len(common)} TPR@1e-4={mu_b:.4f}±{sd_b:.4f}")
    print(f"[C3-ensemble] repeats={len(common)} TPR@1e-4={mu_c:.4f}±{sd_c:.4f}")
    print(f"[delta ensemble-B] {mu_d:+.4f} (95% CI [{lo:+.4f}, {hi:+.4f}], p_sup={p_sup:.4f})")
    print(f"[ok] wrote bootstrap summary: {out_json}")

    if not summary["gate"]["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

