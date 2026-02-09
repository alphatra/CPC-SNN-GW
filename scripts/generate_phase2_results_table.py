#!/usr/bin/env python3
"""Generate final Phase-2 benchmark table from evaluation report text files."""

from __future__ import annotations

import argparse
import datetime as dt
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


FPR_LINE_RE = re.compile(r"^FPR=(1e-\d+):\s+([0-9]*\.?[0-9]+)")
EVT_LINE_RE = re.compile(r"^EVT FPR=(1e-\d+).*\| TPR=([0-9]*\.?[0-9]+)")


def normalize_fpr_label(label: str) -> str:
    """Normalize labels like 1e-03 -> 1e-3 for stable key lookup."""
    base, exp = label.split("e-")
    return f"{base}e-{int(exp)}"


def parse_report_metrics(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        m = FPR_LINE_RE.match(line)
        if m:
            fpr, value = m.groups()
            fpr = normalize_fpr_label(fpr)
            metrics[f"tpr_{fpr}"] = float(value)
            continue

        m_evt = EVT_LINE_RE.match(line)
        if m_evt:
            fpr, value = m_evt.groups()
            fpr = normalize_fpr_label(fpr)
            metrics[f"evt_tpr_{fpr}"] = float(value)
    return metrics


def list_existing(paths: Iterable[Path]) -> List[Path]:
    return [p for p in paths if p.exists()]


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def report_has_metrics(path: Path) -> bool:
    m = parse_report_metrics(path)
    return any(k.startswith("tpr_") for k in m.keys())


def resolve_variant_files(reports_dir: Path, pattern: str, fallback: Optional[str] = None) -> List[Path]:
    files = sorted(reports_dir.glob(pattern))
    valid = [p for p in files if report_has_metrics(p)]
    if valid:
        return valid
    if fallback:
        fb = reports_dir / fallback
        if fb.exists() and report_has_metrics(fb):
            return [fb]
    return []


def mean_std(values: List[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def metric_cell(files: List[Path], metric_key: str) -> str:
    vals: List[float] = []
    for path in files:
        m = parse_report_metrics(path)
        if metric_key in m and not math.isnan(m[metric_key]):
            vals.append(m[metric_key])
    if not vals:
        return "n/a"
    mu, sd = mean_std(vals)
    return f"{mu:.4f} ± {sd:.4f}"


def build_row(name: str, files: List[Path]) -> str:
    n = len(files)
    return (
        f"| {name} | {n} | "
        f"{metric_cell(files, 'tpr_1e-3')} | "
        f"{metric_cell(files, 'tpr_1e-4')} | "
        f"{metric_cell(files, 'tpr_1e-5')} | "
        f"{metric_cell(files, 'evt_tpr_1e-4')} |"
    )


def single_metric(path: Path, key: str) -> Optional[float]:
    if not path.exists():
        return None
    metrics = parse_report_metrics(path)
    return metrics.get(key)


def fmt_optional(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final benchmark markdown table from reports/*.txt")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Directory with reports")
    parser.add_argument("--output_md", type=str, default="reports/final_benchmark_table.md")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    output_md = Path(args.output_md)
    repeats_dir = reports_dir / "repeats"

    a_files = resolve_variant_files(reports_dir, "repeats/a_base_r*.txt", "bg_phase2_tf2d_base_swapped30.txt")
    b_files = resolve_variant_files(reports_dir, "repeats/b_tail_r*.txt", "bg_phase2_tf2d_tail_swapped30.txt")
    c_files = resolve_variant_files(reports_dir, "repeats/c_tail_hn_r*.txt", "bg_phase2_tf2d_tail_hn_v1_swapped30.txt")
    c_cal_files = resolve_variant_files(
        reports_dir,
        "repeats/c_tail_hn_caltemp_r*.txt",
        "bg_phase2_tf2d_tail_hn_v1_swapped30_cal_temp.txt",
    )
    c2_files = resolve_variant_files(
        reports_dir,
        "repeats/c2_tail_hn_soft_r*.txt",
        "bg_phase2_tf2d_tail_hn_v2_soft_swapped30.txt",
    )
    c2_cal_files = resolve_variant_files(
        reports_dir,
        "repeats/c2_tail_hn_soft_caltemp_r*.txt",
        "bg_phase2_tf2d_tail_hn_v2_soft_swapped30_cal_temp.txt",
    )
    c3_files = resolve_variant_files(
        reports_dir,
        "repeats/c3_tail_hn_boost5_r*.txt",
        "bg_phase2_tf2d_tail_hn_v3_boost5_swapped30.txt",
    )
    c3_cal_files = resolve_variant_files(
        reports_dir,
        "repeats/c3_tail_hn_boost5_caltemp_r*.txt",
        "bg_phase2_tf2d_tail_hn_v3_boost5_swapped30_cal_temp.txt",
    )

    now = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%SZ")

    nohn = reports_dir / "bg_phase2_tf2d_tail_hn_v1_swapped30_nohn.txt"
    nohn_cal = reports_dir / "bg_phase2_tf2d_tail_hn_v1_swapped30_nohn_cal_temp.txt"

    nohn_tpr_1e4 = single_metric(nohn, "tpr_1e-4")
    nohn_tpr_1e5 = single_metric(nohn, "tpr_1e-5")
    nohn_cal_tpr_1e4 = single_metric(nohn_cal, "tpr_1e-4")
    nohn_cal_tpr_1e5 = single_metric(nohn_cal, "tpr_1e-5")

    lines: List[str] = []
    lines.append("# Final Benchmark Table (Phase 2)")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append("Protocol: `evaluate_background --method swapped_pairs --swaps 30`")
    lines.append("")
    lines.append("## Main Comparison (A/B/C/C+calib/C2/C2+calib/C3/C3+calib)")
    lines.append("")
    lines.append("| Variant | n | TPR@1e-3 | TPR@1e-4 | TPR@1e-5 | EVT TPR@1e-4 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(build_row("A: TF2D base", a_files))
    lines.append(build_row("B: TF2D + tail-aware", b_files))
    lines.append(build_row("C: TF2D + tail-aware + hard-negatives", c_files))
    lines.append(build_row("C + temperature calibration", c_cal_files))
    lines.append(build_row("C2 (ablation): TF2D + tail-aware + hard-negatives (soft mining)", c2_files))
    lines.append(build_row("C2 + temperature calibration", c2_cal_files))
    lines.append(build_row("C3 (ablation): TF2D + tail-aware + hard-negatives (boost=5)", c3_files))
    lines.append(build_row("C3 + temperature calibration", c3_cal_files))
    lines.append("")
    lines.append("Notes:")
    lines.append("- `n` is the number of report files found for each variant.")
    lines.append(f"- Repeats directory: `{repeats_dir}`")

    if nohn.exists() or nohn_cal.exists():
        lines.append("")
        lines.append("## NOHN Check (Generalization Control)")
        lines.append("")
        lines.append("| Variant | TPR@1e-4 | TPR@1e-5 |")
        lines.append("|---|---:|---:|")
        if nohn.exists():
            lines.append(f"| C (NOHN) | {fmt_optional(nohn_tpr_1e4)} | {fmt_optional(nohn_tpr_1e5)} |")
        if nohn_cal.exists():
            lines.append(
                f"| C + temperature (NOHN) | {fmt_optional(nohn_cal_tpr_1e4)} | {fmt_optional(nohn_cal_tpr_1e5)} |"
            )

    lines.append("")
    lines.append("## KPI Policy")
    lines.append("")
    lines.append("- **Primary KPI**: `TPR@1e-4`")
    lines.append("- **Secondary KPI**: `TPR@1e-5`")
    lines.append("- Operationally:")
    lines.append("  - near `FAR=1e-4` prefer uncalibrated scores,")
    lines.append("  - near `FAR=1e-5` prefer temperature-calibrated scores.")
    lines.append("")

    # Optional strict-OOD ensemble comparison section.
    ood_ens_dir = reports_dir / "ood_ensemble_c3"
    c3_ood_seed_files = sorted(ood_ens_dir.glob("c3_seed*_ood_swapped30.txt"))
    c3_ood_ensemble_file = ood_ens_dir / "c3_ensemble5_ood_swapped30.txt"
    b_ood_file = first_existing(
        [
            reports_dir / "ood_time" / "ood_b_tail_test_late_swapped30.txt",
            reports_dir / "ood_B_swapped30.txt",
        ]
    )
    if c3_ood_seed_files or c3_ood_ensemble_file.exists() or b_ood_file is not None:
        lines.append("## Strict OOD Ensemble Check (Train-Early / Test-Late)")
        lines.append("")
        lines.append("Protocol: `evaluate_background --method swapped_pairs --swaps 30` on `test_late` split.")
        lines.append("")
        lines.append("| Variant | n | TPR@1e-3 | TPR@1e-4 | TPR@1e-5 | EVT TPR@1e-4 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        if b_ood_file is not None:
            lines.append(build_row("B: TF2D + tail-aware (single)", [b_ood_file]))
        if c3_ood_seed_files:
            lines.append(build_row("C3 single checkpoints (seed mean)", c3_ood_seed_files))
        if c3_ood_ensemble_file.exists():
            lines.append(build_row("C3 ensemble (5 checkpoints, score-avg)", [c3_ood_ensemble_file]))
        lines.append("")

    output_md.write_text("\n".join(lines))
    print(f"[OK] wrote {output_md}")


if __name__ == "__main__":
    main()
