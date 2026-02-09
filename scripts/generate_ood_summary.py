#!/usr/bin/env python3
"""Generate strict OOD summary markdown from reports/ood_time/*.txt."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


FPR_LINE_RE = re.compile(r"^FPR=(1e-\d+):\s+([0-9]*\.?[0-9]+)")
EVT_LINE_RE = re.compile(r"^EVT FPR=(1e-\d+).*\| TPR=([0-9]*\.?[0-9]+)")


def normalize_fpr_label(label: str) -> str:
    base, exp = label.split("e-")
    return f"{base}e-{int(exp)}"


def parse_metrics(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        m = FPR_LINE_RE.match(line)
        if m:
            fpr, v = m.groups()
            out[f"tpr_{normalize_fpr_label(fpr)}"] = float(v)
            continue
        m_evt = EVT_LINE_RE.match(line)
        if m_evt:
            fpr, v = m_evt.groups()
            out[f"evt_tpr_{normalize_fpr_label(fpr)}"] = float(v)
    return out


def mean_metrics(paths: List[Path]) -> Dict[str, float]:
    if not paths:
        return {}
    keys = ["tpr_1e-3", "tpr_1e-4", "tpr_1e-5", "evt_tpr_1e-4"]
    merged: Dict[str, float] = {}
    per_path = [parse_metrics(p) for p in paths]
    for key in keys:
        vals = [m[key] for m in per_path if key in m]
        if vals:
            merged[key] = sum(vals) / len(vals)
    return merged


def fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def get(m: Dict[str, float], key: str) -> Optional[float]:
    return m.get(key)


def row(name: str, m: Dict[str, float]) -> str:
    return (
        f"| {name} | "
        f"{fmt(get(m, 'tpr_1e-3'))} | "
        f"{fmt(get(m, 'tpr_1e-4'))} | "
        f"{fmt(get(m, 'tpr_1e-5'))} | "
        f"{fmt(get(m, 'evt_tpr_1e-4'))} |"
    )


def find_first_existing(roots: List[Path], names: List[str]) -> Optional[Path]:
    for root in roots:
        for name in names:
            p = root / name
            if p.exists():
                return p
    return None


def find_split_summary(ood_dir: Path) -> Optional[Path]:
    candidates = [
        ood_dir / "splits" / "split_summary.json",
        ood_dir / "ood_time" / "splits" / "split_summary.json",
        ood_dir.parent / "ood_time" / "splits" / "split_summary.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Generate strict OOD benchmark summary.")
    p.add_argument("--ood_dir", type=str, default="reports/ood_time")
    p.add_argument("--output_md", type=str, default="reports/ood_time/summary.md")
    p.add_argument("--swaps", type=int, default=30)
    args = p.parse_args()

    ood_dir = Path(args.ood_dir)
    output = Path(args.output_md)
    split_summary = find_split_summary(ood_dir)
    roots: List[Path] = [ood_dir]
    if ood_dir.parent != ood_dir:
        roots.append(ood_dir.parent)

    tag = f"swapped{args.swaps}"
    variants: List[Tuple[str, List[str]]] = [
        (
            "A: TF2D base",
            [
                f"ood_a_base_test_late_{tag}.txt",
                f"ood_A_swapped{args.swaps}.txt",
            ],
        ),
        (
            "B: TF2D + tail-aware",
            [
                f"ood_b_tail_test_late_{tag}.txt",
                f"ood_B_swapped{args.swaps}.txt",
            ],
        ),
        (
            "C1: TF2D + hard-negatives (boost=10)",
            [
                f"ood_c_tail_hn_test_late_{tag}.txt",
                f"ood_C1_swapped{args.swaps}.txt",
            ],
        ),
        (
            "C1 + temperature calibration",
            [
                f"ood_c_tail_hn_cal_temp_test_late_{tag}.txt",
                f"ood_C1_swapped{args.swaps}_cal_temp.txt",
            ],
        ),
        (
            "C2: TF2D + hard-negatives (boost=3)",
            [f"ood_C2_swapped{args.swaps}.txt"],
        ),
        (
            "C2 + temperature calibration",
            [f"ood_C2_swapped{args.swaps}_cal_temp.txt"],
        ),
        (
            "C3: TF2D + hard-negatives (boost=5)",
            [f"ood_C3_swapped{args.swaps}.txt"],
        ),
        (
            "C3 + temperature calibration",
            [f"ood_C3_swapped{args.swaps}_cal_temp.txt"],
        ),
    ]
    rows: List[Tuple[str, Dict[str, float]]] = []
    for name, names in variants:
        path = find_first_existing(roots, names)
        if path is None:
            continue
        rows.append((name, parse_metrics(path)))

    # Optional strict-OOD C3 ensemble summary.
    ood_ensemble_roots = [
        ood_dir / "ood_ensemble_c3",
        ood_dir.parent / "ood_ensemble_c3",
        Path("reports/ood_ensemble_c3"),
    ]
    ood_ensemble_dir = next((p for p in ood_ensemble_roots if p.exists()), None)
    if ood_ensemble_dir is not None:
        seed_files = sorted(ood_ensemble_dir.glob("c3_seed*_ood_swapped30.txt"))
        ens_file = ood_ensemble_dir / "c3_ensemble5_ood_swapped30.txt"
        if seed_files:
            rows.append(("C3 (5-seed mean, independent)", mean_metrics(seed_files)))
        if ens_file.exists():
            rows.append(("C3 ensemble (5 checkpoints)", parse_metrics(ens_file)))

    lines = []
    lines.append("# Strict OOD Benchmark (Train-Early / Test-Late)")
    lines.append("")
    if split_summary and split_summary.exists():
        summary = json.loads(split_summary.read_text())
        mode = summary.get("mode", "unknown")
        counts = summary.get("counts", {})
        lines.append(f"Split mode: `{mode}`")
        lines.append(
            "Counts: "
            f"noise_train={counts.get('noise_train_early', 'n/a')}, "
            f"noise_test={counts.get('noise_test_late', 'n/a')}, "
            f"signal_train={counts.get('signal_train_early', 'n/a')}, "
            f"signal_test={counts.get('signal_test_late', 'n/a')}"
        )
        lines.append("")

    lines.append(f"Protocol: `evaluate_background --method swapped_pairs --swaps {args.swaps}` on `test_late` split.")
    lines.append("")
    lines.append("| Variant | TPR@1e-3 | TPR@1e-4 | TPR@1e-5 | EVT TPR@1e-4 |")
    lines.append("|---|---:|---:|---:|---:|")
    for variant_name, metrics in rows:
        lines.append(row(variant_name, metrics))
    lines.append("")
    lines.append("Primary KPI: `TPR@1e-4`")
    lines.append("Secondary KPI: `TPR@1e-5`")
    lines.append("")
    lines.append("Interpretation rule (primary KPI):")
    lines.append("- Prefer variant with highest `TPR@1e-4` on strict OOD.")
    lines.append("- Use calibration primarily for probability quality and deeper-tail operation.")
    lines.append("")

    output.write_text("\n".join(lines))
    print(f"[OK] wrote {output}")


if __name__ == "__main__":
    main()
