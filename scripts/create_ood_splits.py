#!/usr/bin/env python3
"""Create strict OOD train/test splits (time-ordered or run-based)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_ids(path: Path) -> List[str]:
    data = json.loads(path.read_text())
    return [str(x) for x in data]


def numeric_or_lex_key(v: str):
    return (0, int(v)) if v.isdigit() else (1, v)


def split_by_time(ids: List[str], train_frac: float, gap_frac: float) -> Tuple[List[str], List[str]]:
    if not ids:
        return [], []
    ordered = sorted(ids, key=numeric_or_lex_key)
    n = len(ordered)
    n_train = int(n * train_frac)
    n_gap = int(n * gap_frac)
    start_test = min(max(n_train + n_gap, 0), n)
    train_ids = ordered[:n_train]
    test_ids = ordered[start_test:]
    return train_ids, test_ids


def parse_run_map(path: Path) -> Dict[str, str]:
    payload = json.loads(path.read_text())
    mapping: Dict[str, str] = {}
    if isinstance(payload, dict):
        for k, v in payload.items():
            mapping[str(k)] = str(v)
    elif isinstance(payload, list):
        for rec in payload:
            if isinstance(rec, dict) and "id" in rec and "run" in rec:
                mapping[str(rec["id"])] = str(rec["run"])
    else:
        raise ValueError("Unsupported run_map format. Use dict[id]=run or list[{id,run}].")
    return mapping


def split_by_run(ids: List[str], run_map: Dict[str, str], train_runs: Iterable[str], test_runs: Iterable[str]) -> Tuple[List[str], List[str]]:
    train_set = set(train_runs)
    test_set = set(test_runs)
    train_ids: List[str] = []
    test_ids: List[str] = []
    for sid in ids:
        run = run_map.get(str(sid))
        if run in train_set:
            train_ids.append(str(sid))
        elif run in test_set:
            test_ids.append(str(sid))
    train_ids.sort(key=numeric_or_lex_key)
    test_ids.sort(key=numeric_or_lex_key)
    return train_ids, test_ids


def save_json(path: Path, data: List[str]) -> None:
    path.write_text(json.dumps(data, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Create strict OOD splits for GW detection experiments.")
    p.add_argument("--noise_indices", type=str, required=True)
    p.add_argument("--signal_indices", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--mode", type=str, default="time", choices=["time", "run"])
    p.add_argument("--train_frac", type=float, default=0.7, help="Used only in mode=time")
    p.add_argument("--gap_frac", type=float, default=0.05, help="Used only in mode=time")
    p.add_argument("--min_per_class", type=int, default=200)

    p.add_argument("--run_map", type=str, default=None, help="JSON map id->run or list[{id,run}]")
    p.add_argument("--train_runs", type=str, default="", help="Comma-separated run names for training (mode=run)")
    p.add_argument("--test_runs", type=str, default="", help="Comma-separated run names for testing (mode=run)")

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    noise_ids = load_ids(Path(args.noise_indices))
    signal_ids = load_ids(Path(args.signal_indices))

    if args.mode == "time":
        n_train, n_test = split_by_time(noise_ids, args.train_frac, args.gap_frac)
        s_train, s_test = split_by_time(signal_ids, args.train_frac, args.gap_frac)
        split_meta = {
            "mode": "time",
            "train_frac": args.train_frac,
            "gap_frac": args.gap_frac,
        }
    else:
        if not args.run_map:
            raise SystemExit("--run_map is required for mode=run")
        run_map = parse_run_map(Path(args.run_map))
        train_runs = [x.strip() for x in args.train_runs.split(",") if x.strip()]
        test_runs = [x.strip() for x in args.test_runs.split(",") if x.strip()]
        if not train_runs or not test_runs:
            raise SystemExit("Both --train_runs and --test_runs must be non-empty for mode=run")
        n_train, n_test = split_by_run(noise_ids, run_map, train_runs, test_runs)
        s_train, s_test = split_by_run(signal_ids, run_map, train_runs, test_runs)
        split_meta = {
            "mode": "run",
            "train_runs": train_runs,
            "test_runs": test_runs,
            "run_map": args.run_map,
        }

    if len(n_train) < args.min_per_class or len(n_test) < args.min_per_class:
        raise SystemExit(
            f"Too few noise samples after split: train={len(n_train)}, test={len(n_test)}, min={args.min_per_class}"
        )
    if len(s_train) < args.min_per_class or len(s_test) < args.min_per_class:
        raise SystemExit(
            f"Too few signal samples after split: train={len(s_train)}, test={len(s_test)}, min={args.min_per_class}"
        )

    # Save split files
    noise_train_path = out_dir / "indices_noise_train_early.json"
    noise_test_path = out_dir / "indices_noise_test_late.json"
    signal_train_path = out_dir / "indices_signal_train_early.json"
    signal_test_path = out_dir / "indices_signal_test_late.json"

    save_json(noise_train_path, n_train)
    save_json(noise_test_path, n_test)
    save_json(signal_train_path, s_train)
    save_json(signal_test_path, s_test)

    overlap_noise = sorted(set(n_train) & set(n_test))
    overlap_signal = sorted(set(s_train) & set(s_test))
    if overlap_noise or overlap_signal:
        raise SystemExit("Split overlap detected between train and test IDs.")

    summary = {
        **split_meta,
        "counts": {
            "noise_train_early": len(n_train),
            "noise_test_late": len(n_test),
            "signal_train_early": len(s_train),
            "signal_test_late": len(s_test),
        },
        "files": {
            "noise_train_early": str(noise_train_path),
            "noise_test_late": str(noise_test_path),
            "signal_train_early": str(signal_train_path),
            "signal_test_late": str(signal_test_path),
        },
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))

    print("[OOD split] created")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
