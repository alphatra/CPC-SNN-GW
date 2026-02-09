from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_handling.torch_dataset import HDF5SFTPairDataset
from src.evaluation.calibration import apply_calibration_to_logits, load_calibration
from src.evaluation.model_loader import load_cpcsnn_from_checkpoint
from src.inference.trigger_generator import generate_triggers


def _load_ids(h5_path: str, ids_json: str | None) -> List[str]:
    if ids_json:
        with open(ids_json, "r") as f:
            return [str(x) for x in json.load(f)]
    with h5py.File(h5_path, "r") as h5:
        return sorted([k for k in h5.keys() if k.isdigit()])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict p(signal) for HDF5 windows.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--h5_path", required=True, type=str)
    parser.add_argument("--ids_json", type=str, default=None, help="Optional list of sample ids")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min_gap", type=int, default=0, help="Trigger suppression in windows")
    parser.add_argument("--output_csv", type=str, default="predictions.csv")
    parser.add_argument("--output_triggers_json", type=str, default="triggers.json")
    parser.add_argument("--use_metal", action="store_true", default=False)
    parser.add_argument("--calibration_json", type=str, default=None, help="Optional calibration artifact JSON")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, _, model_kwargs, _ = load_cpcsnn_from_checkpoint(
        args.checkpoint, device, use_metal=args.use_metal
    )
    if model_kwargs["use_tf2d"]:
        raise RuntimeError(
            "predict.py currently supports 1D/time-series checkpoints only. "
            "Use evaluate_background.py for TF2D checkpoints."
        )

    calibrator = None
    if args.calibration_json:
        calibrator = load_calibration(args.calibration_json)
        print(f"Loaded calibration: {args.calibration_json} (method={calibrator.method})")

    ids = _load_ids(args.h5_path, args.ids_json)
    dataset = HDF5SFTPairDataset(
        h5_path=args.h5_path,
        index_list=ids,
        return_time_series=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    predictions: List[dict] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            # Consistent normalization with training/eval scripts.
            x = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-8)
            logits = model(x)[0].squeeze(1).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            if calibrator is not None:
                probs = apply_calibration_to_logits(logits, calibrator)
            ids_batch = batch.get("id", [])
            for i, p in enumerate(probs):
                sample_id = str(ids_batch[i]) if i < len(ids_batch) else ""
                predictions.append(
                    {
                        "id": sample_id,
                        "prob_signal": float(p),
                        "pred_label": int(float(p) >= args.threshold),
                    }
                )

    pred_path = Path(args.output_csv)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prob_signal", "pred_label"])
        writer.writeheader()
        writer.writerows(predictions)

    triggers = generate_triggers(predictions, threshold=args.threshold, min_gap=args.min_gap)
    trig_path = Path(args.output_triggers_json)
    trig_path.parent.mkdir(parents=True, exist_ok=True)
    with trig_path.open("w") as f:
        json.dump(triggers, f, indent=2)

    probs = np.array([p["prob_signal"] for p in predictions], dtype=np.float32)
    print(f"Scored windows: {len(predictions)}")
    print(f"Mean prob: {probs.mean():.6f} | Max prob: {probs.max():.6f}")
    print(f"Predictions CSV: {pred_path}")
    print(f"Triggers JSON: {trig_path} (count={len(triggers)})")


if __name__ == "__main__":
    main()
