#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_handling.torch_dataset import HDF5SFTPairDataset
from src.evaluation.calibration import apply_calibration_to_logits, load_calibration
from src.evaluation.model_loader import load_cpcsnn_from_checkpoint


def _infer_tf_layout(in_channels: int):
    if in_channels in (3, 4):
        return ["H1"], in_channels
    if in_channels in (6, 8):
        return ["H1", "L1"], in_channels // 2
    if in_channels % 2 == 0:
        return ["H1", "L1"], in_channels // 2
    return ["H1"], in_channels


def _prepare_tf2d_input(batch, device, in_channels: int):
    ifos, per_ifo = _infer_tf_layout(in_channels)
    feats = []
    for ifo in ifos:
        d = batch[ifo].to(device)
        if d.shape[1] < per_ifo:
            raise RuntimeError(
                f"Input channel mismatch for {ifo}: have {d.shape[1]}, need {per_ifo}."
            )
        feats.append(d[:, :per_ifo, :, :])
    x = torch.cat(feats, dim=1)
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True)
    return (x - mean) / (std + 1e-8)


def _prepare_1d_input(batch, device, in_channels: int):
    x = batch["x"].to(device)
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True)
    x = (x - mean) / (std + 1e-8)
    if in_channels == 1 and x.shape[1] > 1:
        x = x[:, :1, :]
    return x


def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives from noise pool.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--h5_path", type=str, default="data/cpc_snn_train.h5")
    parser.add_argument("--noise_indices", type=str, default="data/indices_noise.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_noise", type=int, default=0, help="Use only first N noise samples (0=all)")
    parser.add_argument("--min_prob", type=float, default=0.0, help="Keep only records with p(signal)>=min_prob")
    parser.add_argument("--top_k", type=int, default=1000, help="Keep top-K highest-score records (0=all)")
    parser.add_argument("--output_json", type=str, default="data/hard_negatives.json")
    parser.add_argument("--output_ids_json", type=str, default="data/hard_negative_ids.json")
    parser.add_argument("--use_metal", action="store_true", default=False)
    parser.add_argument("--calibration_json", type=str, default=None, help="Optional calibration artifact JSON")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model, checkpoint, model_kwargs, load_report = load_cpcsnn_from_checkpoint(
        args.checkpoint_path, device, use_metal=args.use_metal
    )
    if not load_report["strict"]:
        print(
            "[Warning] Non-strict checkpoint load "
            f"(missing={len(load_report['missing_keys'])}, unexpected={len(load_report['unexpected_keys'])})"
        )

    with open(args.noise_indices, "r") as f:
        noise_ids = json.load(f)
    if args.max_noise and args.max_noise > 0:
        noise_ids = noise_ids[:args.max_noise]
    print(f"Noise pool size: {len(noise_ids)}")

    use_tf2d = bool(model_kwargs["use_tf2d"])
    in_channels = int(model_kwargs["in_channels"])
    calibrator = load_calibration(args.calibration_json) if args.calibration_json else None
    if calibrator is not None:
        print(f"Loaded calibration: {args.calibration_json} (method={calibrator.method})")
    dataset = HDF5SFTPairDataset(
        h5_path=args.h5_path,
        index_list=noise_ids,
        return_time_series=not use_tf2d,
        add_mask_channel=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    records = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Mining"):
            if use_tf2d:
                x = _prepare_tf2d_input(batch, device=device, in_channels=in_channels)
            else:
                x = _prepare_1d_input(batch, device=device, in_channels=in_channels)
            logits = model(x)[0]
            logits_np = logits.squeeze(1).detach().cpu().numpy()
            if calibrator is not None:
                probs_np = apply_calibration_to_logits(logits_np, calibrator)
            else:
                probs_np = 1.0 / (1.0 + np.exp(-logits_np))
            ids = batch.get("id", [])

            for i, p in enumerate(probs_np.tolist()):
                sid = str(ids[i]) if i < len(ids) else ""
                records.append({"id": sid, "score": float(p), "logit": float(logits_np[i])})

    records.sort(key=lambda r: r["score"], reverse=True)
    if args.min_prob > 0:
        records = [r for r in records if r["score"] >= args.min_prob]
    if args.top_k and args.top_k > 0:
        records = records[:args.top_k]

    for rank, rec in enumerate(records, start=1):
        rec["rank"] = rank

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(records, f, indent=2)

    ids_only = [r["id"] for r in records if r.get("id", "")]
    if args.output_ids_json:
        out_ids = Path(args.output_ids_json)
        out_ids.parent.mkdir(parents=True, exist_ok=True)
        with out_ids.open("w") as f:
            json.dump(ids_only, f, indent=2)

    print(f"Saved hard negatives: {len(records)}")
    if records:
        print(f"Top score: {records[0]['score']:.6f}")
    print(f"Records JSON: {out_json}")
    if args.output_ids_json:
        print(f"IDs JSON: {args.output_ids_json}")


if __name__ == "__main__":
    main()
