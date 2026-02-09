#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_handling.torch_dataset import HDF5SFTPairDataset
from src.evaluation.calibration import (
    CalibrationArtifact,
    apply_calibration_to_logits,
    fit_isotonic_calibrator,
    fit_temperature_scaler,
    save_calibration,
)
from src.evaluation.metrics import compute_brier_score, compute_ece, compute_tpr_at_fpr
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


def _collect_logits_labels(model, ids, h5_path, batch_size, use_tf2d, in_channels, device):
    dataset = HDF5SFTPairDataset(
        h5_path=h5_path,
        index_list=ids,
        return_time_series=not use_tf2d,
        add_mask_channel=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logits_all = []
    labels_all = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=False):
            if use_tf2d:
                x = _prepare_tf2d_input(batch, device=device, in_channels=in_channels)
            else:
                x = _prepare_1d_input(batch, device=device, in_channels=in_channels)
            logits = model(x)[0].squeeze(1).detach().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy()
            logits_all.extend(logits.tolist())
            labels_all.extend(labels.tolist())

    return np.asarray(logits_all, dtype=np.float64), np.asarray(labels_all, dtype=np.int64)


def _compute_metrics(labels, probs):
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    labels = np.asarray(labels, dtype=np.int64)
    out = {
        "ece": float(compute_ece(labels, probs)),
        "brier": float(compute_brier_score(labels, probs)),
        "nll": float(log_loss(labels, probs)),
    }
    tpr = compute_tpr_at_fpr(labels, probs, fpr_thresholds=[1e-3, 1e-4, 1e-5, 1e-6])
    out["tpr_at_fpr"] = {f"{k:.0e}": float(v["tpr"]) for k, v in tpr.items()}
    out["thr_at_fpr"] = {f"{k:.0e}": float(v["threshold"]) for k, v in tpr.items()}
    return out


def _split_ids(noise_ids, signal_ids, holdout_frac, seed):
    rng = np.random.default_rng(seed)
    noise_ids = np.asarray(noise_ids, dtype=object)
    signal_ids = np.asarray(signal_ids, dtype=object)
    rng.shuffle(noise_ids)
    rng.shuffle(signal_ids)

    n_hold_noise = int(round(len(noise_ids) * holdout_frac))
    n_hold_signal = int(round(len(signal_ids) * holdout_frac))

    fit_noise = noise_ids[n_hold_noise:].tolist()
    fit_signal = signal_ids[n_hold_signal:].tolist()
    eval_noise = noise_ids[:n_hold_noise].tolist()
    eval_signal = signal_ids[:n_hold_signal].tolist()

    fit_ids = fit_noise + fit_signal
    eval_ids = eval_noise + eval_signal
    return fit_ids, eval_ids


def main():
    parser = argparse.ArgumentParser(description="Fit post-hoc score calibration for p(signal).")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--h5_path", type=str, default="data/cpc_snn_train.h5")
    parser.add_argument("--noise_indices", type=str, default="data/indices_noise.json")
    parser.add_argument("--signal_indices", type=str, default="data/indices_signal.json")
    parser.add_argument("--n_samples_per_class", type=int, default=0, help="0=use all")
    parser.add_argument("--holdout_frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--method",
        type=str,
        default="temperature_isotonic",
        choices=["temperature", "isotonic", "temperature_isotonic"],
    )
    parser.add_argument("--output_json", type=str, default="checkpoints/calibration.json")
    parser.add_argument("--use_metal", action="store_true", default=False)
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model, _, model_kwargs, _ = load_cpcsnn_from_checkpoint(
        args.checkpoint_path, device, use_metal=args.use_metal
    )
    use_tf2d = bool(model_kwargs["use_tf2d"])
    in_channels = int(model_kwargs["in_channels"])

    with open(args.noise_indices, "r") as f:
        noise_ids = json.load(f)
    with open(args.signal_indices, "r") as f:
        signal_ids = json.load(f)

    if args.n_samples_per_class and args.n_samples_per_class > 0:
        noise_ids = noise_ids[: args.n_samples_per_class]
        signal_ids = signal_ids[: args.n_samples_per_class]

    fit_ids, eval_ids = _split_ids(noise_ids, signal_ids, args.holdout_frac, args.seed)
    if len(fit_ids) == 0:
        raise RuntimeError("Calibration fit split is empty. Reduce --holdout_frac.")
    if len(eval_ids) == 0:
        print("[Warning] Holdout split empty. Metrics will be computed on fit split.")
        eval_ids = fit_ids

    print(f"Model mode: {'TF2D' if use_tf2d else '1D/time-series'} | in_channels={in_channels}")
    print(f"Calibration fit samples: {len(fit_ids)} | holdout eval samples: {len(eval_ids)}")

    fit_logits, fit_labels = _collect_logits_labels(
        model=model,
        ids=fit_ids,
        h5_path=args.h5_path,
        batch_size=args.batch_size,
        use_tf2d=use_tf2d,
        in_channels=in_channels,
        device=device,
    )

    eval_logits, eval_labels = _collect_logits_labels(
        model=model,
        ids=eval_ids,
        h5_path=args.h5_path,
        batch_size=args.batch_size,
        use_tf2d=use_tf2d,
        in_channels=in_channels,
        device=device,
    )

    fit_probs_raw = 1.0 / (1.0 + np.exp(-fit_logits))
    eval_probs_raw = 1.0 / (1.0 + np.exp(-eval_logits))

    temperature = 1.0
    if args.method in ("temperature", "temperature_isotonic"):
        temperature = fit_temperature_scaler(fit_logits, fit_labels)

    fit_probs_temp = 1.0 / (1.0 + np.exp(-(fit_logits / temperature)))
    eval_probs_temp = 1.0 / (1.0 + np.exp(-(eval_logits / temperature)))

    iso_x = None
    iso_y = None
    if args.method in ("isotonic", "temperature_isotonic"):
        iso_fit_input = fit_probs_temp if args.method == "temperature_isotonic" else fit_probs_raw
        iso_x, iso_y = fit_isotonic_calibrator(iso_fit_input, fit_labels)

    artifact = CalibrationArtifact(
        method=args.method,
        temperature=temperature,
        isotonic_x=iso_x,
        isotonic_y=iso_y,
    )

    eval_probs_cal = apply_calibration_to_logits(eval_logits, artifact)
    before = _compute_metrics(eval_labels, eval_probs_raw)
    after = _compute_metrics(eval_labels, eval_probs_cal)

    extra = {
        "fit_size": int(len(fit_ids)),
        "eval_size": int(len(eval_ids)),
        "before": before,
        "after": after,
    }
    save_calibration(args.output_json, artifact, extra=extra)

    print(f"Calibration method: {args.method}")
    print(f"Temperature: {temperature:.6f}")
    print(
        f"ECE: {before['ece']:.6f} -> {after['ece']:.6f} | "
        f"Brier: {before['brier']:.6f} -> {after['brier']:.6f}"
    )
    print(f"Saved calibration artifact: {args.output_json}")


if __name__ == "__main__":
    main()
