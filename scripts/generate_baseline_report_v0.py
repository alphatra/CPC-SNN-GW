#!/usr/bin/env python3
"""Generate frozen baseline report with ID/OOD metrics.

Outputs:
  - reports/baseline_report_v0.json
  - reports/baseline_report_v0.md

Metrics:
  - TPR@1e-4
  - pAUC(max_fpr=1e-4)
  - ECE
  - Brier
  - latency (p50/p95, ms)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from src.data_handling.torch_dataset import HDF5SFTPairDataset
from src.evaluation.calibration import CalibrationArtifact, apply_calibration_to_logits, fit_temperature_scaler
from src.evaluation.metrics import compute_brier_score, compute_ece
from src.evaluation.model_loader import load_cpcsnn_from_checkpoint


@dataclass
class ModelBundle:
    checkpoints: List[str]
    models: List[torch.nn.Module]
    in_channels: int


@dataclass
class ScoreOutputs:
    noise_probs: np.ndarray
    signal_probs: np.ndarray
    noise_logits: np.ndarray
    signal_logits: np.ndarray


def load_ids(path: Path) -> List[str]:
    return [str(x) for x in json.loads(path.read_text())]


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def _prepare_input(
    batch: dict,
    device: torch.device,
    in_channels: int,
    channel: str | None = None,
) -> torch.Tensor:
    # Mirrors src/evaluation/evaluate_background.py conventions.
    # in_channels 8 -> both IFOs with mask, 6 -> both without mask, 4/3 single IFO.
    use_mask = in_channels in (4, 8)
    slice_idx = 4 if use_mask else 3

    def get_feats(ifo: str) -> torch.Tensor:
        d = batch[ifo].to(device)
        return d[:, 0:slice_idx, :, :]

    feat_list: List[torch.Tensor] = []
    if channel != "L1":
        feat_list.append(get_feats("H1"))
    if channel != "H1":
        feat_list.append(get_feats("L1"))

    x = torch.cat(feat_list, dim=1)
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True)
    x = (x - mean) / (std + 1e-8)
    return x


def load_model_bundle(
    checkpoints: Sequence[str],
    device: torch.device,
    use_metal: bool = False,
) -> ModelBundle:
    models: List[torch.nn.Module] = []
    in_channels: int | None = None
    requested_ckpts = [str(c) for c in checkpoints]
    resolved_ckpts: List[str] = []
    for ckpt in requested_ckpts:
        model, _, kwargs, load_report = load_cpcsnn_from_checkpoint(
            ckpt, device=device, use_metal=use_metal, prefer_kpi=True
        )
        resolved = str(load_report.get("resolved_checkpoint", ckpt))
        resolved_ckpts.append(resolved)
        if not kwargs["use_tf2d"]:
            raise RuntimeError(f"Checkpoint is not TF2D-compatible: {ckpt}")
        if in_channels is None:
            in_channels = int(kwargs["in_channels"])
        elif in_channels != int(kwargs["in_channels"]):
            raise RuntimeError(
                f"Incompatible in_channels in ensemble: {in_channels} vs {kwargs['in_channels']} ({ckpt})"
            )
        model.eval()
        models.append(model)
    assert in_channels is not None
    return ModelBundle(checkpoints=resolved_ckpts, models=models, in_channels=in_channels)


def infer_logits(bundle: ModelBundle, x: torch.Tensor) -> torch.Tensor:
    logits_sum = None
    for model in bundle.models:
        logits, _, _ = model(x)
        logits_sum = logits if logits_sum is None else (logits_sum + logits)
    return logits_sum / float(len(bundle.models))


def run_standard_scores(
    bundle: ModelBundle,
    h5_path: Path,
    noise_ids: Sequence[str],
    signal_ids: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> ScoreOutputs:
    ds_noise = HDF5SFTPairDataset(str(h5_path), list(noise_ids), add_mask_channel=(bundle.in_channels in (4, 8)))
    ds_signal = HDF5SFTPairDataset(str(h5_path), list(signal_ids), add_mask_channel=(bundle.in_channels in (4, 8)))
    loader_noise = DataLoader(ds_noise, batch_size=batch_size, num_workers=0)
    loader_signal = DataLoader(ds_signal, batch_size=batch_size, num_workers=0)

    noise_probs: List[np.ndarray] = []
    signal_probs: List[np.ndarray] = []
    noise_logits: List[np.ndarray] = []
    signal_logits: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader_noise:
            x = _prepare_input(batch, device=device, in_channels=bundle.in_channels)
            logits = infer_logits(bundle, x)
            noise_logits.append(logits.detach().cpu().numpy().reshape(-1))
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            noise_probs.append(probs)

        for batch in loader_signal:
            x = _prepare_input(batch, device=device, in_channels=bundle.in_channels)
            logits = infer_logits(bundle, x)
            signal_logits.append(logits.detach().cpu().numpy().reshape(-1))
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            signal_probs.append(probs)

    return ScoreOutputs(
        noise_probs=np.concatenate(noise_probs),
        signal_probs=np.concatenate(signal_probs),
        noise_logits=np.concatenate(noise_logits),
        signal_logits=np.concatenate(signal_logits),
    )


def run_swapped_noise_scores(
    bundle: ModelBundle,
    h5_path: Path,
    noise_ids: Sequence[str],
    batch_size: int,
    swaps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    ds_noise = HDF5SFTPairDataset(str(h5_path), list(noise_ids), add_mask_channel=(bundle.in_channels in (4, 8)))
    loader_noise = DataLoader(ds_noise, batch_size=batch_size, num_workers=0)

    all_h1 = []
    all_l1 = []
    use_mask = bundle.in_channels in (4, 8)
    slice_idx = 4 if use_mask else 3

    with torch.no_grad():
        for batch in loader_noise:
            h1 = batch["H1"][:, 0:slice_idx, :, :].cpu()
            l1 = batch["L1"][:, 0:slice_idx, :, :].cpu()
            all_h1.append(h1)
            all_l1.append(l1)

    full_h1 = torch.cat(all_h1, dim=0)
    full_l1 = torch.cat(all_l1, dim=0)
    n = full_h1.shape[0]

    out_probs: List[np.ndarray] = []
    out_logits: List[np.ndarray] = []
    with torch.no_grad():
        for _ in range(swaps):
            min_shift = max(1, n // 4)
            max_shift = max(2, n - 1)
            shift = int(np.random.randint(min_shift, max_shift))
            l1_shifted = torch.roll(full_l1, shifts=shift, dims=0)

            for i in range(0, n, batch_size):
                b_h1 = full_h1[i : i + batch_size].to(device)
                b_l1 = l1_shifted[i : i + batch_size].to(device)
                x = torch.cat([b_h1, b_l1], dim=1)
                mean = x.mean(dim=(2, 3), keepdim=True)
                std = x.std(dim=(2, 3), keepdim=True)
                x = (x - mean) / (std + 1e-8)
                logits = infer_logits(bundle, x)
                out_logits.append(logits.detach().cpu().numpy().reshape(-1))
                probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                out_probs.append(probs)

    return np.concatenate(out_probs), np.concatenate(out_logits)


def compute_detection_metrics(noise_probs: np.ndarray, signal_probs: np.ndarray) -> dict:
    y_true = np.concatenate([np.zeros_like(noise_probs), np.ones_like(signal_probs)])
    y_score = np.concatenate([noise_probs, signal_probs])

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # first index with fpr >= target
    target = 1e-4
    idx = np.where(fpr >= target)[0]
    idx = int(idx[0]) if len(idx) else len(fpr) - 1

    # normalized pAUC in sklearn when max_fpr is used
    pauc_norm = float(roc_auc_score(y_true, y_score, max_fpr=1e-4))
    # raw pAUC
    mask = fpr <= 1e-4
    if not np.any(mask):
        pauc_raw = 0.0
    else:
        pauc_raw = float(np.trapezoid(tpr[mask], fpr[mask]))

    ece = compute_ece(y_true.astype(np.int64), y_score.astype(np.float64), n_bins=15)
    brier = compute_brier_score(y_true.astype(np.float64), y_score.astype(np.float64))

    return {
        "tpr_at_1e-4": float(tpr[idx]),
        "threshold_at_1e-4": float(thresholds[idx]),
        "actual_fpr_at_1e-4": float(fpr[idx]),
        "pauc_norm_max_fpr_1e-4": pauc_norm,
        "pauc_raw_fpr_le_1e-4": pauc_raw,
        "ece_15bins": float(ece),
        "brier": float(brier),
        "noise_mean": float(noise_probs.mean()),
        "noise_max": float(noise_probs.max()),
        "signal_mean": float(signal_probs.mean()),
    }


def fit_and_apply_temperature(
    fit_noise_logits: np.ndarray,
    fit_signal_logits: np.ndarray,
    eval_noise_logits: np.ndarray,
    eval_signal_logits: np.ndarray,
) -> tuple[CalibrationArtifact, np.ndarray, np.ndarray]:
    fit_logits = np.concatenate([fit_noise_logits, fit_signal_logits]).astype(np.float64)
    fit_labels = np.concatenate(
        [np.zeros_like(fit_noise_logits, dtype=np.float64), np.ones_like(fit_signal_logits, dtype=np.float64)]
    )
    temperature = fit_temperature_scaler(fit_logits, fit_labels, max_iter=300, lr=0.05)
    artifact = CalibrationArtifact(method="temperature", temperature=float(temperature))
    eval_noise_probs_cal = apply_calibration_to_logits(eval_noise_logits, artifact)
    eval_signal_probs_cal = apply_calibration_to_logits(eval_signal_logits, artifact)
    return artifact, eval_noise_probs_cal, eval_signal_probs_cal


def measure_latency(
    bundle: ModelBundle,
    h5_path: Path,
    noise_ids: Sequence[str],
    signal_ids: Sequence[str],
    batch_size: int,
    device: torch.device,
    n_warmup_batches: int = 5,
    n_measure_batches: int = 30,
) -> dict:
    use_mask = bundle.in_channels in (4, 8)
    # mix half noise/half signal for realistic steady-state
    take_n = min(len(noise_ids), len(signal_ids), batch_size * (n_warmup_batches + n_measure_batches))
    ids = list(noise_ids[:take_n // 2]) + list(signal_ids[:take_n // 2])
    ds = HDF5SFTPairDataset(str(h5_path), ids, add_mask_channel=use_mask)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=False)

    times_ms: List[float] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = _prepare_input(batch, device=device, in_channels=bundle.in_channels)
            _sync_device(device)
            t0 = time.perf_counter()
            _ = infer_logits(bundle, x)
            _sync_device(device)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            if i >= n_warmup_batches:
                times_ms.append(dt_ms)
            if len(times_ms) >= n_measure_batches:
                break

    arr = np.array(times_ms, dtype=np.float64) if times_ms else np.array([0.0], dtype=np.float64)
    bs = float(batch_size)
    return {
        "batch_size": int(batch_size),
        "n_batches": int(len(arr)),
        "batch_latency_ms_p50": float(np.percentile(arr, 50)),
        "batch_latency_ms_p95": float(np.percentile(arr, 95)),
        "batch_latency_ms_mean": float(arr.mean()),
        "sample_latency_ms_p50": float(np.percentile(arr / bs, 50)),
        "sample_latency_ms_p95": float(np.percentile(arr / bs, 95)),
        "sample_latency_ms_mean": float((arr / bs).mean()),
    }


def write_markdown(report: dict, out_md: Path) -> None:
    id_m = report["id_baseline"]["metrics"]
    ood_m = report["ood_baseline"]["metrics"]
    id_l = report["id_baseline"]["latency"]
    ood_l = report["ood_baseline"]["latency"]
    txt = f"""# Baseline Report v0

Date: `{report['date_utc']}`

## Frozen Baselines
- ID baseline: `{report['id_baseline']['name']}`
- OOD baseline: `{report['ood_baseline']['name']}`

## Metrics (requested set)

| Scope | TPR@1e-4 | pAUC(norm,1e-4) | ECE | Brier |
|---|---:|---:|---:|---:|
| ID | {id_m['tpr_at_1e-4']:.6f} | {id_m['pauc_norm_max_fpr_1e-4']:.6f} | {id_m['ece_15bins']:.6f} | {id_m['brier']:.6f} |
| OOD | {ood_m['tpr_at_1e-4']:.6f} | {ood_m['pauc_norm_max_fpr_1e-4']:.6f} | {ood_m['ece_15bins']:.6f} | {ood_m['brier']:.6f} |

## Latency

| Scope | batch p50 [ms] | batch p95 [ms] | sample p50 [ms] | sample p95 [ms] |
|---|---:|---:|---:|---:|
| ID | {id_l['batch_latency_ms_p50']:.3f} | {id_l['batch_latency_ms_p95']:.3f} | {id_l['sample_latency_ms_p50']:.4f} | {id_l['sample_latency_ms_p95']:.4f} |
| OOD | {ood_l['batch_latency_ms_p50']:.3f} | {ood_l['batch_latency_ms_p95']:.3f} | {ood_l['sample_latency_ms_p50']:.4f} | {ood_l['sample_latency_ms_p95']:.4f} |

## Protocol
- ID: standard noise/signal pairs on default indices.
- OOD: noise via `swapped_pairs` on `test_late` split, signal from `test_late` split.
- No calibration applied (uncalibrated scores).
"""
    cal = report.get("calibration", {})
    if cal:
        txt += "\n## Temperature Calibration (post-hoc)\n\n"
        if "id" in cal:
            cid = cal["id"]
            txt += (
                f"- ID temperature: `{cid.get('temperature', 1.0):.6f}`\n"
                f"  - ECE raw -> cal: `{cid['raw_metrics']['ece_15bins']:.6f}` -> `{cid['calibrated_metrics']['ece_15bins']:.6f}`\n"
                f"  - Brier raw -> cal: `{cid['raw_metrics']['brier']:.6f}` -> `{cid['calibrated_metrics']['brier']:.6f}`\n"
            )
        if "ood" in cal:
            cood = cal["ood"]
            txt += (
                f"- OOD temperature: `{cood.get('temperature', 1.0):.6f}`\n"
                f"  - ECE raw -> cal: `{cood['raw_metrics']['ece_15bins']:.6f}` -> `{cood['calibrated_metrics']['ece_15bins']:.6f}`\n"
                f"  - Brier raw -> cal: `{cood['raw_metrics']['brier']:.6f}` -> `{cood['calibrated_metrics']['brier']:.6f}`\n"
            )
    out_md.write_text(txt, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze baseline report v0.")
    ap.add_argument("--decision-json", type=Path, default=Path("reports/final_decision.json"))
    ap.add_argument("--h5-path", type=Path, default=Path("data/cpc_snn_train.h5"))
    ap.add_argument("--indices-noise", type=Path, default=Path("data/indices_noise.json"))
    ap.add_argument("--indices-signal", type=Path, default=Path("data/indices_signal.json"))
    ap.add_argument("--ood-noise", type=Path, default=Path("reports/ood_time/splits/indices_noise_test_late.json"))
    ap.add_argument("--ood-signal", type=Path, default=Path("reports/ood_time/splits/indices_signal_test_late.json"))
    ap.add_argument("--ood-swaps", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-id-noise", type=int, default=0)
    ap.add_argument("--max-id-signal", type=int, default=0)
    ap.add_argument("--max-ood-noise", type=int, default=0)
    ap.add_argument("--max-ood-signal", type=int, default=0)
    ap.add_argument("--latency-warmup-batches", type=int, default=5)
    ap.add_argument("--latency-measure-batches", type=int, default=30)
    ap.add_argument("--fit-temp-id", action="store_true", help="Fit temperature calibrator for ID scope.")
    ap.add_argument("--fit-temp-ood", action="store_true", help="Fit temperature calibrator for OOD scope.")
    ap.add_argument("--id-cal-noise", type=Path, default=None, help="Noise indices for ID calibration fit.")
    ap.add_argument("--id-cal-signal", type=Path, default=None, help="Signal indices for ID calibration fit.")
    ap.add_argument(
        "--ood-cal-noise",
        type=Path,
        default=Path("reports/ood_time/splits/indices_noise_train_early.json"),
        help="Noise indices for OOD calibration fit.",
    )
    ap.add_argument(
        "--ood-cal-signal",
        type=Path,
        default=Path("reports/ood_time/splits/indices_signal_train_early.json"),
        help="Signal indices for OOD calibration fit.",
    )
    ap.add_argument("--id-cal-out", type=Path, default=Path("configs/calibration/id_temperature.json"))
    ap.add_argument("--ood-cal-out", type=Path, default=Path("configs/calibration/ood_temperature.json"))
    ap.add_argument("--protocol-tag", type=str, default="full")
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--use-metal", action="store_true", default=False)
    ap.add_argument("--out-json", type=Path, default=Path("reports/baseline_report_v0.json"))
    ap.add_argument("--out-md", type=Path, default=Path("reports/baseline_report_v0.md"))
    args = ap.parse_args()

    decision = json.loads(args.decision_json.read_text())
    id_ckpt = decision["id_candidate"]["checkpoint"]
    ood_ckpts = decision["ood_candidate"]["checkpoints"]

    device = torch.device(args.device)
    noise_ids = load_ids(args.indices_noise)
    signal_ids = load_ids(args.indices_signal)
    ood_noise_ids = load_ids(args.ood_noise)
    ood_signal_ids = load_ids(args.ood_signal)

    if args.max_id_noise > 0:
        noise_ids = noise_ids[: args.max_id_noise]
    if args.max_id_signal > 0:
        signal_ids = signal_ids[: args.max_id_signal]
    if args.max_ood_noise > 0:
        ood_noise_ids = ood_noise_ids[: args.max_ood_noise]
    if args.max_ood_signal > 0:
        ood_signal_ids = ood_signal_ids[: args.max_ood_signal]

    # Build model bundles
    print("[stage] loading model bundles...")
    id_bundle = load_model_bundle([id_ckpt], device=device, use_metal=args.use_metal)
    ood_bundle = load_model_bundle(ood_ckpts, device=device, use_metal=args.use_metal)

    # ID metrics
    print("[stage] ID scores...")
    id_scores = run_standard_scores(
        id_bundle, args.h5_path, noise_ids, signal_ids, args.batch_size, device
    )
    id_metrics = compute_detection_metrics(id_scores.noise_probs, id_scores.signal_probs)
    print("[stage] ID latency...")
    id_latency = measure_latency(
        id_bundle,
        args.h5_path,
        noise_ids,
        signal_ids,
        args.batch_size,
        device,
        n_warmup_batches=args.latency_warmup_batches,
        n_measure_batches=args.latency_measure_batches,
    )

    # OOD metrics: swapped noise + late signals
    print("[stage] OOD swapped-noise scores...")
    ood_noise_probs, ood_noise_logits = run_swapped_noise_scores(
        ood_bundle, args.h5_path, ood_noise_ids, args.batch_size, args.ood_swaps, device
    )
    # keep signals "natural" and late split
    print("[stage] OOD signal scores...")
    ood_signal_scores = run_standard_scores(
        ood_bundle, args.h5_path, ood_noise_ids[: min(len(ood_noise_ids), len(ood_signal_ids))], ood_signal_ids, args.batch_size, device
    )
    ood_signal_probs = ood_signal_scores.signal_probs
    ood_metrics = compute_detection_metrics(ood_noise_probs, ood_signal_probs)
    print("[stage] OOD latency...")
    ood_latency = measure_latency(
        ood_bundle,
        args.h5_path,
        ood_noise_ids,
        ood_signal_ids,
        args.batch_size,
        device,
        n_warmup_batches=args.latency_warmup_batches,
        n_measure_batches=args.latency_measure_batches,
    )

    calibration_section: dict = {}
    if args.fit_temp_id:
        id_fit_noise = load_ids(args.id_cal_noise) if args.id_cal_noise else noise_ids
        id_fit_signal = load_ids(args.id_cal_signal) if args.id_cal_signal else signal_ids
        id_fit_scores = run_standard_scores(id_bundle, args.h5_path, id_fit_noise, id_fit_signal, args.batch_size, device)
        id_artifact, id_noise_cal, id_signal_cal = fit_and_apply_temperature(
            fit_noise_logits=id_fit_scores.noise_logits,
            fit_signal_logits=id_fit_scores.signal_logits,
            eval_noise_logits=id_scores.noise_logits,
            eval_signal_logits=id_scores.signal_logits,
        )
        args.id_cal_out.parent.mkdir(parents=True, exist_ok=True)
        args.id_cal_out.write_text(json.dumps(id_artifact.to_dict(), indent=2), encoding="utf-8")
        calibration_section["id"] = {
            "method": "temperature",
            "temperature": float(id_artifact.temperature),
            "fit_noise_count": int(len(id_fit_scores.noise_logits)),
            "fit_signal_count": int(len(id_fit_scores.signal_logits)),
            "artifact_path": str(args.id_cal_out),
            "raw_metrics": id_metrics,
            "calibrated_metrics": compute_detection_metrics(id_noise_cal, id_signal_cal),
        }

    if args.fit_temp_ood:
        ood_fit_noise = load_ids(args.ood_cal_noise)
        ood_fit_signal = load_ids(args.ood_cal_signal)
        ood_fit_scores = run_standard_scores(ood_bundle, args.h5_path, ood_fit_noise, ood_fit_signal, args.batch_size, device)
        ood_artifact, ood_noise_cal, ood_signal_cal = fit_and_apply_temperature(
            fit_noise_logits=ood_fit_scores.noise_logits,
            fit_signal_logits=ood_fit_scores.signal_logits,
            eval_noise_logits=ood_noise_logits,
            eval_signal_logits=ood_signal_scores.signal_logits,
        )
        args.ood_cal_out.parent.mkdir(parents=True, exist_ok=True)
        args.ood_cal_out.write_text(json.dumps(ood_artifact.to_dict(), indent=2), encoding="utf-8")
        calibration_section["ood"] = {
            "method": "temperature",
            "temperature": float(ood_artifact.temperature),
            "fit_noise_count": int(len(ood_fit_scores.noise_logits)),
            "fit_signal_count": int(len(ood_fit_scores.signal_logits)),
            "artifact_path": str(args.ood_cal_out),
            "raw_metrics": ood_metrics,
            "calibrated_metrics": compute_detection_metrics(ood_noise_cal, ood_signal_cal),
        }

    report = {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": "baseline_freeze_v0",
        "protocol_tag": args.protocol_tag,
        "primary_kpi": "TPR@1e-4",
        "id_baseline": {
            "name": decision["id_candidate"]["variant"],
            "checkpoints": list(id_bundle.checkpoints),
            "n_scores_noise": int(len(id_scores.noise_probs)),
            "n_scores_signal": int(len(id_scores.signal_probs)),
            "metrics": id_metrics,
            "latency": id_latency,
        },
        "ood_baseline": {
            "name": decision["ood_candidate"]["variant"],
            "checkpoints": list(ood_bundle.checkpoints),
            "protocol": {
                "noise_method": "swapped_pairs",
                "swaps": int(args.ood_swaps),
                "noise_split": str(args.ood_noise),
                "signal_split": str(args.ood_signal),
            },
            "n_scores_noise": int(len(ood_noise_probs)),
            "n_scores_signal": int(len(ood_signal_probs)),
            "metrics": ood_metrics,
            "latency": ood_latency,
        },
        "calibration": calibration_section,
        "paths": {
            "h5": str(args.h5_path),
            "indices_noise": str(args.indices_noise),
            "indices_signal": str(args.indices_signal),
        },
        "runtime": {
            "device": args.device,
            "batch_size": int(args.batch_size),
            "max_id_noise": int(args.max_id_noise),
            "max_id_signal": int(args.max_id_signal),
            "max_ood_noise": int(args.max_ood_noise),
            "max_ood_signal": int(args.max_ood_signal),
            "latency_warmup_batches": int(args.latency_warmup_batches),
            "latency_measure_batches": int(args.latency_measure_batches),
            "fit_temp_id": bool(args.fit_temp_id),
            "fit_temp_ood": bool(args.fit_temp_ood),
        },
    }

    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.out_md)
    print(f"[ok] JSON: {args.out_json}")
    print(f"[ok] MD:   {args.out_md}")


if __name__ == "__main__":
    main()
