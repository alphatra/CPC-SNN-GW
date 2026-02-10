#!/usr/bin/env python3
"""Hard-negative mining v2 from OOD swapped pairs + false positives.

Outputs:
  - JSON ranking with scores and partner IDs
  - IDs list for training sampler
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_handling.torch_dataset import HDF5SFTPairDataset
from src.evaluation.model_loader import load_cpcsnn_from_checkpoint


@dataclass
class Entry:
    noise_id: str
    score: float
    shift: int
    partner_id: str


def _prepare_input(h1: torch.Tensor, l1: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = torch.cat([h1.to(device), l1.to(device)], dim=1)
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True)
    return (x - mean) / (std + 1e-8)


def _infer_logits(models: list[torch.nn.Module], x: torch.Tensor) -> torch.Tensor:
    logits_sum = None
    for m in models:
        logits, _, _ = m(x)
        logits_sum = logits if logits_sum is None else (logits_sum + logits)
    return logits_sum / float(len(models))


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine hard negatives from swapped OOD protocol.")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--ensemble-checkpoints", nargs="*", default=None)
    ap.add_argument("--noise-h5", type=Path, default=Path("data/cpc_snn_train.h5"))
    ap.add_argument("--indices-noise", type=Path, required=True)
    ap.add_argument("--swaps", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--threshold", type=float, default=None, help="FP threshold. If omitted, use lock.")
    ap.add_argument("--lock", type=Path, default=None, help="Lock JSON to read threshold_at_1e-4 from ood_baseline.")
    ap.add_argument("--top-k", type=int, default=2000)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/hardneg_v2/hardneg_v2_ranked.json"),
    )
    ap.add_argument(
        "--out-indices",
        type=Path,
        default=Path("reports/hardneg_v2/indices_hardneg_v2.json"),
    )
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    noise_ids = [str(x) for x in json.loads(args.indices_noise.read_text())]
    if len(noise_ids) < 2:
        raise SystemExit("[error] need at least 2 noise ids")

    if args.threshold is None:
        if not args.lock:
            raise SystemExit("[error] pass --threshold or --lock")
        lock = json.loads(args.lock.read_text())
        args.threshold = float(
            lock["primary_metrics_snapshot"]["ood_baseline"].get("threshold_at_1e-4", float("nan"))
        )
        if args.threshold != args.threshold:
            raise SystemExit("[error] lock missing ood threshold_at_1e-4")

    device = torch.device(args.device)
    model, _, kwargs, _ = load_cpcsnn_from_checkpoint(
        args.checkpoint, device=device, use_metal=(args.device == "mps"), prefer_kpi=True
    )
    models = [model.eval()]
    for ck in args.ensemble_checkpoints or []:
        m, _, kw2, _ = load_cpcsnn_from_checkpoint(
            ck, device=device, use_metal=(args.device == "mps"), prefer_kpi=True
        )
        if int(kw2["in_channels"]) != int(kwargs["in_channels"]):
            raise SystemExit(f"[error] in_channels mismatch for ensemble member: {ck}")
        models.append(m.eval())

    use_mask = int(kwargs["in_channels"]) in (4, 8)
    slice_idx = 4 if use_mask else 3
    ds = HDF5SFTPairDataset(str(args.noise_h5), noise_ids, add_mask_channel=use_mask)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    all_h1: list[torch.Tensor] = []
    all_l1: list[torch.Tensor] = []
    for batch in dl:
        all_h1.append(batch["H1"][:, 0:slice_idx, :, :].cpu())
        all_l1.append(batch["L1"][:, 0:slice_idx, :, :].cpu())

    h1 = torch.cat(all_h1, dim=0)
    l1 = torch.cat(all_l1, dim=0)
    n = h1.shape[0]
    if n != len(noise_ids):
        noise_ids = noise_ids[:n]

    best_score = np.full(n, -1.0, dtype=np.float64)
    best_shift = np.zeros(n, dtype=np.int64)
    best_partner = np.zeros(n, dtype=np.int64)
    rng = np.random.default_rng(args.seed)

    with torch.no_grad():
        for _ in range(args.swaps):
            min_shift = max(1, n // 4)
            max_shift = max(2, n - 1)
            shift = int(rng.integers(min_shift, max_shift))
            l1_shift = torch.roll(l1, shifts=shift, dims=0)
            for i in range(0, n, args.batch_size):
                j = min(i + args.batch_size, n)
                x = _prepare_input(h1[i:j], l1_shift[i:j], device=device)
                probs = torch.sigmoid(_infer_logits(models, x)).detach().cpu().numpy().reshape(-1)
                idx = np.arange(i, j, dtype=np.int64)
                mask = probs > best_score[idx]
                if np.any(mask):
                    upd_idx = idx[mask]
                    best_score[upd_idx] = probs[mask]
                    best_shift[upd_idx] = shift
                    best_partner[upd_idx] = (upd_idx - shift) % n

    entries: List[Entry] = []
    for i in range(n):
        if best_score[i] >= float(args.threshold):
            p = int(best_partner[i])
            entries.append(
                Entry(
                    noise_id=noise_ids[i],
                    score=float(best_score[i]),
                    shift=int(best_shift[i]),
                    partner_id=noise_ids[p],
                )
            )
    entries.sort(key=lambda x: x.score, reverse=True)
    if args.top_k > 0:
        entries = entries[: args.top_k]

    out_ranked = [
        {
            "noise_id": e.noise_id,
            "score": e.score,
            "shift": e.shift,
            "partner_id": e.partner_id,
        }
        for e in entries
    ]
    out_ids = [e.noise_id for e in entries]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out_ranked, indent=2), encoding="utf-8")
    args.out_indices.parent.mkdir(parents=True, exist_ok=True)
    args.out_indices.write_text(json.dumps(out_ids, indent=2), encoding="utf-8")

    print(f"[ok] threshold={args.threshold:.6f}")
    print(f"[ok] hard negatives saved: {len(out_ids)}")
    print(f"[ok] ranked -> {args.out_json}")
    print(f"[ok] ids -> {args.out_indices}")


if __name__ == "__main__":
    main()

