from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_handling.torch_dataset import HDF5SFTPairDataset
from src.evaluation.model_loader import load_cpcsnn_from_checkpoint


def _plot_timeseries(sample_id: str, x: np.ndarray, score: float, save_path: Path) -> None:
    t = np.arange(x.shape[-1])
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    axes[0].plot(t, x[0], color="#1f77b4", alpha=0.9)
    axes[0].set_title(f"H1 | sample={sample_id}")
    axes[0].set_ylabel("Strain")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, x[1], color="#ff7f0e", alpha=0.9)
    axes[1].set_title(f"L1 | score={score:.4f}")
    axes[1].set_xlabel("Time index")
    axes[1].set_ylabel("Strain")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize hard negative noise samples.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--h5_path", default="data/cpc_snn_train.h5", type=str)
    parser.add_argument("--noise_indices", default="data/indices_noise.json", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--max_samples", default=20, type=int)
    parser.add_argument("--output_dir", default="figures/hard_negatives", type=str)
    parser.add_argument("--use_metal", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _, model_kwargs, _ = load_cpcsnn_from_checkpoint(
        args.checkpoint, device, use_metal=args.use_metal
    )
    if model_kwargs["use_tf2d"]:
        raise RuntimeError("This visualizer supports only 1D/time-series checkpoints.")

    with open(args.noise_indices, "r") as f:
        noise_ids = json.load(f)

    dataset = HDF5SFTPairDataset(
        h5_path=args.h5_path,
        index_list=noise_ids,
        return_time_series=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    found = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Scanning noise"):
            x = batch["x"].to(device)
            # Keep preprocessing consistent with evaluate_snn.
            x = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-8)
            probs = torch.sigmoid(model(x)[0]).squeeze(1).cpu().numpy()
            ids = batch.get("id", [])

            for i, p in enumerate(probs):
                if p <= args.threshold:
                    continue
                sid = str(ids[i]) if i < len(ids) else f"idx_{found}"
                save_path = out_dir / f"hard_negative_{sid}_p{p:.3f}.png"
                _plot_timeseries(sid, x[i].detach().cpu().numpy(), float(p), save_path)
                found += 1
                if found >= args.max_samples:
                    print(f"Saved {found} hard negatives to {out_dir}")
                    return

    print(f"Saved {found} hard negatives to {out_dir}")


if __name__ == "__main__":
    main()
