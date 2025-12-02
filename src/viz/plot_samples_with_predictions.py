import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.utils.paths import project_path, ensure_dir
from src.data_handling.torch_dataset import HDF5SFTPairDataset
from src.models.input_adapter import pack_ifo_sft
from src.models.simple_cnn import SimpleGWConvNet


def load_yaml(rel_path: str) -> dict:
    path = project_path(rel_path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataset(cfg_data: dict) -> HDF5SFTPairDataset:
    h5_path = project_path(cfg_data["h5_path"])
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 not found at: {h5_path}")

    import h5py
    with h5py.File(h5_path, "r") as h5:
        ids = list(h5.keys())

    if not ids:
        raise RuntimeError("No sample IDs in HDF5.")

    return HDF5SFTPairDataset(
        h5_path=str(h5_path),
        index_list=ids,
        use_phase=cfg_data.get("use_phase", True),
        add_mask_channel=cfg_data.get("add_mask_channel", True),
        enforce_same_shape=True,
    )


def load_model(cfg_data: dict, cfg_model: dict, device: torch.device) -> SimpleGWConvNet:
    in_channels = 2 * (3 + int(cfg_data.get("add_mask_channel", True)))
    model = SimpleGWConvNet(
        in_channels=in_channels,
        num_classes=1,
    ).to(device)

    ckpt_path = project_path(cfg_model.get("save_path", "checkpoints/mvp_simple_cnn.pt"))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def plot_tf(ax, mag: np.ndarray, f: np.ndarray, title: str):
    im = ax.imshow(
        mag.T,
        origin="lower",
        aspect="auto",
        extent=[0, mag.shape[0], float(f[0]), float(f[-1])],
        cmap="viridis",
    )
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(title)
    return im


def main():
    cfg_data = load_yaml("configs/dataset.yaml")
    cfg_model = load_yaml("configs/model.yaml")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset = build_dataset(cfg_data)
    model = load_model(cfg_data, cfg_model, device)

    loader = DataLoader(
        dataset,
        batch_size=int(cfg_data.get("batch_size", 8)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    batch = next(iter(loader))
    x = pack_ifo_sft(batch).to(device)
    with torch.no_grad():
        logits = model(x).cpu().numpy().reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))

    f = batch["f"][0].numpy()

    figures_dir = project_path("figures")
    ensure_dir(figures_dir)

    for i in range(min(4, x.size(0))):
        y_true = float(batch["y"][i].item())
        p = float(probs[i])

        H1 = batch["H1"][i].numpy()
        L1 = batch["L1"][i].numpy()

        mag_H1 = H1[0]
        mag_L1 = L1[0]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        im1 = plot_tf(axes[0], mag_H1, f, title="H1 |X|")
        im2 = plot_tf(axes[1], mag_L1, f, title="L1 |X|")

        fig.suptitle(f"Sample {i} - label={int(y_true)}  pred={p:.3f}", fontsize=12)
        for ax in axes:
            ax.set_xlabel("Time-bin index")
        fig.colorbar(im1, ax=axes.ravel().tolist(), label="|X|")
        fig.tight_layout()

        out_path = figures_dir / f"sample_{i}_label{int(y_true)}_pred{p:.3f}.png"
        plt.savefig(str(out_path), dpi=200)
        plt.close(fig)

        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()