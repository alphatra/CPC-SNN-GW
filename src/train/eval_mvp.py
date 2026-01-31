import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import yaml

from src.utils.paths import project_path
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


def main():
    cfg_data = load_yaml("configs/dataset.yaml")
    cfg_model = load_yaml("configs/model.yaml")

    # Device (MPS-aware)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset = build_dataset(cfg_data)

    # Dataloader (na MPS: workers=0, pin_memory=False)
    if device.type == "mps":
        num_workers = 0
        pin_memory = False
    else:
        num_workers = int(cfg_data.get("num_workers", 2))
        pin_memory = bool(cfg_data.get("pin_memory", True))

    loader = DataLoader(
        dataset,
        batch_size=int(cfg_data.get("batch_size", 8)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Model
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

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            x = pack_ifo_sft(batch).to(device)
            y = batch["label"].view(-1, 1).to(device)
            logits = model(x)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0).reshape(-1)
    all_labels = np.concatenate(all_labels, axis=0).reshape(-1)

    probs = 1.0 / (1.0 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(np.float32)

    acc = float((preds == all_labels).mean())

    # ROC-AUC jeśli dostępny sklearn
    auc = None
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(all_labels, probs))
    except Exception:
        pass

    print(f"Total samples: {len(all_labels)}")
    print(f"Accuracy: {acc:.3f}")
    if auc is not None:
        print(f"ROC-AUC: {auc:.3f}")

    sig_probs = probs[all_labels == 1.0]
    bkg_probs = probs[all_labels == 0.0]
    if sig_probs.size > 0:
        print(f"Signal probs: mean={sig_probs.mean():.3f}, std={sig_probs.std():.3f}, n={sig_probs.size}")
    if bkg_probs.size > 0:
        print(f"Noise probs:  mean={bkg_probs.mean():.3f}, std={bkg_probs.std():.3f}, n={bkg_probs.size}")


if __name__ == "__main__":
    main()