import os
import torch
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.repro import set_determinism, log_environment
from src.data_handling.torch_dataset import HDF5SFTPairDataset
from src.models.input_adapter import pack_ifo_sft
from src.models.simple_cnn import SimpleGWConvNet


def resolve_path(project_root: str, path: str) -> str:
    """
    Jeśli path jest względna, interpretuj ją względem project_root.
    Dzięki temu działa niezależnie od hydra.run.dir.
    """
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def build_dataset(cfg: DictConfig, project_root: str) -> HDF5SFTPairDataset:
    h5_path = resolve_path(project_root, cfg.dataset.h5_path)
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 not found at: {h5_path}")

    import h5py
    with h5py.File(h5_path, "r") as h5:
        ids = list(h5.keys())
    if not ids:
        raise RuntimeError("No sample IDs in HDF5.")

    return HDF5SFTPairDataset(
        h5_path=h5_path,
        index_list=ids,
        use_phase=cfg.dataset.use_phase,
        add_mask_channel=cfg.dataset.add_mask_channel,
        enforce_same_shape=True,
    )


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss, n_samples = 0.0, 0
    for batch in loader:
        x = pack_ifo_sft(batch).to(device)
        y = batch["y"].view(-1, 1).to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
    return total_loss / max(1, n_samples)


def evaluate(model, loader, device) -> float:
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss, n_samples = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x = pack_ifo_sft(batch).to(device)
            y = batch["y"].view(-1, 1).to(device)
            logits = model(x)
            loss = criterion(logits, y)
            bs = x.size(0)
            total_loss += loss.item() * bs
            n_samples += bs
    return total_loss / max(1, n_samples)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Uruchamiaj:
      cd <repo_root>
      python -m src.train.train_mvp
    Hydra zapisze wyniki w ./outputs/..., ale my kotwiczymy ścieżki danych
    względem katalogu projektu, więc działają zawsze.
    """
    set_determinism(42)

    # Katalog projektu = katalog dwa poziomy nad tym plikiem
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    # Rozwiąż ścieżki zależne od projektu
    h5_path = resolve_path(project_root, cfg.dataset.h5_path)
    save_path = resolve_path(project_root, cfg.model.save_path)
    log_dir = resolve_path(project_root, cfg.model.log_dir)

    dataset = build_dataset(cfg, project_root)

    # prosty split
    val_fraction = float(cfg.dataset.val_fraction)
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Device + DataLoader (MPS bez workers)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        num_workers, pin_memory = 0, False
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        num_workers = int(cfg.dataset.num_workers)
        pin_memory = True
    else:
        device = torch.device("cpu")
        num_workers = int(cfg.dataset.num_workers)
        pin_memory = False

    batch_size = int(cfg.dataset.batch_size)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    in_channels = 2 * (3 + int(cfg.dataset.add_mask_channel))
    model = SimpleGWConvNet(
        in_channels=in_channels,
        num_classes=1,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.model.lr),
        weight_decay=float(cfg.model.weight_decay),
    )

    n_epochs = int(cfg.model.epochs)

    os.makedirs(log_dir, exist_ok=True)
    # zapisujemy "spłaszczony" config (Hydra już daje config.yaml w outputs, ale mieć kopię nie szkodzi)
    log_environment(OmegaConf.to_container(cfg, resolve=True), log_dir)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val = float("inf")

    print(f"Device: {device}")
    print(f"HDF5: {h5_path}")
    print(f"Checkpoints: {save_path}")
    print(f"Train/Val: {n_train}/{n_val}")

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        print(
            f"[Epoch {epoch}/{n_epochs}] "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved to {save_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()