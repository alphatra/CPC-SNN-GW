from __future__ import annotations

import torch


def pack_ifo_sft(batch: dict) -> torch.Tensor:
    """
    Packs multi-IFO SFT tensors into a single model input tensor.

    Expected batch keys:
    - "H1": (B, C, T, F)
    - "L1": (B, C, T, F)

    Returns:
    - x: (B, 2*C, T, F) for dual-IFO
    """
    if "H1" not in batch or "L1" not in batch:
        raise KeyError("Batch must contain both 'H1' and 'L1' tensors.")

    h1 = batch["H1"]
    l1 = batch["L1"]

    if not torch.is_tensor(h1):
        h1 = torch.as_tensor(h1)
    if not torch.is_tensor(l1):
        l1 = torch.as_tensor(l1)

    if h1.dim() != 4 or l1.dim() != 4:
        raise ValueError(f"Expected 4D tensors (B,C,T,F), got H1={h1.shape}, L1={l1.shape}")

    if h1.shape != l1.shape:
        raise ValueError(f"H1/L1 shape mismatch: H1={h1.shape}, L1={l1.shape}")

    return torch.cat([h1, l1], dim=1).float()
