import torch

from src.train.trainer import compute_hard_negative_bce_loss, compute_tail_ranking_loss


def test_tail_ranking_loss_smoke():
    logits = torch.tensor([[2.0], [1.2], [0.5], [-0.2], [-1.0], [-2.5]])
    targets = torch.tensor([[1.0], [1.0], [0.0], [0.0], [0.0], [0.0]])
    loss = compute_tail_ranking_loss(
        logits, targets, hard_frac=0.5, hard_min=1, margin=0.1, max_pairs=128
    )
    assert torch.isfinite(loss).item()
    assert loss.item() >= 0.0


def test_hard_negative_bce_loss_smoke():
    logits = torch.tensor([[2.0], [1.2], [0.5], [-0.2], [-1.0], [-2.5]])
    targets = torch.tensor([[1.0], [1.0], [0.0], [0.0], [0.0], [0.0]])
    loss = compute_hard_negative_bce_loss(logits, targets, hard_frac=0.5, hard_min=1)
    assert torch.isfinite(loss).item()
    assert loss.item() >= 0.0
