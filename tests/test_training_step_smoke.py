import torch

from src.models.cpc_snn import CPCSNN


def test_single_training_step_smoke():
    torch.manual_seed(0)
    model = CPCSNN(
        in_channels=2,
        hidden_dim=16,
        context_dim=16,
        prediction_steps=2,
        use_metal=False,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(4, 2, 512)
    y = torch.randint(0, 2, (4, 1)).float()

    logits, _, _ = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    assert torch.isfinite(loss).item()
