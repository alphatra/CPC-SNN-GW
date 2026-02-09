import torch

from src.models.cpc_snn import CPCSNN


def test_cpcsnn_forward_timeseries_smoke():
    model = CPCSNN(
        in_channels=2,
        hidden_dim=32,
        context_dim=32,
        prediction_steps=4,
        use_metal=False,
        use_tf2d=False,
    )
    x = torch.randn(3, 2, 1024)
    logits, c, z = model(x)
    assert logits.shape == (3, 1)
    assert c.shape[0] == 3
    assert z.shape[0] == 3


def test_cpcsnn_forward_tf2d_smoke():
    model = CPCSNN(
        in_channels=8,
        hidden_dim=32,
        context_dim=32,
        prediction_steps=4,
        use_metal=False,
        use_tf2d=True,
    )
    x = torch.randn(2, 8, 31, 246)
    logits, c, z = model(x)
    assert logits.shape == (2, 1)
    assert c.shape[0] == 2
    assert z.shape[0] == 2
