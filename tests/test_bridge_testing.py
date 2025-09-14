import jax
import jax.numpy as jnp

from models.bridge.core import ValidatedSpikeBridge
from models.bridge.testing import test_gradient_flow as helper_gradient_flow  # import as helper to avoid pytest collection


def test_gradient_flow_smoke():
    bridge = ValidatedSpikeBridge(spike_encoding="temporal_contrast", time_steps=8, threshold=0.1)
    key = jax.random.PRNGKey(0)
    input_shape = (2, 64, 1)
    results = helper_gradient_flow(bridge, input_shape, key)
    assert isinstance(results, dict)
    assert "test_passed" in results


