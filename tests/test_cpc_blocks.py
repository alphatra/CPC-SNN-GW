"""
Tests for CPC building blocks.

Smoke tests for CPC modular components according to refactoring plan.
"""

import pytest
import jax
import jax.numpy as jnp
from models.cpc.blocks import ConvBlock, GRUContext, ProjectionHead, FeatureEncoder


def test_conv_block_shapes():
    """Test ConvBlock output shapes."""
    block = ConvBlock(features=64)
    key = jax.random.PRNGKey(0)
    x = jnp.ones((2, 256, 1))  # [batch, time, channels]
    
    params = block.init(key, x)
    out = block.apply(params, x)
    
    assert out.shape[0] == 2  # Batch preserved
    assert out.shape[-1] == 64  # Features correct
    assert jnp.all(jnp.isfinite(out))  # No NaNs/Infs


def test_gru_context_shapes():
    """Test GRUContext output shapes."""
    context = GRUContext(features=128)
    key = jax.random.PRNGKey(42)
    x = jnp.ones((1, 64, 256))  # [batch, time, features]
    
    params = context.init(key, x)
    out = context.apply(params, x)
    
    assert out.shape == (1, 64, 128)  # [batch, time, context_features]
    assert jnp.all(jnp.isfinite(out))


def test_projection_head_shapes():
    """Test ProjectionHead output shapes."""
    head = ProjectionHead(latent_dim=128)
    key = jax.random.PRNGKey(123)
    x = jnp.ones((4, 256))  # [batch, features]
    
    params = head.init(key, x)
    out = head.apply(params, x)
    
    assert out.shape == (4, 128)  # [batch, latent_dim]
    assert jnp.all(jnp.isfinite(out))


def test_feature_encoder_regression():
    """Basic regression test for FeatureEncoder."""
    encoder = FeatureEncoder(layer_dims=(32, 64, 128))
    key = jax.random.PRNGKey(999)
    x = jnp.ones((1, 512, 1))  # [batch, sequence, input_dim]
    
    params = encoder.init(key, x)
    out = encoder.apply(params, x)
    
    # Basic checks
    assert out.shape[0] == 1  # Batch preserved
    assert out.shape[-1] == 128  # Final feature dim
    assert jnp.all(jnp.isfinite(out))
    
    # Regression: output should not be all zeros
    assert jnp.abs(jnp.mean(out)) > 1e-6

