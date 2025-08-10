"""
CPC Components: Utility Neural Network Layers

Reusable components for Contrastive Predictive Coding architecture:
- RMSNorm: Root Mean Square Layer Normalization  
- WeightNormDense: Dense layer with weight normalization
- EquinoxGRUWrapper: Wrapper for Equinox GRU integration with Flax
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional
import logging

try:
    import equinox as eqx
    EQUINOX_AVAILABLE = True
except ImportError:
    EQUINOX_AVAILABLE = False
    eqx = None

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More stable than LayerNorm for long sequences and doesn't require mean centering.
    Particularly good for transformer-like architectures.
    """
    features: int
    eps: float = 1e-6
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply RMS normalization."""
        scale = self.param(
            'scale', 
            nn.initializers.ones, 
            (self.features,)
        )
        
        # Compute RMS
        mean_square = jnp.mean(x**2, axis=-1, keepdims=True)
        rms = jnp.sqrt(mean_square + self.eps)
        
        # Normalize and scale
        return (x / rms) * scale


class WeightNormDense(nn.Module):
    """Dense layer with weight normalization."""
    features: int
    use_bias: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply weight-normalized dense layer."""
        # Weight normalization parameters
        v = self.param(
            'v',
            nn.initializers.xavier_uniform(),
            (x.shape[-1], self.features)
        )
        g = self.param(
            'g',
            nn.initializers.ones,
            (self.features,)
        )
        
        # Weight normalization: W = g * v / ||v||
        v_norm = jnp.linalg.norm(v, axis=0, keepdims=True)
        w = g * v / (v_norm + 1e-8)
        
        # Apply linear transformation
        out = jnp.dot(x, w)
        
        if self.use_bias:
            bias = self.param(
                'bias',
                nn.initializers.zeros,
                (self.features,)
            )
            out = out + bias
        
        return out


class EquinoxGRUWrapper(nn.Module):
    """Wrapper for Equinox GRU that integrates with Flax."""
    hidden_size: int
    
    def setup(self):
        if not EQUINOX_AVAILABLE:
            logger.warning("Equinox not available, falling back to Flax GRU")
            self.use_equinox = False
        else:
            self.use_equinox = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, initial_state: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Apply GRU with improved scan compatibility."""
        
        if self.use_equinox and EQUINOX_AVAILABLE:
            # Use Equinox GRU (better for scan operations)
            gru_cell = eqx.nn.GRUCell(x.shape[-1], self.hidden_size, key=jax.random.PRNGKey(0))
            
            if initial_state is None:
                initial_state = jnp.zeros((x.shape[0], self.hidden_size))
            
            # Manual scan implementation for Equinox compatibility
            def gru_step(carry, x_t):
                h_t = gru_cell(x_t, carry)
                return h_t, h_t
            
            _, h_sequence = jax.lax.scan(gru_step, initial_state, x)
            return h_sequence
        
        else:
            # Fallback to Flax GRU
            if initial_state is None:
                initial_state = nn.GRUCell(self.hidden_size).initialize_carry(
                    jax.random.PRNGKey(0), x.shape[:-1]
                )
            
            gru_cell = nn.GRUCell(self.hidden_size)
            _, h_sequence = nn.scan(
                gru_cell,
                variable_broadcast="params",
                split_rngs={"params": False}
            )(initial_state, x)
            
            return h_sequence 