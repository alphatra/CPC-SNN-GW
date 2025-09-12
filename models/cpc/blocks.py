"""
CPC building blocks and layers.

This module contains reusable building blocks extracted for 
enhanced modularity according to the refactoring plan.

Created for finer modularity beyond the initial split.
"""

import logging
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """
    Convolutional block for CPC encoder.
    
    Standard building block with conv + norm + activation.
    """
    
    features: int
    kernel_size: Tuple[int, ...] = (7,)
    strides: Tuple[int, ...] = (2,)
    use_layer_norm: bool = True
    activation: str = 'relu'
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Apply convolutional block."""
        # Convolution
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='SAME'
        )(x)
        
        # Layer normalization
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        
        # Activation
        if self.activation == 'relu':
            x = nn.relu(x)
        elif self.activation == 'gelu':
            x = nn.gelu(x)
        elif self.activation == 'tanh':
            x = nn.tanh(x)
        
        return x


class GRUContext(nn.Module):
    """
    GRU-based context network for CPC.
    
    Processes temporal sequences for context modeling.
    """
    
    features: int
    use_layer_norm: bool = True
    
    def setup(self):
        """Initialize GRU context components."""
        self.gru_cell = nn.GRUCell(features=self.features)
        
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm()
    
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Process sequence through GRU context.
        
        Args:
            x: Input sequence [batch, time, features]
            training: Training mode flag
            
        Returns:
            Context representations [batch, time, features]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Initialize carry state
        initial_carry = self.gru_cell.initialize_carry(
            jax.random.PRNGKey(0),
            (batch_size, self.features)
        )
        
        # Process sequence
        carry = initial_carry
        outputs = []
        
        for t in range(seq_len):
            carry, output = self.gru_cell(carry, x[:, t, :])
            outputs.append(output)
        
        # Stack outputs
        context_outputs = jnp.stack(outputs, axis=1)
        
        # Layer normalization
        if self.use_layer_norm:
            context_outputs = self.layer_norm(context_outputs)
        
        return context_outputs


class ProjectionHead(nn.Module):
    """
    Projection head for CPC encoders.
    
    Projects features to contrastive learning space.
    """
    
    latent_dim: int
    hidden_dim: Optional[int] = None
    use_layer_norm: bool = True
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Apply projection head."""
        # Optional intermediate projection
        if self.hidden_dim:
            x = nn.Dense(self.hidden_dim)(x)
            
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            
            x = nn.relu(x)
            
            # Dropout
            if training and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        
        # Final projection
        x = nn.Dense(self.latent_dim)(x)
        
        # Final layer norm
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        
        return x


class FeatureEncoder(nn.Module):
    """
    Feature encoder stack for CPC.
    
    Stack of convolutional blocks for feature extraction.
    """
    
    layer_dims: Tuple[int, ...] = (64, 128, 256)
    kernel_sizes: Tuple[int, ...] = (7, 5, 3)
    strides: Tuple[int, ...] = (2, 2, 1)
    
    def setup(self):
        """Initialize encoder blocks."""
        self.conv_blocks = []
        
        for i, (dim, kernel, stride) in enumerate(zip(self.layer_dims, self.kernel_sizes, self.strides)):
            block = ConvBlock(
                features=dim,
                kernel_size=(kernel,),
                strides=(stride,),
                name=f'conv_block_{i+1}'
            )
            self.conv_blocks.append(block)
    
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Apply feature encoder stack."""
        for block in self.conv_blocks:
            x = block(x, training=training)
        
        return x


# Export CPC building blocks
__all__ = [
    "ConvBlock",
    "GRUContext", 
    "ProjectionHead",
    "FeatureEncoder"
]
