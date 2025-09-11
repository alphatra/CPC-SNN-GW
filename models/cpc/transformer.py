"""
Transformer-based components for CPC encoders.

This module contains transformer implementations extracted from
cpc_encoder.py for better modularity:
- TemporalTransformerCPC: Multi-scale temporal transformer

Split from cpc_encoder.py for better maintainability.
"""

import logging
from typing import Dict, Any, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from .config import TemporalTransformerConfig

logger = logging.getLogger(__name__)


class TemporalTransformerCPC(nn.Module):
    """
    🚀 ENHANCED: Advanced Temporal Transformer for CPC.
    
    Replaces simple Dense temporal processor with sophisticated architecture:
    - Multi-scale temporal convolutions (1, 3, 5, 7 time steps)
    - Self-attention for long-range dependencies  
    - Residual connections and layer normalization
    - Optimized for gravitational wave temporal patterns
    """
    transformer_config: TemporalTransformerConfig
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Dict[str, jnp.ndarray]:
        """
        Enhanced temporal modeling with multi-scale attention.
        
        Args:
            x: Input features [batch, time, features]
            training: Training mode flag
            
        Returns:
            Dictionary with processed features and attention weights
        """
        batch_size, seq_len, feature_dim = x.shape
        
        # Multi-scale temporal convolutions
        multi_scale_outputs = []
        for kernel_size in self.transformer_config.multi_scale_kernels:
            conv_output = nn.Conv(
                features=feature_dim,
                kernel_size=(kernel_size,),
                padding='SAME',
                name=f'multi_scale_conv_{kernel_size}'
            )(x)
            
            if self.transformer_config.use_layer_norm:
                conv_output = nn.LayerNorm(name=f'layer_norm_conv_{kernel_size}')(conv_output)
            
            conv_output = nn.relu(conv_output)
            
            if training and self.transformer_config.dropout_rate > 0:
                conv_output = nn.Dropout(
                    rate=self.transformer_config.dropout_rate,
                    name=f'dropout_conv_{kernel_size}'
                )(conv_output, deterministic=not training)
            
            multi_scale_outputs.append(conv_output)
        
        # ✅ FUSION: Combine multi-scale outputs
        if len(multi_scale_outputs) > 1:
            # Concatenate different scales
            combined_features = jnp.concatenate(multi_scale_outputs, axis=-1)
            
            # Project back to original feature dimension
            combined_features = nn.Dense(
                feature_dim,
                name='scale_fusion_projection'
            )(combined_features)
        else:
            combined_features = multi_scale_outputs[0]
        
        # ✅ ATTENTION: Self-attention layers
        current_features = combined_features
        attention_weights = []
        
        for layer_idx in range(self.transformer_config.num_layers):
            # Multi-head self-attention
            attention_output = nn.MultiHeadDotProductAttention(
                num_heads=self.transformer_config.num_heads,
                dropout_rate=self.transformer_config.attention_dropout if training else 0.0,
                name=f'attention_layer_{layer_idx}'
            )(current_features, deterministic=not training)
            
            # Residual connection
            if self.transformer_config.use_residual_connections:
                attention_output = current_features + attention_output
            
            # Layer normalization
            if self.transformer_config.use_layer_norm:
                attention_output = nn.LayerNorm(name=f'attention_norm_{layer_idx}')(attention_output)
            
            # Feed-forward network
            ff_output = nn.Sequential([
                nn.Dense(self.transformer_config.feed_forward_dim, name=f'ff_dense1_{layer_idx}'),
                nn.relu,
                nn.Dense(feature_dim, name=f'ff_dense2_{layer_idx}')
            ])(attention_output)
            
            # Another residual connection
            if self.transformer_config.use_residual_connections:
                ff_output = attention_output + ff_output
            
            # Final layer norm for this transformer layer
            if self.transformer_config.use_layer_norm:
                ff_output = nn.LayerNorm(name=f'ff_norm_{layer_idx}')(ff_output)
            
            # Apply dropout
            if training and self.transformer_config.dropout_rate > 0:
                ff_output = nn.Dropout(
                    rate=self.transformer_config.dropout_rate,
                    name=f'transformer_dropout_{layer_idx}'
                )(ff_output, deterministic=not training)
            
            # Store attention weights for analysis
            # Note: In real implementation, we'd extract these from attention layer
            attention_weights.append(jnp.mean(jnp.abs(attention_output), axis=(0, 1)))
            
            current_features = ff_output
        
        # ✅ OUTPUT: Return processed features with metadata
        return {
            'features': current_features,
            'attention_weights': attention_weights,
            'multi_scale_features': multi_scale_outputs
        }


# Export transformer components
__all__ = [
    "TemporalTransformerCPC"
]

