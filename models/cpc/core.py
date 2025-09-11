"""
Core CPC encoder implementations.

This module contains the main CPC encoder classes extracted from
cpc_encoder.py for better modularity:
- CPCEncoder: Basic CPC implementation
- RealCPCEncoder: Production-ready CPC with advanced features
- EnhancedCPCEncoder: Advanced CPC with transformer components

Split from cpc_encoder.py for better maintainability.
"""

import logging
from typing import Optional, Dict, Any, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from .config import RealCPCConfig, ExperimentConfig
from .transformer import TemporalTransformerCPC
from ..cpc_components import RMSNorm, WeightNormDense

logger = logging.getLogger(__name__)


class CPCEncoder(nn.Module):
    """
    Basic Contrastive Predictive Coding encoder for gravitational wave signals.
    
    Implements standard CPC architecture with:
    - Temporal convolution encoder
    - Context network (GRU-based)
    - Prediction network for InfoNCE loss
    """
    
    latent_dim: int = 256
    context_length: int = 32
    prediction_steps: int = 8
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Encode input sequences into latent representations.
        
        Args:
            x: Input sequences [batch_size, sequence_length, input_dim]
            training: Training mode flag
            
        Returns:
            Latent features [batch_size, time_steps, latent_dim]
        """
        batch_size, sequence_length, input_dim = x.shape
        
        # ✅ ENCODER: Temporal convolution layers
        # Conv layer 1: capture local temporal patterns
        conv1 = nn.Conv(
            features=64,
            kernel_size=(7,),
            strides=(2,),
            padding='SAME',
            name='encoder_conv1'
        )(x)
        conv1 = nn.relu(conv1)
        
        # Conv layer 2: capture medium-range dependencies  
        conv2 = nn.Conv(
            features=128,
            kernel_size=(5,),
            strides=(2,),
            padding='SAME',
            name='encoder_conv2'
        )(conv1)
        conv2 = nn.relu(conv2)
        
        # Conv layer 3: capture long-range patterns
        conv3 = nn.Conv(
            features=self.latent_dim,
            kernel_size=(3,),
            strides=(2,),
            padding='SAME',
            name='encoder_conv3'
        )(conv2)
        conv3 = nn.relu(conv3)
        
        # ✅ CONTEXT: Context network using GRU
        context_cell = nn.GRUCell(features=self.latent_dim, name='context_gru')
        
        # Initialize context state
        batch_size = conv3.shape[0]
        initial_carry = context_cell.initialize_carry(
            jax.random.PRNGKey(0),
            (batch_size, self.latent_dim)
        )
        
        # Process temporal sequence
        carry = initial_carry
        context_outputs = []
        
        for t in range(conv3.shape[1]):
            carry, output = context_cell(carry, conv3[:, t, :])
            context_outputs.append(output)
        
        # Stack context outputs
        context_features = jnp.stack(context_outputs, axis=1)
        
        # ✅ PREDICTION: Project to prediction space
        prediction_features = nn.Dense(
            self.latent_dim,
            name='prediction_projection'
        )(context_features)
        
        return prediction_features


class RealCPCEncoder(nn.Module):
    """
    Production-ready CPC encoder with advanced features.
    
    Enhanced version with:
    - Configurable architecture
    - Advanced normalization
    - Residual connections
    - Attention mechanisms
    - Gradient flow optimization
    """
    
    config: RealCPCConfig
    
    def setup(self):
        """Initialize CPC encoder components."""
        # ✅ ENCODER: Multi-layer temporal encoder
        self.encoder_layers = []
        
        layer_dims = [64, 128, 256, self.config.latent_dim]
        kernel_sizes = [7, 5, 3, 3]
        strides = [2, 2, 1, 1]
        
        for i in range(self.config.num_layers):
            layer = nn.Conv(
                features=layer_dims[min(i, len(layer_dims)-1)],
                kernel_size=(kernel_sizes[min(i, len(kernel_sizes)-1)],),
                strides=(strides[min(i, len(strides)-1)],),
                padding='SAME',
                name=f'encoder_conv_{i+1}'
            )
            self.encoder_layers.append(layer)
        
        # ✅ CONTEXT: Advanced context network
        if self.config.use_attention:
            # Use transformer for context (if enabled)
            from .transformer import TemporalTransformerCPC, TemporalTransformerConfig
            transformer_config = TemporalTransformerConfig(
                num_heads=4,
                num_layers=2,
                dropout_rate=self.config.dropout_rate
            )
            self.context_network = TemporalTransformerCPC(transformer_config)
        else:
            # Use GRU for context
            self.context_network = nn.GRUCell(
                features=self.config.context_dim,
                name='context_gru'
            )
        
        # ✅ PREDICTION: Prediction head with normalization
        self.prediction_head = nn.Sequential([
            nn.Dense(self.config.hidden_dim, name='pred_dense1'),
            nn.LayerNorm(name='pred_norm1') if self.config.use_layer_normalization else lambda x: x,
            nn.relu,
            nn.Dense(self.config.latent_dim, name='pred_dense2'),
            nn.LayerNorm(name='pred_norm2') if self.config.use_layer_normalization else lambda x: x
        ])
        
        # ✅ PROJECTION: Final projection for InfoNCE
        self.projection_head = nn.Dense(
            self.config.latent_dim,
            name='projection_head'
        )
        
        logger.debug(f"RealCPCEncoder initialized: latent_dim={self.config.latent_dim}, "
                    f"context_dim={self.config.context_dim}, attention={self.config.use_attention}")
    
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Advanced CPC encoding with configurable architecture.
        
        Args:
            x: Input sequences [batch_size, sequence_length, input_dim]
            training: Training mode flag
            
        Returns:
            Latent representations [batch_size, time_steps, latent_dim]
        """
        # ✅ INPUT VALIDATION
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D input (batch, sequence, features), got shape {x.shape}")
        
        current_features = x
        
        # ✅ ENCODER: Progressive feature extraction
        for i, encoder_layer in enumerate(self.encoder_layers):
            current_features = encoder_layer(current_features)
            
            # Apply normalization if enabled
            if self.config.use_layer_normalization:
                current_features = nn.LayerNorm(name=f'encoder_norm_{i+1}')(current_features)
            
            # Apply activation
            current_features = nn.relu(current_features)
            
            # Apply dropout if in training
            if training and self.config.dropout_rate > 0:
                current_features = nn.Dropout(
                    rate=self.config.dropout_rate,
                    name=f'encoder_dropout_{i+1}'
                )(current_features, deterministic=not training)
            
            # Residual connections (if enabled and dimensions match)
            if (self.config.use_residual_connections and i > 0 and 
                current_features.shape[-1] == getattr(self, f'_prev_features_{i}', current_features).shape[-1]):
                current_features = current_features + getattr(self, f'_prev_features_{i}')
            
            # Store for potential residual connection
            setattr(self, f'_prev_features_{i+1}', current_features)
        
        # ✅ CONTEXT: Process through context network
        if self.config.use_attention:
            # Transformer-based context
            context_output = self.context_network(current_features, training=training)
            if isinstance(context_output, dict):
                context_features = context_output['features']
            else:
                context_features = context_output
        else:
            # GRU-based context
            batch_size = current_features.shape[0]
            initial_carry = self.context_network.initialize_carry(
                jax.random.PRNGKey(0),
                (batch_size, self.config.context_dim)
            )
            
            carry = initial_carry
            context_outputs = []
            
            for t in range(current_features.shape[1]):
                carry, output = self.context_network(carry, current_features[:, t, :])
                context_outputs.append(output)
            
            context_features = jnp.stack(context_outputs, axis=1)
        
        # ✅ PREDICTION: Generate predictions for InfoNCE
        prediction_features = self.prediction_head(context_features)
        
        # ✅ PROJECTION: Final projection for contrastive learning
        final_features = self.projection_head(prediction_features)
        
        return final_features


class EnhancedCPCEncoder(nn.Module):
    """
    Enhanced CPC encoder with transformer and advanced features.
    
    Most advanced CPC implementation with:
    - Multi-head self-attention
    - Multi-scale temporal processing
    - Advanced normalization
    - Gradient flow optimization
    """
    
    config: ExperimentConfig
    
    def setup(self):
        """Initialize enhanced CPC components."""
        # ✅ MULTI-SCALE ENCODER: Different temporal scales
        self.multi_scale_encoders = {}
        
        scales = [1, 2, 4, 8]  # Different temporal downsampling factors
        for scale in scales:
            encoder = nn.Sequential([
                nn.Conv(
                    features=64,
                    kernel_size=(7,),
                    strides=(scale,),
                    padding='SAME',
                    name=f'scale_{scale}_conv1'
                ),
                nn.LayerNorm(name=f'scale_{scale}_norm1'),
                nn.relu,
                nn.Conv(
                    features=128,
                    kernel_size=(5,),
                    strides=(1,),
                    padding='SAME',
                    name=f'scale_{scale}_conv2'
                ),
                nn.LayerNorm(name=f'scale_{scale}_norm2'),
                nn.relu,
                nn.Conv(
                    features=self.config.latent_dim,
                    kernel_size=(3,),
                    strides=(1,),
                    padding='SAME',
                    name=f'scale_{scale}_conv3'
                )
            ])
            self.multi_scale_encoders[f'scale_{scale}'] = encoder
        
        # ✅ ATTENTION: Transformer for context modeling
        if self.config.use_transformer and self.config.transformer_config:
            self.transformer = TemporalTransformerCPC(self.config.transformer_config)
        
        # ✅ FUSION: Multi-scale feature fusion
        self.scale_fusion = nn.Dense(
            self.config.latent_dim,
            name='scale_fusion'
        )
        
        # ✅ PROJECTION: Final projection layers
        self.projection_layers = nn.Sequential([
            nn.Dense(self.config.hidden_dim, name='proj_dense1'),
            nn.LayerNorm(name='proj_norm1'),
            nn.relu,
            nn.Dense(self.config.latent_dim, name='proj_dense2'),
            nn.LayerNorm(name='proj_norm2')
        ])
    
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Enhanced CPC encoding with multi-scale processing.
        
        Args:
            x: Input sequences [batch_size, sequence_length, input_dim]
            training: Training mode flag
            
        Returns:
            Enhanced latent representations [batch_size, time_steps, latent_dim]
        """
        # ✅ MULTI-SCALE PROCESSING
        scale_features = []
        
        for scale_name, encoder in self.multi_scale_encoders.items():
            scale_feat = encoder(x)
            scale_features.append(scale_feat)
        
        # ✅ FUSION: Combine multi-scale features
        # Interpolate to common temporal resolution
        target_length = scale_features[0].shape[1]  # Use finest scale as target
        
        aligned_features = []
        for feat in scale_features:
            if feat.shape[1] != target_length:
                # Simple interpolation for temporal alignment
                aligned_feat = jax.image.resize(
                    feat,
                    shape=(feat.shape[0], target_length, feat.shape[2]),
                    method='linear'
                )
            else:
                aligned_feat = feat
            aligned_features.append(aligned_feat)
        
        # Concatenate and fuse
        concatenated = jnp.concatenate(aligned_features, axis=-1)
        fused_features = self.scale_fusion(concatenated)
        
        # ✅ ATTENTION: Process through transformer if enabled
        if self.config.use_transformer and hasattr(self, 'transformer'):
            transformer_output = self.transformer(fused_features, training=training)
            if isinstance(transformer_output, dict):
                attended_features = transformer_output['features']
            else:
                attended_features = transformer_output
        else:
            attended_features = fused_features
        
        # ✅ PROJECTION: Final projection
        final_features = self.projection_layers(attended_features)
        
        return final_features


# Export core encoder classes
__all__ = [
    "CPCEncoder",
    "RealCPCEncoder", 
    "EnhancedCPCEncoder"
]

