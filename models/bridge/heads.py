"""
Spike bridge output projections and heads.

This module contains output projection components for spike bridge
created for enhanced modularity according to the refactoring plan.

Created for finer modularity beyond the initial split.
"""

import logging
from typing import Optional
import jax.numpy as jnp
import flax.linen as nn

logger = logging.getLogger(__name__)


class SpikeProjectionHead(nn.Module):
    """
    Projection head for spike bridge outputs.
    
    Projects spike trains to desired output space.
    """
    
    output_dim: int
    use_temporal_pooling: bool = True
    pooling_strategy: str = 'adaptive'  # 'mean', 'max', 'adaptive'
    use_layer_norm: bool = True
    
    @nn.compact
    def __call__(self, spike_trains: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Project spike trains to output space.
        
        Args:
            spike_trains: Input spike trains [batch, time, neurons]
            training: Training mode flag
            
        Returns:
            Projected features [batch, output_dim]
        """
        # Temporal pooling if enabled
        if self.use_temporal_pooling:
            if self.pooling_strategy == 'mean':
                pooled = jnp.mean(spike_trains, axis=1)
            elif self.pooling_strategy == 'max':
                pooled = jnp.max(spike_trains, axis=1)
            elif self.pooling_strategy == 'adaptive':
                # Combine multiple statistics
                mean_pool = jnp.mean(spike_trains, axis=1)
                max_pool = jnp.max(spike_trains, axis=1)
                std_pool = jnp.std(spike_trains, axis=1)
                
                pooled = jnp.concatenate([
                    mean_pool,
                    max_pool * 0.5,
                    std_pool * 0.3
                ], axis=-1)
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        else:
            # Flatten temporal dimension
            batch_size = spike_trains.shape[0]
            pooled = spike_trains.reshape(batch_size, -1)
        
        # Projection to output dimension
        projected = nn.Dense(self.output_dim)(pooled)
        
        # Layer normalization
        if self.use_layer_norm:
            projected = nn.LayerNorm()(projected)
        
        return projected


class MultiChannelProjection(nn.Module):
    """
    Multi-channel projection for spike bridge outputs.
    
    Projects different channels/aspects of spike trains separately.
    """
    
    output_channels: int
    channel_dim: int
    use_channel_attention: bool = True
    
    def setup(self):
        """Initialize multi-channel components."""
        # Channel-wise projections
        self.channel_projections = []
        for i in range(self.output_channels):
            projection = nn.Dense(
                self.channel_dim,
                name=f'channel_projection_{i+1}'
            )
            self.channel_projections.append(projection)
        
        # Channel attention
        if self.use_channel_attention:
            self.channel_attention = nn.Dense(
                self.output_channels,
                name='channel_attention'
            )
    
    def __call__(self, spike_trains: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Apply multi-channel projection.
        
        Args:
            spike_trains: Input spike trains [batch, time, neurons]
            training: Training mode flag
            
        Returns:
            Multi-channel projections [batch, output_channels, channel_dim]
        """
        batch_size = spike_trains.shape[0]
        
        # Temporal pooling for channel processing
        pooled_spikes = jnp.mean(spike_trains, axis=1)  # [batch, neurons]
        
        # Apply channel projections
        channel_features = []
        for projection in self.channel_projections:
            channel_feat = projection(pooled_spikes)
            channel_features.append(channel_feat)
        
        # Stack channels
        multi_channel = jnp.stack(channel_features, axis=1)  # [batch, channels, channel_dim]
        
        # Channel attention if enabled
        if self.use_channel_attention:
            # Compute attention weights
            attention_input = jnp.mean(multi_channel, axis=-1)  # [batch, channels]
            attention_weights = jax.nn.softmax(
                self.channel_attention(attention_input), axis=-1
            )  # [batch, output_channels]
            
            # Apply attention (broadcasting)
            attended_features = multi_channel * attention_weights[..., None]
            
            return attended_features
        
        return multi_channel


class AdaptiveProjection(nn.Module):
    """
    Adaptive projection that adjusts based on input characteristics.
    
    Dynamically adapts projection based on spike statistics.
    """
    
    base_output_dim: int
    adaptation_factor: float = 0.1
    
    def setup(self):
        """Initialize adaptive components."""
        # Base projection
        self.base_projection = nn.Dense(self.base_output_dim)
        
        # Adaptation network
        self.adaptation_net = nn.Sequential([
            nn.Dense(64),
            nn.relu,
            nn.Dense(self.base_output_dim)
        ])
    
    def __call__(self, spike_trains: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Apply adaptive projection.
        
        Args:
            spike_trains: Input spike trains [batch, time, neurons]
            training: Training mode flag
            
        Returns:
            Adaptively projected features [batch, base_output_dim]
        """
        # Base features
        pooled_spikes = jnp.mean(spike_trains, axis=1)
        base_features = self.base_projection(pooled_spikes)
        
        # Compute adaptation based on spike statistics
        spike_stats = jnp.concatenate([
            jnp.mean(spike_trains, axis=(1, 2), keepdims=False),  # Mean rate
            jnp.std(spike_trains, axis=(1, 2), keepdims=False),   # Std rate
            jnp.max(spike_trains, axis=(1, 2), keepdims=False)    # Max rate
        ], axis=0)
        
        # Adaptation features
        adaptation = self.adaptation_net(spike_stats[None, :])  # Add batch dim
        adaptation = jnp.broadcast_to(adaptation, base_features.shape)
        
        # Combine base and adaptive features
        adapted_features = base_features + self.adaptation_factor * adaptation
        
        return adapted_features


# Export spike bridge head components
__all__ = [
    "SpikeProjectionHead", 
    "MultiChannelProjection",
    "AdaptiveProjection"
]

# Alias for compatibility with plan
SpikeProjectionHead = SpikeProjectionHead

