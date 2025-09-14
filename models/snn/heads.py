"""
SNN classification heads and readout layers.

This module contains classification heads extracted for 
enhanced modularity according to the refactoring plan.

Created for finer modularity beyond the initial split.
"""

import logging
from typing import Optional, Tuple
import jax.numpy as jnp
import flax.linen as nn

logger = logging.getLogger(__name__)


class SNNReadout(nn.Module):
    """
    Readout layer for converting spike trains to features.
    
    Handles temporal pooling and feature extraction from spikes.
    """
    
    pooling_strategy: str = 'temporal_mean'  # 'temporal_mean', 'temporal_max', 'adaptive'
    use_attention: bool = False
    attention_heads: int = 4
    
    def setup(self):
        """Initialize readout components."""
        if self.use_attention:
            self.attention = nn.MultiHeadDotProductAttention(
                num_heads=self.attention_heads,
                name='spike_attention'
            )
    
    def __call__(self, spike_trains: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Convert spike trains to readout features.
        
        Args:
            spike_trains: Spike trains [batch, time, neurons]
            training: Training mode flag
            
        Returns:
            Readout features [batch, features]
        """
        batch_size, time_steps, num_neurons = spike_trains.shape
        
        # Apply attention if enabled
        if self.use_attention:
            attended_spikes = self.attention(
                spike_trains, 
                deterministic=not training
            )
            # Residual connection
            spike_trains = spike_trains + attended_spikes
        
        # Temporal pooling
        if self.pooling_strategy == 'temporal_mean':
            # Average pooling over time
            readout_features = jnp.mean(spike_trains, axis=1)
            
        elif self.pooling_strategy == 'temporal_max':
            # Max pooling over time
            readout_features = jnp.max(spike_trains, axis=1)
            
        elif self.pooling_strategy == 'adaptive':
            # Adaptive pooling: combine mean and max
            mean_features = jnp.mean(spike_trains, axis=1)
            max_features = jnp.max(spike_trains, axis=1)
            std_features = jnp.std(spike_trains, axis=1)
            
            # Concatenate different statistics
            readout_features = jnp.concatenate([
                mean_features,
                max_features * 0.5,  # Scale down max
                std_features * 0.3   # Scale down std
            ], axis=-1)
            
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return readout_features


class ClassificationHead(nn.Module):
    """
    Classification head for SNN outputs.
    
    Converts readout features to class logits.
    """
    
    num_classes: int
    hidden_dims: Tuple[int, ...] = ()
    use_layer_norm: bool = True
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, features: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Apply classification head.
        
        Args:
            features: Input features [batch, feature_dim]
            training: Training mode flag
            
        Returns:
            Class logits [batch, num_classes]
        """
        x = features
        
        # Optional hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dim, name=f'hidden_{i+1}')(x)
            
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f'norm_{i+1}')(x)
            
            x = nn.relu(x)
            
            # Dropout
            if training and self.dropout_rate > 0:
                x = nn.Dropout(
                    rate=self.dropout_rate,
                    name=f'dropout_{i+1}'
                )(x, deterministic=not training)
        
        # Final classification layer
        logits = nn.Dense(self.num_classes, name='classification')(x)
        
        return logits


class MultiScaleReadout(nn.Module):
    """
    Multi-scale readout for capturing different temporal patterns.
    
    Combines readouts at different temporal scales.
    """
    
    scales: Tuple[int, ...] = (1, 2, 4, 8)  # Different temporal scales
    pooling_strategy: str = 'temporal_mean'
    
    def setup(self):
        """Initialize multi-scale readouts."""
        self.readouts = []
        
        for scale in self.scales:
            readout = SNNReadout(
                pooling_strategy=self.pooling_strategy,
                name=f'readout_scale_{scale}'
            )
            self.readouts.append(readout)
        
        # Fusion layer
        # Note: This would need proper sizing based on actual usage
        self.fusion = nn.Dense(256, name='scale_fusion')  # Placeholder size
    
    def __call__(self, spike_trains: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Apply multi-scale readout.
        
        Args:
            spike_trains: Spike trains [batch, time, neurons]
            training: Training mode flag
            
        Returns:
            Multi-scale readout features
        """
        scale_features = []
        
        for scale, readout in zip(self.scales, self.readouts):
            # Downsample spike trains for this scale
            if scale > 1:
                # Simple downsampling by averaging
                seq_len = spike_trains.shape[1]
                new_len = seq_len // scale
                
                if new_len > 0:
                    # Reshape and average
                    downsampled = spike_trains[:, :new_len*scale, :].reshape(
                        spike_trains.shape[0], new_len, scale, spike_trains.shape[2]
                    )
                    scale_spikes = jnp.mean(downsampled, axis=2)
                else:
                    # Fallback: use original
                    scale_spikes = spike_trains
            else:
                scale_spikes = spike_trains
            
            # Apply readout for this scale
            scale_feature = readout(scale_spikes, training=training)
            scale_features.append(scale_feature)
        
        # Concatenate features from different scales
        if scale_features:
            combined_features = jnp.concatenate(scale_features, axis=-1)
            
            # Fusion layer to combine scales
            fused_features = self.fusion(combined_features)
            
            return fused_features
        else:
            # Fallback
            return jnp.mean(spike_trains, axis=1)


# Export SNN head components
__all__ = [
    "SNNReadout",
    "ClassificationHead",
    "MultiScaleReadout"
]

