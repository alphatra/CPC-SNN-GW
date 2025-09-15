"""
Core SNN classifier implementations.

This module contains main SNN classifier classes extracted from
snn_classifier.py for better modularity:
- SNNClassifier: Basic SNN for binary classification
- EnhancedSNNClassifier: Advanced SNN with enhanced features

Split from snn_classifier.py for better maintainability.
"""

import logging
from typing import Optional, Dict, Any

import jax
import jax.numpy as jnp
import flax.linen as nn

from .config import SNNConfig, EnhancedSNNConfig
from .layers import LIFLayer, VectorizedLIFLayer, EnhancedLIFWithMemory

logger = logging.getLogger(__name__)


class SNNClassifier(nn.Module):
    """
    Spiking Neural Network classifier for gravitational wave detection.
    
    Multi-layer SNN with LIF neurons for energy-efficient classification:
    - Configurable depth (default: 3 layers)
    - Binary classification (noise vs GW signal)  
    - LayerNorm for gradient stability
    - Adaptive dropout per layer
    """
    
    hidden_size: int = 128
    num_classes: int = 2  # ✅ FIXED: Binary classification
    num_layers: int = 3   # ✅ Increased depth for better capacity
    
    @nn.compact
    def __call__(self, spikes: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Classify spike trains using multi-layer SNN.
        
        Args:
            spikes: Input spike trains [batch_size, time_steps, input_features]
            training: Training mode flag
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        batch_size, time_steps, input_features = spikes.shape
        
        # ✅ ARCHITECTURE: 3-layer SNN (256-128-64)
        layer_sizes = [256, 128, 64]
        current_spikes = spikes
        
        # ✅ INPUT PROJECTION: Project to first layer size
        if input_features != layer_sizes[0]:
            projection = nn.Dense(layer_sizes[0], name='input_projection')
            current_spikes = projection(current_spikes)
        
        # ✅ SNN LAYERS: Process through LIF layers
        for i in range(self.num_layers):
            layer_size = layer_sizes[min(i, len(layer_sizes)-1)]
            
            # Create LIF layer
            lif_layer = LIFLayer(
                features=layer_size,
                tau_mem=20e-3,
                tau_syn=5e-3,
                threshold=1.0,
                surrogate_beta=10.0 + i * 5.0,  # ✅ Adaptive: increases with depth
                name=f'lif_layer_{i+1}'
            )
            
            # Process spikes
            current_spikes = lif_layer(current_spikes, training=training)
            
            # ✅ NORMALIZATION: LayerNorm after each layer for stability
            # Apply LayerNorm to temporal mean (reduce time dimension)
            temporal_mean = jnp.mean(current_spikes, axis=1)  # [batch, features]
            normalized_mean = nn.LayerNorm(name=f'layer_norm_{i+1}')(temporal_mean)
            
            # Broadcast back to temporal dimension
            current_spikes = current_spikes * (normalized_mean[:, None, :] / (jnp.mean(current_spikes, axis=1, keepdims=True) + 1e-8))
            
            # ✅ DROPOUT: Adaptive dropout (decreases with depth)
            dropout_rate = max(0.0, 0.2 - i * 0.1)  # ✅ Decreases: 0.2 → 0.1 → 0.0
            if training and dropout_rate > 0:
                current_spikes = nn.Dropout(
                    rate=dropout_rate,
                    name=f'dropout_{i+1}'
                )(current_spikes, deterministic=not training)
        
        # ✅ READOUT: Convert spikes to logits
        # Temporal pooling: average over time
        temporal_features = jnp.mean(current_spikes, axis=1)  # [batch, features]
        
        # Classification head
        logits = nn.Dense(
            self.num_classes,
            name='classification_head'
        )(temporal_features)
        
        return logits


class EnhancedSNNClassifier(nn.Module):
    """
    Enhanced SNN classifier with advanced features.
    
    Advanced features:
    - Configurable architecture through SNNConfig
    - Enhanced LIF layers with memory
    - Attention mechanisms (optional)
    - Adaptive thresholds
    - Advanced readout strategies
    """
    
    config: EnhancedSNNConfig
    
    def setup(self):
        """Initialize enhanced SNN components."""
        # ✅ LAYERS: Create enhanced LIF layers
        self.snn_layers = []
        
        layer_sizes = list(self.config.hidden_sizes)
        
        for i in range(len(layer_sizes)):
            # Enhanced LIF with memory
            lif_layer = EnhancedLIFWithMemory(
                features=layer_sizes[i],
                tau_mem=self.config.tau_mem,
                tau_syn=self.config.tau_syn,
                threshold=self.config.threshold,
                use_long_term_memory=self.config.use_long_term_memory,
                memory_decay=self.config.memory_decay,
                use_adaptive_threshold=self.config.use_adaptive_threshold,
                threshold_adaptation_rate=self.config.threshold_adaptation_rate,
                surrogate_beta=self.config.surrogate_beta + i * 2.0,  # Adaptive beta
                name=f'enhanced_lif_{i+1}'
            )
            self.snn_layers.append(lif_layer)
        
        # ✅ ATTENTION: Optional attention mechanism
        if self.config.use_attention:
            self.attention_layer = nn.MultiHeadDotProductAttention(
                num_heads=self.config.attention_heads,
                name='snn_attention'
            )
        
        # ✅ READOUT: Advanced classification head
        self.classification_layers = nn.Sequential([
            nn.Dense(
                max(self.config.hidden_sizes[-1], 64),
                name='readout_dense1'
            ),
            nn.LayerNorm(name='readout_norm1') if self.config.use_layer_norm else lambda x: x,
            nn.relu,
            nn.Dense(self.config.num_classes, name='readout_final')
        ])
        
        logger.debug(f"EnhancedSNNClassifier setup: {len(self.snn_layers)} layers, "
                    f"attention={self.config.use_attention}")
    
    def __call__(self, spikes: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Enhanced SNN classification with advanced features.
        
        Args:
            spikes: Input spike trains [batch_size, time_steps, input_features]
            training: Training mode flag
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        batch_size, time_steps, input_features = spikes.shape
        current_spikes = spikes
        
        # ✅ INPUT PROJECTION: Project to first layer if needed
        if self.config.use_input_projection and input_features != self.config.hidden_sizes[0]:
            input_proj = nn.Dense(self.config.hidden_sizes[0], name='enhanced_input_projection')
            current_spikes = input_proj(current_spikes)
        
        # ✅ SNN PROCESSING: Process through enhanced LIF layers
        layer_outputs = []
        
        for i, lif_layer in enumerate(self.snn_layers):
            # Process through LIF layer
            current_spikes = lif_layer(current_spikes, training=training)
            layer_outputs.append(current_spikes)
            
            # ✅ NORMALIZATION: Layer normalization if enabled
            if self.config.use_layer_norm:
                # Normalize temporal statistics
                temporal_mean = jnp.mean(current_spikes, axis=1)
                temporal_norm = nn.LayerNorm(name=f'enhanced_norm_{i+1}')(temporal_mean)
                
                # Apply normalization
                norm_factor = temporal_norm / (jnp.mean(current_spikes, axis=1) + 1e-8)
                current_spikes = current_spikes * norm_factor[:, None, :]
            
            # ✅ DROPOUT: Layer-specific dropout
            if training and self.config.dropout_rate > 0:
                # Adaptive dropout rate per layer
                layer_dropout = self.config.dropout_rate * (1.0 - i / len(self.snn_layers))
                current_spikes = nn.Dropout(
                    rate=layer_dropout,
                    name=f'enhanced_dropout_{i+1}'
                )(current_spikes, deterministic=not training)
        
        # ✅ ATTENTION: Optional attention processing
        if self.config.use_attention and hasattr(self, 'attention_layer'):
            # Apply attention across time dimension
            attended_spikes = self.attention_layer(
                current_spikes, 
                deterministic=not training
            )
            current_spikes = current_spikes + attended_spikes  # Residual connection
        
        # ✅ READOUT: Advanced temporal readout
        # Multiple readout strategies
        temporal_mean = jnp.mean(current_spikes, axis=1)  # Average pooling
        temporal_max = jnp.max(current_spikes, axis=1)    # Max pooling
        temporal_std = jnp.std(current_spikes, axis=1)    # Variance information
        
        # Combine readout features
        readout_features = jnp.concatenate([
            temporal_mean,
            temporal_max * 0.5,     # Scale down max features
            temporal_std * 0.3      # Scale down variance features
        ], axis=-1)
        
        # ✅ CLASSIFICATION: Final classification
        logits = self.classification_layers(readout_features)

        # ✅ FIX: Spike rate regularization should be handled in the loss, not logits
        # Any regularization based on spike statistics must be added to the training
        # objective externally (e.g., snn_combined_loss), not mixed into logits.
        return logits


# Export core classifier classes
__all__ = [
    "SNNClassifier",
    "EnhancedSNNClassifier"
]

