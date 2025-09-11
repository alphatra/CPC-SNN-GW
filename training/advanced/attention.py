"""
Attention-based CPC encoder for advanced training.

This module contains the AttentionCPCEncoder class extracted from
advanced_training.py for better modularity.

Split from advanced_training.py for better maintainability.
"""

import logging
import jax
import jax.numpy as jnp
import flax.linen as nn

logger = logging.getLogger(__name__)


class AttentionCPCEncoder(nn.Module):
    """
    CPC encoder with multi-head self-attention for enhanced representation learning.
    Executive Summary implementation: attention-enhanced CPC.
    """
    
    latent_dim: int = 256
    num_heads: int = 8
    dropout_rate: float = 0.1
    
    def setup(self):
        """Initialize attention CPC encoder components."""
        # Convolutional feature extraction
        self.conv_stack = [
            nn.Conv(64, kernel_size=(7,), strides=(2,), padding='SAME'),
            nn.LayerNorm(),
            nn.Conv(128, kernel_size=(5,), strides=(2,), padding='SAME'), 
            nn.LayerNorm(),
            nn.Conv(self.latent_dim, kernel_size=(3,), strides=(1,), padding='SAME'),
            nn.LayerNorm()
        ]
        
        # Multi-head self-attention for temporal modeling
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=jnp.float32,
            dropout_rate=self.dropout_rate,
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform()
        )
        
        # Position encoding for temporal sequences
        self.pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (1, 512, self.latent_dim)  # Max sequence length = 512
        )
        
        # Final projection layer
        self.projection = nn.Dense(
            self.latent_dim,
            kernel_init=nn.initializers.xavier_uniform()
        )
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass with attention-enhanced temporal modeling.
        
        Args:
            x: Input sequences [batch_size, sequence_length, input_dim]
            training: Training mode flag
            
        Returns:
            Attention-enhanced features [batch_size, time_steps, latent_dim]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # ✅ FEATURE EXTRACTION: Convolutional feature extraction
        features = x
        for layer in self.conv_stack:
            if isinstance(layer, nn.LayerNorm):
                features = layer(features)
            else:
                features = nn.relu(layer(features))
            
            # Apply dropout if in training mode
            if training and self.dropout_rate > 0:
                features = nn.Dropout(
                    rate=self.dropout_rate,
                    deterministic=not training
                )(features)
        
        # ✅ POSITIONAL ENCODING: Add positional embeddings
        batch_size_updated, time_steps, feature_dim = features.shape
        
        # Trim or pad positional embeddings to match sequence length
        if time_steps <= self.pos_embedding.shape[1]:
            pos_emb = self.pos_embedding[:, :time_steps, :]
        else:
            # Repeat positional embeddings if sequence is longer
            repeats = (time_steps + self.pos_embedding.shape[1] - 1) // self.pos_embedding.shape[1]
            extended_pos_emb = jnp.tile(self.pos_embedding, (1, repeats, 1))
            pos_emb = extended_pos_emb[:, :time_steps, :]
        
        # Add positional encoding
        features_with_pos = features + pos_emb
        
        # ✅ ATTENTION: Multi-head self-attention
        # Apply attention across time dimension
        attended_features = self.attention(
            features_with_pos,
            deterministic=not training
        )
        
        # ✅ RESIDUAL: Residual connection
        features_with_attention = features_with_pos + attended_features
        
        # ✅ NORMALIZATION: Final layer normalization
        normalized_features = nn.LayerNorm()(features_with_attention)
        
        # ✅ PROJECTION: Final projection to latent space
        final_features = self.projection(normalized_features)
        
        # ✅ FINAL ACTIVATION: Apply activation and final dropout
        final_features = nn.tanh(final_features)
        
        if training and self.dropout_rate > 0:
            final_features = nn.Dropout(
                rate=self.dropout_rate * 0.5,  # Reduced dropout for final layer
                deterministic=not training
            )(final_features)
        
        return final_features
    
    def get_attention_weights(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Extract attention weights for visualization and analysis.
        
        Args:
            x: Input sequences
            
        Returns:
            Attention weights [batch_size, num_heads, time_steps, time_steps]
        """
        # This would require modifying the attention mechanism to return weights
        # For now, return placeholder
        batch_size, seq_len, _ = x.shape
        
        # Process through conv stack to get features
        features = x
        for layer in self.conv_stack:
            if isinstance(layer, nn.LayerNorm):
                features = layer(features)
            else:
                features = nn.relu(layer(features))
        
        # Get features shape after convolution
        time_steps = features.shape[1]
        
        # Return placeholder attention weights
        attention_weights = jnp.ones((batch_size, self.num_heads, time_steps, time_steps))
        attention_weights = attention_weights / jnp.sum(attention_weights, axis=-1, keepdims=True)
        
        return attention_weights


# Export attention encoder
__all__ = [
    "AttentionCPCEncoder"
]
