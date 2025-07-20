"""
Enhanced Contrastive Predictive Coding (CPC) Encoder

Self-supervised learning architecture for gravitational wave strain data.
Enhanced with findings from CPC+SNN Integration Paper (2025).

Key improvements:
- Modular component architecture via cpc_components
- Enhanced loss functions via cpc_losses  
- Configurable sizes via ExperimentConfig
- Enhanced numerical stability and gradient flow
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Tuple, Optional, Dict, Any, Union, List
from dataclasses import dataclass
import logging

# Import local components
from .cpc_components import RMSNorm, WeightNormDense, EquinoxGRUWrapper, EQUINOX_AVAILABLE
from .cpc_losses import enhanced_info_nce_loss, info_nce_loss

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for CPC encoder experiments."""
    # Model architecture
    latent_dim: int = 256
    conv_channels: Tuple[int, ...] = (32, 64, 128)
    conv_kernel_size: int = 9
    conv_stride: int = 2
    gru_hidden_size: int = 256
    
    # Regularization
    use_batch_norm: bool = True
    use_weight_norm: bool = True
    dropout_rate: float = 0.1
    
    # Training
    temperature: float = 0.1
    num_negatives: int = 8
    use_hard_negatives: bool = False
    
    # Data preprocessing
    input_scaling: float = 1e20  # Scale GW strain data
    sequence_length: int = 4096
    
    # Advanced features
    use_equinox_gru: bool = True
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True


class EnhancedCPCEncoder(nn.Module):
    """
    Enhanced CPC Encoder with advanced features.
    
    Features:
    - Equinox GRU integration for better scan compatibility
    - Weight normalization (RMSNorm) for stable training
    - Configurable architecture via ExperimentConfig
    - Gradient checkpointing for memory efficiency
    - Mixed precision training support
    """
    config: ExperimentConfig
    
    def setup(self):
        """Initialize encoder components."""
        # Validate GRU hidden size matches latent dimension
        if self.config.gru_hidden_size != self.config.latent_dim:
            logger.warning(
                f"GRU hidden size ({self.config.gru_hidden_size}) != latent dim ({self.config.latent_dim}). "
                f"Using latent_dim for consistency."
            )
            self.config.gru_hidden_size = self.config.latent_dim
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """
        Forward pass through enhanced CPC encoder.
        
        Args:
            x: Input strain data [batch, time]
            train: Training mode flag
            
        Returns:
            Latent representations [batch, time_downsampled, latent_dim]
        """
        # Preprocess input
        x = self._preprocess_input(x)
        
        # Convolution layers with downsampling
        x = self._apply_conv_layers(x, train)
        
        # Temporal modeling with GRU
        x = self._apply_recurrent_layer(x)
        
        # Final projection
        x = self._apply_final_projection(x)
        
        # L2 normalization for contrastive learning
        x = self._apply_l2_normalization(x)
        
        return x
    
    def _preprocess_input(self, x: jnp.ndarray) -> jnp.ndarray:
        """Preprocess input strain data."""
        # Scale input for numerical stability
        x = x * self.config.input_scaling
        
        # Add channel dimension for convolution
        x = x[..., None]  # [batch, time, 1]
        
        return x
    
    def _apply_conv_layers(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        """Apply convolutional feature extraction layers."""
        for i, channels in enumerate(self.config.conv_channels):
            # Convolution
            x = nn.Conv(
                features=channels,
                kernel_size=(self.config.conv_kernel_size,),
                strides=(self.config.conv_stride,),
                padding='SAME',
                name=f'conv_{i}'
            )(x)
            
            # Batch normalization or RMS normalization
            if self.config.use_batch_norm:
                if self.config.use_weight_norm:
                    x = RMSNorm(features=channels, name=f'rms_norm_{i}')(x)
                else:
                    x = nn.BatchNorm(use_running_average=not train, name=f'batch_norm_{i}')(x)
            
            # Activation
            x = nn.gelu(x)
            
            # Dropout
            if self.config.dropout_rate > 0:
                x = nn.Dropout(rate=self.config.dropout_rate, deterministic=not train)(x)
        
        # Remove channel dimension for RNN
        x = x.squeeze(-1)  # [batch, time_downsampled, channels]
        
        return x
    
    def _apply_recurrent_layer(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply recurrent layer for temporal modeling."""
        if self.config.use_equinox_gru and EQUINOX_AVAILABLE:
            # Use Equinox GRU wrapper
            if self.config.use_gradient_checkpointing:
                @nn.remat(prevent_cse=True)
                def checkpointed_gru(x):
                    return EquinoxGRUWrapper(
                        hidden_size=self.config.gru_hidden_size,
                        name='equinox_gru'
                    )(x)
                x = checkpointed_gru(x)
            else:
                x = EquinoxGRUWrapper(
                    hidden_size=self.config.gru_hidden_size,
                    name='equinox_gru'
                )(x)
        else:
            # Fallback to Flax GRU
            x = self._apply_flax_gru(x)
        
        return x
    
    def _apply_flax_gru(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply Flax GRU as fallback."""
        # Initialize GRU state
        gru_cell = nn.GRUCell(features=self.config.gru_hidden_size)
        carry = gru_cell.initialize_carry(jax.random.PRNGKey(0), x.shape[:-1])
        
        # Apply GRU with optional gradient checkpointing
        if self.config.use_gradient_checkpointing:
            @nn.remat(prevent_cse=True)
            def scan_fn(carry, x_t):
                carry, y = gru_cell(carry, x_t)
                return carry, y
        else:
            def scan_fn(carry, x_t):
                carry, y = gru_cell(carry, x_t)
                return carry, y
        
        # Apply scan
        _, outputs = jax.lax.scan(scan_fn, carry, x)
        
        return outputs
    
    def _apply_final_projection(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply final projection to latent dimension."""
        if self.config.use_weight_norm:
            # Weight normalized dense layer
            x = WeightNormDense(
                features=self.config.latent_dim,
                name='projection'
            )(x)
        else:
            # Standard dense layer
            x = nn.Dense(
                self.config.latent_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
                name='projection'
            )(x)
        
        return x
    
    def _apply_l2_normalization(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply L2 normalization for contrastive learning."""
        norms = jnp.linalg.norm(x, axis=-1, keepdims=True)
        x = jnp.where(
            norms > 1e-6,
            x / (norms + 1e-8),
            x  # Keep original if norm too small
        )
        return x


class CPCEncoder(nn.Module):
    """Backward compatible CPC encoder."""
    latent_dim: int = 256
    conv_channels: Tuple[int, ...] = (32, 64, 128)
    use_batch_norm: bool = False
    dropout_rate: float = 0.0
    
    def setup(self):
        """Convert to ExperimentConfig and use EnhancedCPCEncoder."""
        self.config = ExperimentConfig(
            latent_dim=self.latent_dim,
            conv_channels=self.conv_channels,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
            use_weight_norm=False,  # Disable for backward compatibility
            use_equinox_gru=False   # Disable for backward compatibility
        )
        self.enhanced_encoder = EnhancedCPCEncoder(config=self.config)
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Forward pass using enhanced encoder."""
        return self.enhanced_encoder(x, train)


# Factory functions
def create_enhanced_cpc_encoder(config: Optional[ExperimentConfig] = None) -> EnhancedCPCEncoder:
    """Create enhanced CPC encoder with configuration."""
    if config is None:
        config = ExperimentConfig()
    return EnhancedCPCEncoder(config=config)


def create_standard_cpc_encoder(latent_dim: int = 256,
                              conv_channels: Tuple[int, ...] = (32, 64, 128)) -> CPCEncoder:
    """Create standard CPC encoder for backward compatibility."""
    return CPCEncoder(
        latent_dim=latent_dim,
        conv_channels=conv_channels
    )


def create_experiment_config(
    latent_dim: int = 256,
    conv_channels: Tuple[int, ...] = (32, 64, 128),
    use_equinox_gru: bool = True,
    use_weight_norm: bool = True,
    sequence_length: int = 4096,
    **kwargs
) -> ExperimentConfig:
    """Create experiment configuration with common parameters."""
    return ExperimentConfig(
        latent_dim=latent_dim,
        conv_channels=conv_channels,
        use_equinox_gru=use_equinox_gru,
        use_weight_norm=use_weight_norm,
        sequence_length=sequence_length,
        **kwargs
    ) 