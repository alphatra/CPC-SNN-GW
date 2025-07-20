"""
Enhanced Contrastive Predictive Coding (CPC) Encoder

Self-supervised learning architecture for gravitational wave strain data.
Enhanced with findings from CPC+SNN Integration Paper (2025).

Key improvements:
- Equinox.nn.GRUCell for better scan compatibility
- Weight normalization (RMSNorm) for stable training on long sequences
- Configurable sizes via ExperimentConfig
- Enhanced numerical stability and gradient flow
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import logging

try:
    import equinox as eqx
    EQUINOX_AVAILABLE = True
except ImportError:
    EQUINOX_AVAILABLE = False
    eqx = None

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
        
        # Normalize weights
        w = g * v / jnp.linalg.norm(v, axis=0, keepdims=True)
        
        # Apply linear transformation
        y = jnp.dot(x, w)
        
        # Add bias if requested
        if self.use_bias:
            b = self.param('b', nn.initializers.zeros, (self.features,))
            y = y + b
        
        return y


class EquinoxGRUWrapper(nn.Module):
    """Wrapper for Equinox GRU that integrates with Flax."""
    hidden_size: int
    
    def setup(self):
        """Initialize the Equinox GRU."""
        if not EQUINOX_AVAILABLE:
            raise ImportError("Equinox not available. Install with: pip install equinox")
        
        # Static GRU model will be created in __call__
        self.static_gru = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, initial_state: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Apply Equinox GRU with proper Flax integration.
        
        Args:
            x: Input tensor [batch, time, features]
            initial_state: Initial hidden state [batch, hidden_size]
            
        Returns:
            hidden_states: All hidden states [batch, time, hidden_size]
        """
        batch_size, seq_len, input_size = x.shape
        
        # Create GRU model if not exists
        if self.static_gru is None:
            # Use proper random key from Flax
            gru_key = self.make_rng("params")
            self.static_gru = eqx.nn.GRU(
                input_size=input_size,
                hidden_size=self.hidden_size,
                key=gru_key
            )
        
        # Split model into parameters and static structure
        params, static = eqx.filter(self.static_gru, eqx.is_array)
        
        # Register parameters with Flax
        gru_params = self.param(
            'gru_params',
            lambda rng, shape: params,
            ()
        )
        
        # Combine parameters with static structure (correct order: static first, params second)
        gru_model = eqx.combine(static, gru_params)
        
        # Initialize hidden state if not provided
        if initial_state is None:
            initial_state = jnp.zeros((batch_size, self.hidden_size))
        
        # Apply GRU using scan for efficiency
        def gru_step(carry, x_t):
            h_prev = carry
            h_new = eqx.nn.GRU.__call__(gru_model, x_t, h_prev)
            return h_new, h_new
        
        # Transpose for scan: [time, batch, features]
        x_transposed = jnp.transpose(x, (1, 0, 2))
        
        # Run GRU over sequence
        final_state, hidden_states = jax.lax.scan(
            gru_step,
            initial_state,
            x_transposed
        )
        
        # Transpose back to [batch, time, hidden_size]
        return jnp.transpose(hidden_states, (1, 0, 2))


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
        """Initialize encoder layers."""
        self.conv_layers = []
        self.norm_layers = []
        
        # Build convolutional layers
        for i, channels in enumerate(self.config.conv_channels):
            conv = nn.Conv(
                channels, 
                kernel_size=(self.config.conv_kernel_size,), 
                strides=(self.config.conv_stride,),
                kernel_init=nn.initializers.he_normal(),
                bias_init=nn.initializers.zeros
            )
            self.conv_layers.append(conv)
            
            # Add normalization layer
            if self.config.use_weight_norm:
                norm = RMSNorm(channels)
                self.norm_layers.append(norm)
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """
        Apply CPC encoder with enhanced features.
        
        Args:
            x: Input tensor [batch, time]
            train: Training mode flag
            
        Returns:
            latent_features: Encoded features [batch, time, latent_dim]
        """
        # Input preprocessing
        x = self._preprocess_input(x)
        
        # Convolutional feature extraction
        x = self._apply_conv_layers(x, train)
        
        # Recurrent processing
        x = self._apply_recurrent_layer(x)
        
        # Final projection with weight normalization
        x = self._apply_final_projection(x)
        
        # L2 normalization for contrastive learning
        x = self._apply_l2_normalization(x)
        
        return x
    
    def _preprocess_input(self, x: jnp.ndarray) -> jnp.ndarray:
        """Preprocess input with scaling and dimension handling."""
        # Scale input to reasonable range (GW strain data is very small ~1e-23)
        x = x * self.config.input_scaling
        
        # Add channel dimension for convolutions
        x = x[..., None]  # [batch, time, 1]
        
        return x
    
    def _apply_conv_layers(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        """Apply convolutional layers with normalization and regularization."""
        for i, conv in enumerate(self.conv_layers):
            # Convolution
            x = conv(x)
            
            # Activation
            x = nn.gelu(x)
            
            # Normalization
            if self.config.use_weight_norm:
                x = self.norm_layers[i](x)
            elif self.config.use_batch_norm:
                x = nn.BatchNorm(use_running_average=not train)(x)
            
            # Dropout
            if self.config.dropout_rate > 0.0:
                x = nn.Dropout(rate=self.config.dropout_rate, deterministic=not train)(x)
        
        # Remove channel dimension for RNN
        x = x.squeeze(-1)  # [batch, downsampled_time, features]
        
        return x
    
    def _apply_recurrent_layer(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply recurrent layer (GRU) with optional gradient checkpointing."""
        if self.config.use_equinox_gru and EQUINOX_AVAILABLE:
            # Use Equinox GRU wrapper
            gru = EquinoxGRUWrapper(hidden_size=self.config.gru_hidden_size)
            
            if self.config.use_gradient_checkpointing:
                # Apply gradient checkpointing with CSE prevention for long sequences
                @nn.remat(prevent_cse=True)
                def checkpointed_gru(x):
                    return gru(x)
                x = checkpointed_gru(x)
            else:
                x = gru(x)
        else:
            # Fallback to Flax GRU
            logger.warning("Equinox not available, using Flax GRU fallback")
            x = self._apply_flax_gru(x)
        
        return x
    
    def _apply_flax_gru(self, x: jnp.ndarray) -> jnp.ndarray:
        """Fallback Flax GRU implementation."""
        batch_size, seq_len, features = x.shape
        
        # Use Flax's built-in GRU
        gru_cell = nn.GRUCell(features=self.config.gru_hidden_size, name="flax_gru_cell")
        
        # Initialize carry with proper key from Flax
        carry = gru_cell.initialize_carry(
            self.make_rng('params'),  # ✅ Use proper key from Flax framework
            batch_dims=(batch_size,),
            size=self.config.gru_hidden_size
        )
        
        # Apply with optional gradient checkpointing
        if self.config.use_gradient_checkpointing:
            @nn.remat(prevent_cse=True)
            def scan_fn(carry, x_t):
                carry, y = gru_cell(carry, x_t)
                return carry, y
        else:
            def scan_fn(carry, x_t):
                carry, y = gru_cell(carry, x_t)
                return carry, y
        
        # Scan over time dimension
        x_transposed = jnp.transpose(x, (1, 0, 2))
        final_carry, hidden_states = jax.lax.scan(scan_fn, carry, x_transposed)
        
        # Transpose back
        return jnp.transpose(hidden_states, (1, 0, 2))
    
    def _apply_final_projection(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply final projection layer with weight normalization."""
        if self.config.use_weight_norm:
            # Use weight-normalized dense layer
            projection = WeightNormDense(features=self.config.latent_dim)
            x = projection(x)
        else:
            # Standard dense layer
            x = nn.Dense(
                self.config.latent_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros
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


# Backward compatibility
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


def enhanced_info_nce_loss(z_context: jnp.ndarray, 
                          z_target: jnp.ndarray, 
                          temperature: float = 0.1,
                          num_negatives: int = 8,
                          use_hard_negatives: bool = False) -> jnp.ndarray:
    """
    Enhanced InfoNCE loss with improved numerical stability and vectorized computation.
    
    Improvements:
    - Better numerical stability with eps guards
    - Cosine similarity computation
    - Optional hard negative mining
    - Vectorized implementation with jax.vmap for efficiency
    - Gradient-friendly implementation
    """
    batch_size, context_len, feature_dim = z_context.shape
    _, target_len, _ = z_target.shape
    
    # Ensure equal lengths for proper alignment
    min_len = min(context_len, target_len)
    z_context = z_context[:, :min_len, :]
    z_target = z_target[:, :min_len, :]
    
    # Normalize for cosine similarity (should already be normalized)
    z_context_norm = z_context / (jnp.linalg.norm(z_context, axis=-1, keepdims=True) + 1e-8)
    z_target_norm = z_target / (jnp.linalg.norm(z_target, axis=-1, keepdims=True) + 1e-8)
    
    # Prepare data for vmap: [time, batch, features]
    z_context_T = jnp.transpose(z_context_norm, (1, 0, 2))
    z_target_T = jnp.transpose(z_target_norm, (1, 0, 2))
    
    def loss_for_single_timestep(context_t, target_t):
        """Compute loss for single timestep - this will be vectorized."""
        # Compute similarity matrix
        similarity_matrix = jnp.dot(context_t, target_t.T)  # [batch, batch]
        
        # Apply temperature scaling
        logits = similarity_matrix / temperature
        
        # Optional hard negative mining
        if use_hard_negatives:
            # Find hardest negatives (highest similarity but wrong pairs)
            mask = jnp.eye(batch_size)  # Positive pairs
            negative_similarities = jnp.where(mask, -jnp.inf, similarity_matrix)
            
            # Keep only top-k hardest negatives
            hard_negatives = jnp.argsort(negative_similarities, axis=1)[:, -num_negatives:]
            
            # Create masked logits with hard negatives
            hard_mask = jnp.zeros_like(logits)
            
            # Vectorized hard negative mask creation
            batch_indices = jnp.arange(batch_size)[:, None]
            hard_mask = hard_mask.at[batch_indices, hard_negatives].set(1.0)
            hard_mask = hard_mask.at[jnp.arange(batch_size), jnp.arange(batch_size)].set(1.0)  # Keep positives
            
            logits = jnp.where(hard_mask, logits, -jnp.inf)
        
        # Labels: diagonal elements are positive pairs
        labels = jnp.arange(batch_size)
        
        # Compute cross-entropy loss
        step_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return jnp.mean(step_loss)
    
    # Vectorize loss computation over time dimension (axis 0)
    losses_per_timestep = jax.vmap(loss_for_single_timestep)(z_context_T, z_target_T)
    
    # Return mean loss across all timesteps
    return jnp.mean(losses_per_timestep)


def info_nce_loss(z_context: jnp.ndarray, z_target: jnp.ndarray, temperature: float = 0.1) -> jnp.ndarray:
    """Backward compatible InfoNCE loss."""
    return enhanced_info_nce_loss(z_context, z_target, temperature, use_hard_negatives=False)


# Factory functions with enhanced configurations
def create_enhanced_cpc_encoder(config: Optional[ExperimentConfig] = None) -> EnhancedCPCEncoder:
    """Create enhanced CPC encoder with full configuration support."""
    if config is None:
        config = ExperimentConfig()
    
    return EnhancedCPCEncoder(config=config)


def create_standard_cpc_encoder(latent_dim: int = 256,
                              conv_channels: Tuple[int, ...] = (32, 64, 128)) -> CPCEncoder:
    """Create standard CPC encoder for backward compatibility."""
    return CPCEncoder(
        latent_dim=latent_dim, 
        conv_channels=conv_channels,
        use_batch_norm=False,
        dropout_rate=0.0
    ) 


def create_experiment_config(
    latent_dim: int = 256,
    conv_channels: Tuple[int, ...] = (32, 64, 128),
    use_equinox_gru: bool = True,
    use_weight_norm: bool = True,
    sequence_length: int = 4096,
    **kwargs
) -> ExperimentConfig:
    """Create experiment configuration with common overrides."""
    config = ExperimentConfig(
        latent_dim=latent_dim,
        conv_channels=conv_channels,
        use_equinox_gru=use_equinox_gru,
        use_weight_norm=use_weight_norm,
        sequence_length=sequence_length
    )
    
    # Apply any additional overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config parameter: {key}")
    
    return config


# Enhanced test functions
def test_enhanced_cpc_encoder():
    """Test enhanced CPC encoder with full configuration."""
    print("Testing Enhanced CPC Encoder...")
    
    try:
        # Create test configuration
        config = create_experiment_config(
            latent_dim=64,
            conv_channels=(16, 32),
            use_equinox_gru=EQUINOX_AVAILABLE,
            use_weight_norm=True,
            sequence_length=1024
        )
        
        # Create test data
        key = jax.random.PRNGKey(42)
        batch_size = 4
        data = jax.random.normal(key, (batch_size, config.sequence_length)) * 1e-23
        
        # Create enhanced encoder
        encoder = create_enhanced_cpc_encoder(config)
        
        # Initialize and test
        params = encoder.init(key, data, train=True)
        output = encoder.apply(params, data, train=True)
        
        print(f"✅ Input shape: {data.shape}")
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Config: equinox_gru={config.use_equinox_gru}, weight_norm={config.use_weight_norm}")
        print(f"✅ Output range: [{jnp.min(output):.3f}, {jnp.max(output):.3f}]")
        print(f"✅ Output norm: {jnp.linalg.norm(output, axis=-1).mean():.3f}")
        
        # Test gradient computation
        loss_fn = lambda p, x: jnp.mean(encoder.apply(p, x, train=True)**2)
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params, data)
        
        print(f"✅ Gradient computation successful")
        print(f"✅ Gradient norm: {jnp.linalg.norm(jax.tree_flatten(grads)[0][0]):.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced CPC encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_weight_normalization():
    """Test weight normalization components."""
    print("Testing Weight Normalization...")
    
    try:
        key = jax.random.PRNGKey(42)
        
        # Test RMSNorm
        rms_norm = RMSNorm(features=64)
        x = jax.random.normal(key, (4, 100, 64))
        
        params = rms_norm.init(key, x)
        output = rms_norm.apply(params, x)
        
        print(f"✅ RMSNorm input shape: {x.shape}")
        print(f"✅ RMSNorm output shape: {output.shape}")
        print(f"✅ RMSNorm output RMS: {jnp.sqrt(jnp.mean(output**2)):.3f}")
        
        # Test WeightNormDense
        wn_dense = WeightNormDense(features=32)
        x_dense = jax.random.normal(key, (4, 64))
        
        params_dense = wn_dense.init(key, x_dense)
        output_dense = wn_dense.apply(params_dense, x_dense)
        
        print(f"✅ WeightNormDense input shape: {x_dense.shape}")
        print(f"✅ WeightNormDense output shape: {output_dense.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Weight normalization test failed: {e}")
        return False


def test_experiment_config():
    """Test experiment configuration system."""
    print("Testing Experiment Configuration...")
    
    try:
        # Test default config
        config = ExperimentConfig()
        print(f"✅ Default config created: latent_dim={config.latent_dim}")
        
        # Test custom config
        custom_config = create_experiment_config(
            latent_dim=128,
            conv_channels=(64, 128, 256),
            use_equinox_gru=False,
            temperature=0.05
        )
        print(f"✅ Custom config created: latent_dim={custom_config.latent_dim}")
        print(f"✅ Custom conv channels: {custom_config.conv_channels}")
        print(f"✅ Custom temperature: {custom_config.temperature}")
        
        # Test encoder creation with config
        encoder = create_enhanced_cpc_encoder(custom_config)
        print(f"✅ Enhanced encoder created with custom config")
        
        return True
        
    except Exception as e:
        print(f"❌ Experiment config test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧠 Testing Enhanced CPC Encoder Implementation...")
    print(f"Equinox available: {EQUINOX_AVAILABLE}")
    print()
    
    success1 = test_enhanced_cpc_encoder()
    print()
    success2 = test_weight_normalization()
    print()
    success3 = test_experiment_config()
    print()
    
    # Test InfoNCE loss
    print("Testing Enhanced InfoNCE Loss...")
    try:
        key = jax.random.PRNGKey(42)
        z_context = jax.random.normal(key, (4, 16, 64))
        z_target = jax.random.normal(key, (4, 16, 64))
        
        loss = enhanced_info_nce_loss(z_context, z_target, temperature=0.1)
        print(f"✅ Enhanced InfoNCE loss: {loss:.4f}")
        
        # Test with hard negatives
        loss_hard = enhanced_info_nce_loss(z_context, z_target, temperature=0.1, use_hard_negatives=True)
        print(f"✅ Hard negatives loss: {loss_hard:.4f}")
        
        success4 = True
    except Exception as e:
        print(f"❌ InfoNCE loss test failed: {e}")
        success4 = False
    
    overall_success = success1 and success2 and success3 and success4
    print(f"\n🎯 Overall: {'SUCCESS' if overall_success else 'FAILED'}")
    
    if overall_success:
        print("🎉 All enhanced CPC encoder tests passed!")
    else:
        print("❌ Some tests failed - check implementation") 