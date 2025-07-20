"""
Enhanced Spiking Neural Network (SNN) Classifier

Neuromorphic binary classifier using optimized JAX/Flax LIF neurons
for energy-efficient gravitational wave detection.

Key improvements:
- Vectorized LIF update for single fused kernel performance
- Modular utilities via snn_utils (surrogate gradients, validation)
- Memory-efficient implementations for long sequences
- Comprehensive backward compatibility
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging

# Import local utilities  
from .snn_utils import (
    SurrogateGradientType, create_surrogate_gradient_fn, 
    spike_function_with_surrogate, BatchedSNNValidator
)

logger = logging.getLogger(__name__)


@dataclass
class SNNConfig:
    """Configuration for SNN classifier."""
    # Architecture
    hidden_size: int = 128
    num_classes: int = 3  # NOISE=0, CONTINUOUS_GW=1, BINARY_MERGER=2
    num_layers: int = 2
    
    # LIF parameters
    tau_mem: float = 20e-3  # Membrane time constant
    tau_syn: float = 5e-3   # Synaptic time constant
    threshold: float = 1.0  # Spike threshold
    dt: float = 1e-3        # Time step
    
    # Surrogate gradient
    surrogate_type: SurrogateGradientType = SurrogateGradientType.FAST_SIGMOID
    surrogate_beta: float = 10.0  # Surrogate gradient steepness
    
    # Training
    dropout_rate: float = 0.1
    use_batch_norm: bool = False
    
    # Optimization
    use_fused_kernel: bool = True
    memory_efficient: bool = True


class VectorizedLIFLayer(nn.Module):
    """
    Vectorized LIF layer optimized for single fused kernel.
    
    All operations are vectorized across batch and time dimensions
    for maximum GPU/TPU efficiency.
    """
    
    config: SNNConfig
    
    def setup(self):
        """Initialize LIF layer parameters."""
        self.surrogate_fn = create_surrogate_gradient_fn(
            self.config.surrogate_type, 
            self.config.surrogate_beta
        )
    
    @nn.compact
    def __call__(self, spikes: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Apply vectorized LIF dynamics.
        
        Args:
            spikes: Input spikes [batch, time, input_dim]
            training: Training mode flag
            
        Returns:
            Output spikes [batch, time, hidden_size]
        """
        batch_size, time_steps, input_dim = spikes.shape
        
        # Weight and bias parameters
        W = self.param(
            'weight',
            nn.initializers.xavier_uniform(),
            (input_dim, self.config.hidden_size)
        )
        b = self.param(
            'bias',
            nn.initializers.zeros,
            (self.config.hidden_size,)
        )
        
        # Choose implementation strategy
        if self.config.use_fused_kernel and self.config.memory_efficient:
            return self._optimized_lif_forward(spikes, W, b, training)
        else:
            return self._scan_lif_forward(spikes, W, b, training)
    
    def _optimized_lif_forward(self, spikes: jnp.ndarray, W: jnp.ndarray, b: jnp.ndarray, training: bool) -> jnp.ndarray:
        """Optimized LIF forward pass with vectorized operations."""
        batch_size, time_steps, _ = spikes.shape
        
        # Precompute decay factors
        alpha_mem = jnp.exp(-self.config.dt / self.config.tau_mem)
        alpha_syn = jnp.exp(-self.config.dt / self.config.tau_syn)
        
        # Initialize states
        v_mem = jnp.zeros((batch_size, self.config.hidden_size))
        i_syn = jnp.zeros((batch_size, self.config.hidden_size))
        
        output_spikes = []
        
        # Time loop (unrolled for small sequences)
        for t in range(time_steps):
            # Synaptic current update
            i_syn = alpha_syn * i_syn + jnp.dot(spikes[:, t], W) + b
            
            # Membrane potential update
            v_mem = alpha_mem * v_mem + i_syn
            
            # Spike generation with surrogate gradient
            spikes_out = spike_function_with_surrogate(
                v_mem, self.config.threshold, self.surrogate_fn
            )
            
            # Reset membrane potential
            v_mem = v_mem - spikes_out * self.config.threshold
            
            output_spikes.append(spikes_out)
        
        # Stack outputs: [batch, time, hidden_size]
        return jnp.stack(output_spikes, axis=1)
    
    def _scan_lif_forward(self, spikes: jnp.ndarray, W: jnp.ndarray, b: jnp.ndarray, training: bool) -> jnp.ndarray:
        """LIF forward pass using JAX scan for memory efficiency."""
        batch_size, time_steps, _ = spikes.shape
        
        # Precompute decay factors
        alpha_mem = jnp.exp(-self.config.dt / self.config.tau_mem)
        alpha_syn = jnp.exp(-self.config.dt / self.config.tau_syn)
        
        # Initial state
        init_state = {
            'v_mem': jnp.zeros((batch_size, self.config.hidden_size)),
            'i_syn': jnp.zeros((batch_size, self.config.hidden_size))
        }
        
        def lif_step(carry, spike_t):
            v_mem, i_syn = carry['v_mem'], carry['i_syn']
            
            # Synaptic current update
            i_syn = alpha_syn * i_syn + jnp.dot(spike_t, W) + b
            
            # Membrane potential update  
            v_mem = alpha_mem * v_mem + i_syn
            
            # Spike generation
            spikes_out = spike_function_with_surrogate(
                v_mem, self.config.threshold, self.surrogate_fn
            )
            
            # Reset
            v_mem = v_mem - spikes_out * self.config.threshold
            
            new_carry = {'v_mem': v_mem, 'i_syn': i_syn}
            return new_carry, spikes_out
        
        # Apply scan
        _, output_spikes = jax.lax.scan(
            lif_step, init_state, spikes.transpose(1, 0, 2)
        )
        
        # Transpose back: [batch, time, hidden_size]
        return output_spikes.transpose(1, 0, 2)


class EnhancedSNNClassifier(nn.Module):
    """
    Enhanced SNN classifier with vectorized LIF and surrogate gradients.
    
    Features:
    - Vectorized LIF layers for optimal GPU/TPU performance
    - Configurable surrogate gradient methods
    - Batch normalization and dropout support
    - Memory-efficient implementations
    """
    
    config: SNNConfig
    
    @nn.compact
    def __call__(self, spikes: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Forward pass through SNN classifier.
        
        Args:
            spikes: Input spikes [batch, time, input_dim]
            training: Training mode flag
            
        Returns:
            Classification logits [batch, num_classes]
        """
        x = spikes
        
        # Multiple LIF layers
        for i in range(self.config.num_layers):
            x = VectorizedLIFLayer(
                config=self.config,
                name=f'lif_layer_{i}'
            )(x, training=training)
            
            # Optional batch normalization
            if self.config.use_batch_norm:
                x = nn.BatchNorm(
                    use_running_average=not training,
                    name=f'batch_norm_{i}'
                )(x)
            
            # Optional dropout
            if self.config.dropout_rate > 0:
                x = nn.Dropout(
                    rate=self.config.dropout_rate,
                    deterministic=not training
                )(x)
        
        # Global average pooling over time
        x_pooled = jnp.mean(x, axis=1)  # [batch, hidden_size]
        
        # Final classification layer
        logits = nn.Dense(
            self.config.num_classes,
            kernel_init=nn.initializers.xavier_uniform(),
            name='classifier'
        )(x_pooled)
        
        return logits


# Backward compatibility classes
class LIFLayer(nn.Module):
    """Backward compatible LIF layer."""
    hidden_size: int
    tau_mem: float = 20e-3
    tau_syn: float = 5e-3
    threshold: float = 1.0
    dt: float = 1e-3
    
    def setup(self):
        """Convert to SNNConfig and use VectorizedLIFLayer."""
        self.config = SNNConfig(
            hidden_size=self.hidden_size,
            tau_mem=self.tau_mem,
            tau_syn=self.tau_syn,
            threshold=self.threshold,
            dt=self.dt,
            use_fused_kernel=False  # Disable for compatibility
        )
        self.vectorized_layer = VectorizedLIFLayer(config=self.config)
    
    @nn.compact
    def __call__(self, spikes: jnp.ndarray) -> jnp.ndarray:
        """Forward pass using vectorized layer."""
        return self.vectorized_layer(spikes, training=False)


class SNNClassifier(nn.Module):
    """Backward compatible SNN classifier."""
    hidden_size: int = 128
    num_classes: int = 2
    tau_mem: float = 20e-3
    tau_syn: float = 5e-3
    threshold: float = 1.0
    
    def setup(self):
        """Convert to SNNConfig and use EnhancedSNNClassifier."""
        self.config = SNNConfig(
            hidden_size=self.hidden_size,
            num_classes=self.num_classes,
            tau_mem=self.tau_mem,
            tau_syn=self.tau_syn,
            threshold=self.threshold,
            num_layers=2,
            use_fused_kernel=False,  # Disable for compatibility
            dropout_rate=0.0
        )
        self.enhanced_classifier = EnhancedSNNClassifier(config=self.config)
    
    @nn.compact
    def __call__(self, spikes: jnp.ndarray) -> jnp.ndarray:
        """Forward pass using enhanced classifier."""
        return self.enhanced_classifier(spikes, training=False)


# Factory functions
def create_enhanced_snn_classifier(config: Optional[SNNConfig] = None) -> EnhancedSNNClassifier:
    """Create enhanced SNN classifier with configuration."""
    if config is None:
        config = SNNConfig()
    return EnhancedSNNClassifier(config=config)


def create_snn_config(
    hidden_size: int = 128,
    num_classes: int = 2,
    surrogate_type: SurrogateGradientType = SurrogateGradientType.FAST_SIGMOID,
    **kwargs
) -> SNNConfig:
    """Create SNN configuration with common parameters."""
    return SNNConfig(
        hidden_size=hidden_size,
        num_classes=num_classes,
        surrogate_type=surrogate_type,
        **kwargs
    )


def create_snn_classifier(hidden_size: int = 128, num_classes: int = 3) -> SNNClassifier:
    """Create standard SNN classifier for backward compatibility."""
    return SNNClassifier(hidden_size=hidden_size, num_classes=num_classes)


# Training utilities
class SNNTrainer:
    """Simple SNN trainer for backward compatibility."""
    
    def __init__(self, 
                 snn_model: nn.Module,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 num_classes: int = 2):
        self.snn_model = snn_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        
        # Create optimizer
        self.optimizer = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create validator
        self.validator = BatchedSNNValidator(num_classes=num_classes)
    
    def create_train_state(self, key: jax.random.PRNGKey, sample_input: jnp.ndarray):
        """Create initial training state."""
        params = self.snn_model.init(key, sample_input)
        opt_state = self.optimizer.init(params)
        return {'params': params, 'opt_state': opt_state}
    
    def train_step(self, params, opt_state, batch, labels):
        """Single training step."""
        def loss_fn(params):
            logits = self.snn_model.apply(params, batch, training=True)
            loss = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            )
            return loss, logits
        
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        metrics = self.validator.compute_metrics(logits, labels)
        metrics['loss'] = loss
        
        return new_params, new_opt_state, metrics
    
    def validation_step(self, params, batch, labels):
        """Single validation step."""
        return self.validator.validation_step(self.snn_model, params, batch, labels) 