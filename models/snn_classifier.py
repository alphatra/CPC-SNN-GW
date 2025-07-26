"""
Enhanced Spiking Neural Network (SNN) Classifier

Neuromorphic binary classifier using optimized JAX/Flax LIF neurons
for energy-efficient gravitational wave detection.

Key improvements:
- Vectorized LIF update for single fused kernel performance
- Modular utilities via snn_utils (surrogate gradients, validation)
- Memory-efficient implementations for long sequences
- Comprehensive backward compatibility
- 🚀 NEW: Enhanced LIF with refractory period and state persistence
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
    spike_function_with_surrogate, spike_function_with_enhanced_surrogate,
    BatchedSNNValidator
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
    
    # 🚀 NEW: Enhanced LIF parameters
    use_enhanced_lif: bool = True  # Use enhanced LIF with memory and refractory period
    tau_ref: float = 2e-3   # Refractory period time constant
    tau_adaptation: float = 100e-3  # Spike frequency adaptation
    use_refractory_period: bool = True  # Enable refractory period
    use_adaptation: bool = True  # Enable spike frequency adaptation
    use_learnable_dynamics: bool = True  # Learn time constants
    reset_mechanism: str = "soft"  # "hard" or "soft" reset
    reset_factor: float = 0.8  # For soft reset (0.0=hard, 1.0=no reset)
    refractory_time_constant: float = 2.0  # Alias for compatibility
    adaptation_time_constant: float = 20.0  # Alias for compatibility
    
    # Surrogate gradient
    surrogate_type: SurrogateGradientType = SurrogateGradientType.ADAPTIVE_MULTI_SCALE  # 🚀 Use enhanced
    surrogate_beta: float = 10.0  # Surrogate gradient steepness
    
    # Training
    dropout_rate: float = 0.1
    use_batch_norm: bool = False
    
    # Optimization
    use_fused_kernel: bool = True
    memory_efficient: bool = True


class EnhancedLIFWithMemory(nn.Module):
    """
    🚀 ENHANCED: LIF neuron with refractory period, adaptation, and persistent state.
    
    Biologically realistic features:
    - Refractory period: neurons can't spike immediately after spiking
    - Spike frequency adaptation: neurons adapt their excitability
    - Learnable time constants: network learns optimal dynamics
    - Soft/hard reset mechanisms: configurable membrane reset
    - State persistence: maintains state across time steps
    """
    
    config: SNNConfig
    
    def setup(self):
        """Initialize enhanced LIF parameters."""
        # 🧠 LEARNABLE TIME CONSTANTS (if enabled)
        if self.config.use_learnable_dynamics:
            # Learnable membrane time constant
            self.tau_mem_param = self.param(
                'tau_mem_learnable',
                lambda key, shape: jnp.log(self.config.tau_mem),  # Log-space for positivity
                ()
            )
            
            # Learnable synaptic time constant
            self.tau_syn_param = self.param(
                'tau_syn_learnable', 
                lambda key, shape: jnp.log(self.config.tau_syn),
                ()
            )
            
            # Learnable refractory time constant
            if self.config.use_refractory_period:
                self.tau_ref_param = self.param(
                    'tau_ref_learnable',
                    lambda key, shape: jnp.log(self.config.tau_ref),
                    ()
                )
            
            # Learnable adaptation time constant
            if self.config.use_adaptation:
                self.tau_adapt_param = self.param(
                    'tau_adapt_learnable',
                    lambda key, shape: jnp.log(self.config.tau_adaptation),
                    ()
                )
            
            logger.debug("🚀 Using learnable LIF time constants")
        else:
            logger.debug("⚠️  Using fixed LIF time constants")
        
        # Enhanced surrogate gradient
        self.surrogate_fn = create_surrogate_gradient_fn(
            self.config.surrogate_type, 
            self.config.surrogate_beta
        )
    
    def _get_time_constants(self) -> Dict[str, float]:
        """Get current time constants (learnable or fixed)."""
        if self.config.use_learnable_dynamics:
            return {
                'tau_mem': jnp.exp(self.tau_mem_param),  # Ensure positivity
                'tau_syn': jnp.exp(self.tau_syn_param),
                'tau_ref': jnp.exp(self.tau_ref_param) if self.config.use_refractory_period else 0.0,
                'tau_adapt': jnp.exp(self.tau_adapt_param) if self.config.use_adaptation else 0.0
            }
        else:
            return {
                'tau_mem': self.config.tau_mem,
                'tau_syn': self.config.tau_syn,
                'tau_ref': self.config.tau_ref if self.config.use_refractory_period else 0.0,
                'tau_adapt': self.config.tau_adaptation if self.config.use_adaptation else 0.0
            }
    
    @nn.compact
    def __call__(self, spikes: jnp.ndarray, training: bool = False, training_progress: float = 0.0) -> jnp.ndarray:
        """
        Enhanced LIF forward pass with refractory period and adaptation.
        
        Args:
            spikes: Input spikes [batch, time, input_dim] or [batch, time, seq_len, feature_dim]
            training: Training mode flag
            training_progress: Training progress for adaptive surrogate (0.0 to 1.0)
            
        Returns:
            Output spikes [batch, time, hidden_size]
        """
        # ✅ CRITICAL FIX: Handle both 3D and 4D input shapes
        if len(spikes.shape) == 4:
            # 4D input from spike bridge: [batch, time_steps, seq_len, feature_dim]
            batch_size, time_steps, seq_len, feature_dim = spikes.shape
            # Flatten spatial dimensions: [batch, time, seq_len * feature_dim]
            spikes = spikes.reshape(batch_size, time_steps, seq_len * feature_dim)
            input_dim = seq_len * feature_dim
        elif len(spikes.shape) == 3:
            # 3D input: [batch, time, input_dim]
            batch_size, time_steps, input_dim = spikes.shape
        else:
            raise ValueError(f"Expected 3D or 4D spike input, got {len(spikes.shape)}D: {spikes.shape}")
        
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
        
        # Get current time constants
        time_constants = self._get_time_constants()
        
        # 🚀 ENHANCED LIF with refractory period and adaptation
        return self._enhanced_lif_forward(spikes, W, b, time_constants, training, training_progress)
    
    def _enhanced_lif_forward(self, 
                             spikes: jnp.ndarray, 
                             W: jnp.ndarray, 
                             b: jnp.ndarray,
                             time_constants: Dict[str, float],
                             training: bool,
                             training_progress: float) -> jnp.ndarray:
        """Enhanced LIF forward pass using JAX scan for memory efficiency."""
        batch_size, time_steps, input_dim = spikes.shape
        
        # Precompute decay factors
        alpha_mem = jnp.exp(-self.config.dt / time_constants['tau_mem'])
        alpha_syn = jnp.exp(-self.config.dt / time_constants['tau_syn'])
        
        # Enhanced state with refractory period and adaptation
        if self.config.use_refractory_period:
            alpha_ref = jnp.exp(-self.config.dt / time_constants['tau_ref'])
        
        if self.config.use_adaptation:
            alpha_adapt = jnp.exp(-self.config.dt / time_constants['tau_adapt'])
        
        # 🧠 ENHANCED INITIAL STATE
        init_state = {
            'v_mem': jnp.zeros((batch_size, self.config.hidden_size)),  # Membrane potential
            'i_syn': jnp.zeros((batch_size, self.config.hidden_size)),  # Synaptic current
        }
        
        # Add refractory state if enabled
        if self.config.use_refractory_period:
            init_state['refrac_timer'] = jnp.zeros((batch_size, self.config.hidden_size))
        
        # Add adaptation state if enabled  
        if self.config.use_adaptation:
            init_state['adapt_current'] = jnp.zeros((batch_size, self.config.hidden_size))
        
        def enhanced_lif_step(carry, spike_t):
            """Enhanced LIF step with all biological features."""
            v_mem = carry['v_mem']
            i_syn = carry['i_syn']
            
            # 🚨 REFRACTORY PERIOD HANDLING
            if self.config.use_refractory_period:
                refrac_timer = carry['refrac_timer']
                # Neurons in refractory period can't receive input
                input_mask = (refrac_timer <= 0).astype(jnp.float32)
            else:
                input_mask = 1.0
            
            # 🧠 SPIKE FREQUENCY ADAPTATION
            if self.config.use_adaptation:
                adapt_current = carry['adapt_current']
                # Adaptation reduces excitability after spiking
                effective_threshold = self.config.threshold + adapt_current
            else:
                effective_threshold = self.config.threshold
                adapt_current = 0.0
            
            # Synaptic current update (masked by refractory period)
            synaptic_input = jnp.dot(spike_t, W) + b
            i_syn = alpha_syn * i_syn + synaptic_input * input_mask
            
            # Membrane potential update
            v_mem = alpha_mem * v_mem + i_syn
            
            # 🚀 ENHANCED SPIKE GENERATION with adaptive surrogate
            if self.config.surrogate_type == SurrogateGradientType.ADAPTIVE_MULTI_SCALE:
                spikes_out = spike_function_with_enhanced_surrogate(
                    v_mem - effective_threshold,
                    threshold=0.0,
                    training_progress=training_progress
                )
            else:
                spikes_out = spike_function_with_surrogate(
                    v_mem, effective_threshold, self.surrogate_fn
                )
            
            # 🔄 MEMBRANE RESET (soft or hard)
            if self.config.reset_mechanism == "hard":
                # Hard reset: set to 0
                v_mem = v_mem * (1.0 - spikes_out)
            else:
                # Soft reset: subtract scaled threshold
                v_mem = v_mem - spikes_out * effective_threshold * self.config.reset_factor
            
            # 📊 UPDATE ENHANCED STATES
            new_carry = {
                'v_mem': v_mem,
                'i_syn': i_syn
            }
            
            # Update refractory timer
            if self.config.use_refractory_period:
                # Start refractory period on spike, decay otherwise
                refrac_timer = jnp.where(
                    spikes_out > 0,
                    time_constants['tau_ref'] / self.config.dt,  # Reset timer on spike
                    jnp.maximum(0.0, refrac_timer - 1.0)  # Decay timer
                )
                new_carry['refrac_timer'] = refrac_timer
            
            # Update adaptation current
            if self.config.use_adaptation:
                # Increase adaptation on spike, decay otherwise
                adapt_current = alpha_adapt * adapt_current + spikes_out * 0.1  # Adaptive increment
                new_carry['adapt_current'] = adapt_current
            
            return new_carry, spikes_out
        
        # Apply enhanced scan
        _, output_spikes = jax.lax.scan(
            enhanced_lif_step, init_state, spikes.transpose(1, 0, 2)
        )
        
        # Transpose back: [batch, time, hidden_size]
        return output_spikes.transpose(1, 0, 2)


class VectorizedLIFLayer(nn.Module):
    """
    Vectorized LIF layer optimized for single fused kernel.
    🚀 ENHANCED: Now uses EnhancedLIFWithMemory by default
    """
    
    config: SNNConfig
    
    def setup(self):
        """Initialize LIF layer with enhanced dynamics."""
        # Use enhanced LIF by default
        self.enhanced_lif = EnhancedLIFWithMemory(config=self.config)
    
    @nn.compact
    def __call__(self, spikes: jnp.ndarray, training: bool = False, training_progress: float = 0.0) -> jnp.ndarray:
        """Forward pass using enhanced LIF neurons."""
        return self.enhanced_lif(spikes, training=training, training_progress=training_progress)


class EnhancedSNNClassifier(nn.Module):
    """
    Enhanced SNN classifier with vectorized LIF and surrogate gradients.
    
    Features:
    - Vectorized LIF layers for optimal GPU/TPU performance
    - Configurable surrogate gradient methods
    - Batch normalization and dropout support
    - Memory-efficient implementations
    - 🚀 ENHANCED: Now supports enhanced LIF with refractory period and adaptation
    """
    
    config: SNNConfig
    
    @nn.compact
    def __call__(self, spikes: jnp.ndarray, training: bool = False, training_progress: float = 0.0) -> jnp.ndarray:
        """
        Forward pass through SNN classifier.
        
        Args:
            spikes: Input spikes [batch, time, input_dim]
            training: Training mode flag
            training_progress: Training progress (0.0 to 1.0) for adaptive components
            
        Returns:
            Classification logits [batch, num_classes]
        """
        x = spikes
        
        # Multiple enhanced LIF layers
        for i in range(self.config.num_layers):
            x = VectorizedLIFLayer(
                config=self.config,
                name=f'lif_layer_{i}'
            )(x, training=training, training_progress=training_progress)
            
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
        
        # Final classification layer with enhanced initialization
        logits = nn.Dense(
            self.config.num_classes,
            kernel_init=nn.initializers.he_normal(),  # Better for GELU/ReLU-like
            bias_init=nn.initializers.zeros,
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