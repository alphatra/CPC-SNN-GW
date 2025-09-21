"""
LIF layer implementations for SNN components.

This module contains spiking neuron layer implementations extracted from
snn_classifier.py for better modularity:
- LIFLayer: Basic Leaky Integrate-and-Fire layer
- VectorizedLIFLayer: Vectorized LIF implementation
- EnhancedLIFWithMemory: Advanced LIF with memory mechanisms

Split from snn_classifier.py for better maintainability.
"""

import logging
from typing import Tuple, Optional, Dict, Any

import jax
import jax.numpy as jnp
import flax.linen as nn

from .config import LIFConfig
from ..snn_utils import create_surrogate_gradient_fn, SurrogateGradientType

logger = logging.getLogger(__name__)


class LIFLayer(nn.Module):
    """
    Basic Leaky Integrate-and-Fire neuron layer.
    
    Implements standard LIF dynamics:
    - Membrane potential integration
    - Spike generation with threshold
    - Reset mechanism after spikes
    """
    
    features: int
    tau_mem: float = 20e-3
    tau_syn: float = 5e-3  
    threshold: float = 0.5
    reset_potential: float = 0.0
    surrogate_beta: float = 4.0
    
    @nn.compact
    def __call__(self, spikes: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Process spike trains through LIF layer.
        
        Args:
            spikes: Input spike trains [batch_size, time_steps, input_features]
            training: Training mode flag
            
        Returns:
            Output spike trains [batch_size, time_steps, features]
        """
        batch_size, time_steps, input_features = spikes.shape
        
        # ✅ WEIGHTS: Dense connection weights
        dense_layer = nn.Dense(
            self.features,
            use_bias=True,
            name='lif_dense'
        )
        
        # ✅ SURROGATE: Create surrogate gradient function (continuous output)
        surrogate_fn = create_surrogate_gradient_fn(
            SurrogateGradientType.FAST_SIGMOID,
            self.surrogate_beta
        )
        
        # ✅ DYNAMICS: LIF membrane dynamics simulation
        # Initialize membrane potential
        v_mem = jnp.zeros((batch_size, self.features))
        output_spikes = []
        
        for t in range(time_steps):
            # Get input current at time t
            input_current = dense_layer(spikes[:, t, :])
            
            # ✅ MEMBRANE INTEGRATION: Leaky integration
            # dv/dt = (I - v)/tau_mem
            alpha_mem = jnp.exp(-1.0 / self.tau_mem)  # Discrete time constant
            
            # Update membrane potential
            v_mem = alpha_mem * v_mem + (1 - alpha_mem) * input_current
            
            # ✅ SPIKE GENERATION: Continuous surrogate output (no hard threshold)
            spike_continuous = surrogate_fn(v_mem - self.threshold)
            
            # ✅ SOFT RESET: Mix towards reset potential using continuous spike
            v_mem = v_mem * (1.0 - spike_continuous) + self.reset_potential * spike_continuous
            
            # ✅ LAYERNORM: Stabilize per-time-step activations
            spike_continuous = nn.LayerNorm(name=f'lif_norm_t{t+1}')(spike_continuous)
            
            output_spikes.append(spike_continuous)
        
        # Stack temporal outputs
        output_spike_trains = jnp.stack(output_spikes, axis=1)
        
        return output_spike_trains


class VectorizedLIFLayer(nn.Module):
    """
    Vectorized LIF layer for improved performance.
    
    Optimized implementation that processes all time steps simultaneously
    for better computational efficiency.
    """
    
    features: int
    tau_mem: float = 20e-3
    tau_syn: float = 5e-3
    threshold: float = 0.5
    reset_potential: float = 0.0
    surrogate_beta: float = 4.0
    
    @nn.compact
    def __call__(self, spikes: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Vectorized LIF processing.
        
        Args:
            spikes: Input spike trains [batch_size, time_steps, input_features]
            training: Training mode flag
            
        Returns:
            Output spike trains [batch_size, time_steps, features]
        """
        batch_size, time_steps, input_features = spikes.shape
        
        # ✅ DENSE: Input transformation
        dense_layer = nn.Dense(self.features, name='vectorized_lif_dense')
        input_currents = dense_layer(spikes)  # [batch, time, features]
        
        # ✅ VECTORIZED DYNAMICS: Process all time steps at once using scan
        def lif_step(carry, current_input):
            v_mem = carry
            
            # Leaky integration
            alpha = jnp.exp(-1.0 / self.tau_mem)
            v_mem = alpha * v_mem + (1 - alpha) * current_input
            
            # Continuous surrogate spike
            surrogate_fn = create_surrogate_gradient_fn(
                SurrogateGradientType.FAST_SIGMOID,
                self.surrogate_beta
            )
            spike_cont = surrogate_fn(v_mem - self.threshold)
            
            # Soft reset using continuous spike
            v_mem = v_mem * (1.0 - spike_cont) + self.reset_potential * spike_cont
            
            return v_mem, spike_cont
        
        # Initialize membrane state
        initial_v_mem = jnp.zeros((batch_size, self.features))
        
        # Scan over time dimension
        final_v_mem, spike_outputs = jax.lax.scan(
            lif_step, 
            initial_v_mem, 
            jnp.transpose(input_currents, (1, 0, 2))  # [time, batch, features]
        )
        
        # Transpose back to [batch, time, features]
        spike_outputs = jnp.transpose(spike_outputs, (1, 0, 2))
        
        # ✅ LAYERNORM across features per time step
        spike_outputs = nn.LayerNorm(name='vectorized_lif_norm')(spike_outputs)
        
        return spike_outputs


class EnhancedLIFWithMemory(nn.Module):
    """
    Enhanced LIF layer with memory mechanisms and advanced features.
    
    Features:
    - Long-term synaptic memory
    - Adaptive thresholds
    - Multi-compartment dynamics
    - Enhanced gradient flow
    """
    
    features: int
    tau_mem: float = 20e-3
    tau_syn: float = 5e-3
    tau_adapt: float = 100e-3  # Adaptation time constant
    threshold: float = 0.5
    reset_potential: float = 0.0
    
    # Memory parameters
    use_long_term_memory: bool = True
    memory_decay: float = 0.95
    
    # Adaptive features
    use_adaptive_threshold: bool = True
    threshold_adaptation_rate: float = 0.01
    
    # Enhanced dynamics
    use_multi_compartment: bool = False
    surrogate_beta: float = 15.0  # Higher beta for enhanced layers
    
    def setup(self):
        """Initialize enhanced LIF components."""
        # ✅ WEIGHTS: Input and recurrent connections
        self.input_projection = nn.Dense(self.features, name='enhanced_lif_input')
        
        if self.use_multi_compartment:
            self.recurrent_projection = nn.Dense(self.features, name='enhanced_lif_recurrent')
        
        # ✅ ADAPTATION: Learnable adaptation parameters
        if self.use_adaptive_threshold:
            self.threshold_adaptation = self.param(
                'threshold_adaptation',
                nn.initializers.constant(0.0),
                (self.features,)
            )
        
        # ✅ MEMORY: Long-term memory parameters
        if self.use_long_term_memory:
            self.memory_weights = self.param(
                'memory_weights',
                nn.initializers.uniform(scale=0.1),
                (self.features,)
            )
    
    def __call__(self, spikes: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Enhanced LIF processing with memory and adaptation.
        
        Args:
            spikes: Input spike trains [batch_size, time_steps, input_features]
            training: Training mode flag
            
        Returns:
            Enhanced output spike trains [batch_size, time_steps, features]
        """
        batch_size, time_steps, input_features = spikes.shape
        
        # ✅ INPUT: Transform input spikes
        input_currents = self.input_projection(spikes)
        
        # ✅ DYNAMICS: Enhanced LIF simulation with memory
        def enhanced_lif_step(carry, inputs):
            v_mem, adaptation, memory = carry
            current_input = inputs
            
            # ✅ SYNAPTIC: Synaptic filtering
            alpha_syn = jnp.exp(-1.0 / self.tau_syn)
            filtered_input = alpha_syn * current_input
            
            # ✅ MEMBRANE: Membrane integration
            alpha_mem = jnp.exp(-1.0 / self.tau_mem)
            
            # Add memory contribution if enabled
            if self.use_long_term_memory:
                memory_contribution = self.memory_weights * memory
                total_input = filtered_input + memory_contribution
            else:
                total_input = filtered_input
            
            # Membrane update
            v_mem = alpha_mem * v_mem + (1 - alpha_mem) * total_input
            
            # ✅ ADAPTATION: Adaptive threshold
            if self.use_adaptive_threshold:
                current_threshold = self.threshold + self.threshold_adaptation
            else:
                current_threshold = self.threshold
            
            # ✅ SPIKES: Continuous surrogate spikes
            surrogate_fn = create_surrogate_gradient_fn(
                SurrogateGradientType.FAST_SIGMOID,
                self.surrogate_beta
            )
            spike_output = surrogate_fn(v_mem - current_threshold)
            
            # ✅ RESET: Soft reset with adaptation
            if self.use_adaptive_threshold:
                # Update adaptation based on spike activity
                alpha_adapt = jnp.exp(-1.0 / self.tau_adapt)
                adaptation = alpha_adapt * adaptation + (1 - alpha_adapt) * spike_output * self.threshold_adaptation_rate
                v_mem_reset = (v_mem * (1.0 - spike_output)) + (self.reset_potential - adaptation) * spike_output
            else:
                v_mem_reset = (v_mem * (1.0 - spike_output)) + self.reset_potential * spike_output
                adaptation = jnp.zeros_like(v_mem)
            
            # ✅ MEMORY: Update long-term memory
            if self.use_long_term_memory:
                memory = self.memory_decay * memory + (1 - self.memory_decay) * spike_output
            else:
                memory = jnp.zeros_like(v_mem)
            
            new_carry = (v_mem_reset, adaptation, memory)
            return new_carry, spike_output
        
        # ✅ INITIALIZATION: Initialize states
        initial_v_mem = jnp.zeros((batch_size, self.features))
        initial_adaptation = jnp.zeros((batch_size, self.features))
        initial_memory = jnp.zeros((batch_size, self.features))
        initial_carry = (initial_v_mem, initial_adaptation, initial_memory)
        
        # ✅ SCAN: Process temporal sequence
        final_carry, spike_outputs = jax.lax.scan(
            enhanced_lif_step,
            initial_carry,
            jnp.transpose(input_currents, (1, 0, 2))  # [time, batch, features]
        )
        
        # Transpose back to [batch, time, features]
        spike_outputs = jnp.transpose(spike_outputs, (1, 0, 2))
        
        # ✅ LAYERNORM
        spike_outputs = nn.LayerNorm(name='enhanced_lif_norm')(spike_outputs)
        
        return spike_outputs


# Export layer classes
__all__ = [
    "LIFLayer",
    "VectorizedLIFLayer",
    "EnhancedLIFWithMemory"
]

