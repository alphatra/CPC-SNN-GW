"""
Deep SNN implementations for advanced training.

This module contains deep spiking neural network components extracted from
advanced_training.py for better modularity.

Split from advanced_training.py for better maintainability.
"""

import logging
from typing import Dict, Any, List, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn

logger = logging.getLogger(__name__)


class LIFLayer(nn.Module):
    """
    Leaky Integrate-and-Fire neuron layer with proper dynamics.
    """
    
    hidden_dim: int
    tau_mem: float = 20.0
    tau_syn: float = 5.0
    threshold: float = 1.0
    surrogate_beta: float = 4.0
    reset_potential: float = 0.0
    
    def setup(self):
        """Initialize LIF layer components."""
        # Dense layer for input connections
        self.dense = nn.Dense(self.hidden_dim, use_bias=True)
        
        # Membrane time constant (discrete)
        self.alpha_mem = jnp.exp(-1.0 / self.tau_mem)
        self.alpha_syn = jnp.exp(-1.0 / self.tau_syn)
    
    def __call__(self, 
                 inputs: jnp.ndarray, 
                 training: bool = True) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Process inputs through LIF layer.
        
        Args:
            inputs: Input tensor [batch_size, time_steps, input_dim]
            training: Training mode flag
            
        Returns:
            Tuple of (output_spikes, membrane_states)
        """
        batch_size, time_steps, input_dim = inputs.shape
        
        # Initialize membrane potential and synaptic current
        v_mem = jnp.zeros((batch_size, self.hidden_dim))
        i_syn = jnp.zeros((batch_size, self.hidden_dim))
        
        output_spikes = []
        membrane_potentials = []
        
        # ✅ LIF DYNAMICS: Simulate for each time step
        for t in range(time_steps):
            # Get input current
            input_current = self.dense(inputs[:, t, :])
            
            # ✅ SYNAPTIC FILTERING: Update synaptic current
            i_syn = self.alpha_syn * i_syn + input_current
            
            # ✅ MEMBRANE INTEGRATION: Update membrane potential
            v_mem = self.alpha_mem * v_mem + (1 - self.alpha_mem) * i_syn
            
            # ✅ SPIKE GENERATION: Generate spikes with surrogate gradients
            # Forward pass: hard threshold
            spikes = (v_mem >= self.threshold).astype(jnp.float32)
            
            # Backward pass: surrogate gradient (fast sigmoid)
            surrogate_grad = self.surrogate_beta / (2 * (1 + jnp.cosh(self.surrogate_beta * (v_mem - self.threshold))))
            
            # Straight-through estimator
            spikes_with_grad = spikes + jax.lax.stop_gradient(spikes - surrogate_grad)
            
            # ✅ RESET: Reset membrane potential after spike
            v_mem = jnp.where(spikes > 0, self.reset_potential, v_mem)
            
            output_spikes.append(spikes_with_grad)
            membrane_potentials.append(v_mem.copy())
        
        # Stack outputs across time
        spike_output = jnp.stack(output_spikes, axis=1)  # [batch, time, hidden_dim]
        membrane_states = jnp.stack(membrane_potentials, axis=1)  # [batch, time, hidden_dim]
        
        # Return outputs and states
        states = {
            'membrane_potentials': membrane_states,
            'final_membrane': v_mem,
            'spike_rate': jnp.mean(spike_output)
        }
        
        return spike_output, states


class DeepSNN(nn.Module):
    """
    Deep 3-layer Spiking Neural Network for classification.
    Executive Summary implementation: deep SNN (256→128→64→classes).
    """
    
    hidden_dims: tuple = (256, 128, 64)
    num_classes: int = 2
    time_steps: int = 16
    tau_mem: float = 20.0
    tau_syn: float = 5.0
    threshold: float = 1.0
    surrogate_beta: float = 4.0
    
    def setup(self):
        """Initialize deep SNN layers."""
        # LIF neuron layers
        self.lif_layers = []
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            self.lif_layers.append(
                LIFLayer(
                    hidden_dim=hidden_dim,
                    tau_mem=self.tau_mem,
                    tau_syn=self.tau_syn,
                    threshold=self.threshold,
                    surrogate_beta=self.surrogate_beta,
                    name=f'lif_layer_{i}'
                )
            )
        
        # Final classification layer
        self.classifier = nn.Dense(self.num_classes)
        
        # Layer normalization for stability
        self.layer_norms = [nn.LayerNorm() for _ in self.hidden_dims]
        
        logger.debug(f"DeepSNN initialized: {len(self.hidden_dims)} layers, "
                    f"dims={self.hidden_dims}, classes={self.num_classes}")
    
    def __call__(self, 
                 spike_trains: jnp.ndarray,
                 training: bool = True) -> Dict[str, jnp.ndarray]:
        """
        Process spike trains through deep SNN.
        
        Args:
            spike_trains: Input spikes [batch_size, time_steps, feature_dim] or [batch_size, time_steps, seq_len, feature_dim]
            training: Training mode flag
            
        Returns:
            Classification outputs and intermediate activations
        """
        # Handle different input shapes
        if len(spike_trains.shape) == 4:
            batch_size, time_steps, seq_len, feature_dim = spike_trains.shape
            # Flatten spatial dimensions for processing
            x = spike_trains.reshape(batch_size, time_steps, -1)
        elif len(spike_trains.shape) == 3:
            batch_size, time_steps, feature_dim = spike_trains.shape
            x = spike_trains
        else:
            raise ValueError(f"Invalid spike_trains shape: {spike_trains.shape}")
        
        # Process through LIF layers
        layer_outputs = []
        layer_states = []
        
        for i, (lif_layer, layer_norm) in enumerate(zip(self.lif_layers, self.layer_norms)):
            # Apply layer normalization to input (except first layer with raw spikes)
            if i == 0:
                x_norm = x  # Don't normalize raw spikes
            else:
                # Apply layer norm across features, maintaining time dimension
                x_norm = layer_norm(x)
            
            # Process through LIF layer
            x, states = lif_layer(x_norm, training=training)
            
            layer_outputs.append(x)
            layer_states.append(states)
        
        # ✅ READOUT: Global average pooling over time
        # [batch_size, time_steps, final_hidden_dim] → [batch_size, final_hidden_dim]
        pooled_output = jnp.mean(x, axis=1)  # Average over time
        
        # ✅ CLASSIFICATION: Final classification
        logits = self.classifier(pooled_output)
        
        # ✅ ANALYSIS: Compute layer statistics
        layer_spike_rates = [jnp.mean(output) for output in layer_outputs]
        
        return {
            'logits': logits,  # [batch_size, num_classes]
            'pooled_features': pooled_output,
            'layer_outputs': layer_outputs,
            'layer_states': layer_states,
            'layer_spike_rates': layer_spike_rates,
            'final_spike_rate': jnp.mean(x)
        }
    
    def get_layer_statistics(self, spike_trains: jnp.ndarray) -> Dict[str, Any]:
        """
        Get detailed statistics for each SNN layer.
        
        Args:
            spike_trains: Input spike trains
            
        Returns:
            Dictionary with layer-wise statistics
        """
        # Forward pass to get layer outputs
        outputs = self(spike_trains, training=False)
        
        statistics = {
            'num_layers': len(self.hidden_dims),
            'layer_dimensions': list(self.hidden_dims),
            'total_parameters': sum(dim * (prev_dim if i > 0 else spike_trains.shape[-1]) 
                                  for i, (dim, prev_dim) in enumerate(zip(self.hidden_dims, [spike_trains.shape[-1]] + list(self.hidden_dims[:-1])))),
            'layer_spike_rates': [float(rate) for rate in outputs['layer_spike_rates']],
            'overall_spike_rate': float(outputs['final_spike_rate']),
            'classification_ready': outputs['logits'].shape[-1] == self.num_classes
        }
        
        # Add membrane potential statistics if available
        if outputs['layer_states']:
            membrane_stats = []
            for i, states in enumerate(outputs['layer_states']):
                if 'membrane_potentials' in states:
                    mem_pot = states['membrane_potentials']
                    membrane_stats.append({
                        'layer': i,
                        'mean_potential': float(jnp.mean(mem_pot)),
                        'std_potential': float(jnp.std(mem_pot)),
                        'max_potential': float(jnp.max(mem_pot)),
                        'threshold_crossings': float(jnp.mean(mem_pot >= self.threshold))
                    })
            
            statistics['membrane_statistics'] = membrane_stats
        
        return statistics


# Export deep SNN components
__all__ = [
    "DeepSNN",
    "LIFLayer"
]
