"""
LHC-Optimized SNN Implementation based on Dillon et al. (2508.00063v1)

This module implements SNN layers optimized based on the LHC physics paper:
"Anomaly detection with spiking neural networks for LHC physics"

Key optimizations:
- Time steps: 5-10 (vs current 32-64)  
- Threshold: 1.2 (vs current 0.55)
- Beta decay: 0.9 (membrane potential decay)
- Surrogate gradients: Optimized for stability
- Multi-step processing with same input

References:
- Dillon et al. "Anomaly detection with spiking neural networks for LHC physics" 
  arXiv:2508.00063v1 [hep-ph] 31 Jul 2025
"""

import logging
from typing import Tuple, Optional, Dict, Any, List
import jax
import jax.numpy as jnp
import flax.linen as nn

logger = logging.getLogger(__name__)


class LHCOptimizedLIFLayer(nn.Module):
    """
    LIF Layer optimized based on LHC paper parameters.
    
    Key differences from standard LIF:
    - Higher threshold (1.2 vs 0.55) for more selective spiking
    - Optimized beta decay (0.9) for stable membrane dynamics
    - Multi-step processing with consistent input
    - Surrogate gradients optimized for anomaly detection
    """
    
    features: int
    time_steps: int = 5  # ✅ LHC OPTIMAL: 5-10 steps (vs current 32)
    threshold: float = 1.2  # ✅ LHC OPTIMAL: 1.2 (vs current 0.55)  
    beta: float = 0.9  # ✅ LHC OPTIMAL: Membrane potential decay
    reset_potential: float = 0.0
    surrogate_beta: float = 4.0
    
    def setup(self):
        """Initialize layer parameters."""
        # Dense layer for input transformation
        self.dense = nn.Dense(
            self.features,
            kernel_init=nn.initializers.xavier_normal(),
            bias_init=nn.initializers.zeros,
            name='lhc_lif_dense'
        )
        
        # Output layer threshold (prevent spiking on final layer)
        self.is_output_layer = False
        
    def __call__(self, x: jnp.ndarray, is_output_layer: bool = False) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Multi-step LIF processing based on LHC paper methodology.
        
        Args:
            x: Input data [batch_size, input_features] 
            is_output_layer: If True, prevent spiking and return potentials
            
        Returns:
            Tuple of (output, metrics) where:
            - output: Final membrane potentials or spike trains
            - metrics: Dictionary with spike statistics
        """
        batch_size = x.shape[0]
        
        # Initialize membrane potentials
        v_mem = jnp.zeros((batch_size, self.features))
        
        # Storage for multi-step processing
        spike_trains = []
        membrane_potentials = []
        
        # ✅ LHC METHODOLOGY: Same input fed at each step
        for step in range(self.time_steps):
            # Input current from dense layer
            input_current = self.dense(x)
            
            # ✅ LHC DYNAMICS: Leaky integration with beta decay
            v_mem = self.beta * v_mem + (1 - self.beta) * input_current
            
            if is_output_layer:
                # ✅ LHC OUTPUT: No spiking on output layer, accumulate potentials
                spikes = jnp.zeros_like(v_mem)
            else:
                # ✅ LHC SPIKING: Generate spikes when threshold exceeded
                spikes = self._generate_spikes(v_mem)
                
                # ✅ LHC RESET: Subtract threshold after spike (better than hard reset)
                v_mem = v_mem - spikes * self.threshold
            
            spike_trains.append(spikes)
            membrane_potentials.append(v_mem.copy())
        
        # ✅ LHC OUTPUT STRATEGY: Use final membrane potentials or spike accumulation
        if is_output_layer:
            # For output layer, return accumulated membrane potentials
            output = v_mem
        else:
            # For hidden layers, return final spike state or accumulated spikes
            output = spike_trains[-1]  # Final step spikes
        
        # Calculate metrics for monitoring
        metrics = self._calculate_metrics(spike_trains, membrane_potentials)
        
        return output, metrics
    
    def _generate_spikes(self, v_mem: jnp.ndarray) -> jnp.ndarray:
        """
        Generate spikes using surrogate gradients (LHC methodology).
        
        Args:
            v_mem: Membrane potentials
            
        Returns:
            Binary spike trains with surrogate gradients
        """
        # ✅ LHC SURROGATE: Hard sigmoid for stable gradients
        def surrogate_spike_fn(v):
            # Forward pass: Heaviside step function
            spikes_hard = (v >= self.threshold).astype(jnp.float32)
            
            # Backward pass: Hard sigmoid surrogate gradient
            surrogate_grad = jnp.clip(
                0.5 + 0.5 * jnp.tanh(self.surrogate_beta * (v - self.threshold)), 
                0.0, 1.0
            )
            
            # Use stop_gradient to separate forward and backward passes
            return spikes_hard + jax.lax.stop_gradient(spikes_hard - surrogate_grad) + surrogate_grad
        
        return surrogate_spike_fn(v_mem)
    
    def _calculate_metrics(self, spike_trains: List[jnp.ndarray], 
                          membrane_potentials: List[jnp.ndarray]) -> Dict[str, Any]:
        """Calculate spike and membrane metrics for monitoring."""
        # Stack time steps
        spikes_stacked = jnp.stack(spike_trains, axis=1)  # [batch, time, features]
        potentials_stacked = jnp.stack(membrane_potentials, axis=1)
        
        # Spike rate statistics
        spike_rate_mean = jnp.mean(spikes_stacked)
        spike_rate_std = jnp.std(spikes_stacked)
        
        # Membrane potential statistics  
        potential_mean = jnp.mean(potentials_stacked)
        potential_std = jnp.std(potentials_stacked)
        
        # Active neurons (neurons that spiked at least once)
        active_neurons = jnp.sum(jnp.any(spikes_stacked > 0, axis=1), axis=1)
        active_neuron_ratio = jnp.mean(active_neurons) / self.features
        
        return {
            'spike_rate_mean': spike_rate_mean,
            'spike_rate_std': spike_rate_std,
            'potential_mean': potential_mean,
            'potential_std': potential_std,
            'active_neuron_ratio': active_neuron_ratio,
            'time_steps_used': self.time_steps,
            'threshold_used': self.threshold
        }


class LHCOptimizedSNN(nn.Module):
    """
    Complete SNN classifier optimized based on LHC paper architecture.
    
    Architecture from paper:
    - Encoder: (19, 24, 12) for LHC data → adapted for GW data
    - Latent: 4 dimensions (binary latent space)
    - Decoder: (12, 24, 19) → adapted for reconstruction
    
    Key optimizations:
    - Small architecture for low-latency requirements
    - Time steps: 5-10 (computational efficiency vs performance trade-off)
    - Binary latent space: 2^(T*Dz) possible configurations
    """
    
    num_classes: int = 2
    input_features: int = 1024  # GW data features (downsampled)
    hidden_sizes: List[int] = None  # Will be set in setup
    latent_dim: int = 4  # ✅ LHC OPTIMAL: Small latent space
    time_steps: int = 5  # ✅ LHC OPTIMAL: 5-10 steps
    threshold: float = 1.2  # ✅ LHC OPTIMAL: Higher threshold
    beta: float = 0.9  # ✅ LHC OPTIMAL: Decay factor
    
    def setup(self):
        """Initialize SNN architecture based on LHC paper."""
        # ✅ LHC ARCHITECTURE: Adapted from (19,24,12) to GW data dimensions
        if self.hidden_sizes is None:
            # Scale architecture based on input features
            scale_factor = self.input_features // 19  # Original LHC input was 19D
            self.hidden_sizes = [
                min(256, 24 * max(1, scale_factor // 4)),  # ~24 scaled
                min(128, 12 * max(1, scale_factor // 8)),  # ~12 scaled
                self.latent_dim
            ]
        
        # Create LIF layers
        self.layers = []
        layer_sizes = [self.input_features] + self.hidden_sizes
        
        for i in range(len(layer_sizes) - 1):
            layer = LHCOptimizedLIFLayer(
                features=layer_sizes[i + 1],
                time_steps=self.time_steps,
                threshold=self.threshold,
                beta=self.beta,
                name=f'lhc_lif_layer_{i}'
            )
            self.layers.append(layer)
        
        # Output classification layer (no spiking)
        self.output_layer = LHCOptimizedLIFLayer(
            features=self.num_classes,
            time_steps=self.time_steps,
            threshold=self.threshold * 10,  # Very high threshold to prevent spiking
            beta=self.beta,
            name='lhc_output_layer'
        )
    
    def __call__(self, x: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Forward pass through LHC-optimized SNN.
        
        Args:
            x: Input features [batch_size, input_features]
            training: Training mode flag
            
        Returns:
            Tuple of (logits, metrics) where:
            - logits: Classification logits [batch_size, num_classes]
            - metrics: Comprehensive metrics from all layers
        """
        current_input = x
        all_metrics = {}
        
        # Process through hidden layers
        for i, layer in enumerate(self.layers):
            current_input, layer_metrics = layer(current_input, is_output_layer=False)
            all_metrics[f'layer_{i}'] = layer_metrics
        
        # Final output layer (no spiking)
        logits, output_metrics = self.output_layer(current_input, is_output_layer=True)
        all_metrics['output'] = output_metrics
        
        # ✅ LHC METRICS: Aggregate statistics for monitoring
        aggregate_metrics = self._aggregate_metrics(all_metrics)
        
        return logits, aggregate_metrics
    
    def _aggregate_metrics(self, all_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across all layers."""
        # Average spike rates across layers (excluding output)
        spike_rates = [metrics['spike_rate_mean'] for key, metrics in all_metrics.items() 
                      if 'layer_' in key]
        
        active_ratios = [metrics['active_neuron_ratio'] for key, metrics in all_metrics.items()
                        if 'layer_' in key]
        
        return {
            'avg_spike_rate': jnp.mean(jnp.array(spike_rates)) if spike_rates else 0.0,
            'avg_active_ratio': jnp.mean(jnp.array(active_ratios)) if active_ratios else 0.0,
            'total_layers': len(self.layers),
            'time_steps': self.time_steps,
            'threshold': self.threshold,
            'architecture': f"LHC-Optimized: {self.hidden_sizes}",
            'latent_configurations': 2**(self.time_steps * self.latent_dim)  # Binary latent space
        }


def create_lhc_optimized_snn(
    num_classes: int = 2,
    input_features: int = 1024,
    time_steps: int = 5,
    threshold: float = 1.2,
    latent_dim: int = 4
) -> LHCOptimizedSNN:
    """
    Factory function to create LHC-optimized SNN with validated parameters.
    
    Args:
        num_classes: Number of output classes
        input_features: Input feature dimension
        time_steps: Number of processing steps (5-10 optimal)
        threshold: Spike threshold (1.2 optimal from LHC paper)
        latent_dim: Latent space dimension (4 optimal for binary space)
        
    Returns:
        Configured LHC-optimized SNN model
    """
    # ✅ LHC VALIDATION: Ensure parameters are in optimal ranges
    if not (5 <= time_steps <= 10):
        logger.warning(f"Time steps {time_steps} outside LHC optimal range [5,10]")
    
    if not (1.0 <= threshold <= 1.5):
        logger.warning(f"Threshold {threshold} outside LHC optimal range [1.0,1.5]")
    
    if latent_dim > 8:
        logger.warning(f"Latent dim {latent_dim} may be too large for binary latent space efficiency")
    
    logger.info(f"Creating LHC-optimized SNN:")
    logger.info(f"  📊 Architecture: {input_features} → latent({latent_dim}) → {num_classes}")
    logger.info(f"  ⚡ Time steps: {time_steps} (LHC optimal: 5-10)")
    logger.info(f"  🎯 Threshold: {threshold} (LHC optimal: 1.2)")
    logger.info(f"  🔥 Binary latent configs: {2**(time_steps * latent_dim):,}")
    
    return LHCOptimizedSNN(
        num_classes=num_classes,
        input_features=input_features,
        latent_dim=latent_dim,
        time_steps=time_steps,
        threshold=threshold,
        beta=0.9  # LHC optimal
    )


# ✅ LHC PERFORMANCE COMPARISON UTILITIES
def compare_snn_architectures(input_data: jnp.ndarray, 
                             current_snn, 
                             lhc_snn) -> Dict[str, Any]:
    """
    Compare current SNN vs LHC-optimized SNN performance.
    
    Args:
        input_data: Test input data
        current_snn: Current SNN model
        lhc_snn: LHC-optimized SNN model
        
    Returns:
        Comparison metrics
    """
    # Run both models
    current_out, current_metrics = current_snn(input_data)
    lhc_out, lhc_metrics = lhc_snn(input_data)
    
    return {
        'current_model': {
            'output_shape': current_out.shape,
            'metrics': current_metrics
        },
        'lhc_optimized': {
            'output_shape': lhc_out.shape,
            'metrics': lhc_metrics
        },
        'efficiency_gain': {
            'time_steps_reduction': f"{current_metrics.get('time_steps', 32)} → {lhc_metrics['time_steps']}",
            'threshold_change': f"{current_metrics.get('threshold', 0.55)} → {lhc_metrics['threshold']}",
            'latent_configs': lhc_metrics['latent_configurations']
        }
    }
