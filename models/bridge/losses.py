"""
Spike bridge loss functions and surrogate gradients.

This module contains loss functions extracted from gradients.py for 
enhanced modularity according to the refactoring plan.

Created for finer modularity beyond the initial split.
"""

import logging
from typing import Callable
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def sigmoid_surrogate(x: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
    """
    Sigmoid-based surrogate gradient for spike functions.
    
    Args:
        x: Input (membrane potential - threshold)
        beta: Steepness parameter
        
    Returns:
        Surrogate gradient values
    """
    return beta * jax.nn.sigmoid(beta * x) * (1 - jax.nn.sigmoid(beta * x))


def tanh_surrogate(x: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
    """
    Tanh-based surrogate gradient for spike functions.
    
    Args:
        x: Input (membrane potential - threshold)
        beta: Steepness parameter
        
    Returns:
        Surrogate gradient values
    """
    return beta * (1 - jnp.tanh(beta * x) ** 2)


def arctan_surrogate(x: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
    """
    Arctan-based surrogate gradient for spike functions.
    
    Args:
        x: Input (membrane potential - threshold)
        beta: Steepness parameter
        
    Returns:
        Surrogate gradient values
    """
    return beta / (1 + (beta * x) ** 2)


def linear_surrogate(x: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
    """
    Linear surrogate gradient with clipping.
    
    Args:
        x: Input (membrane potential - threshold)
        beta: Steepness parameter
        
    Returns:
        Surrogate gradient values
    """
    return jnp.clip(beta * (1 - jnp.abs(x)), 0, beta)


def spike_rate_loss(spike_trains: jnp.ndarray, 
                   target_rate: float = 0.1,
                   loss_weight: float = 0.01) -> jnp.ndarray:
    """
    Spike rate loss for encouraging target firing rates.
    
    Args:
        spike_trains: Spike trains [batch, time, neurons]
        target_rate: Target spike rate (0.0 to 1.0)
        loss_weight: Loss weighting factor
        
    Returns:
        Spike rate loss value
    """
    # Calculate actual spike rate
    actual_rate = jnp.mean(spike_trains)
    
    # L2 penalty for deviation from target
    rate_penalty = (actual_rate - target_rate) ** 2
    
    return loss_weight * rate_penalty


def temporal_spike_consistency_loss(spike_trains: jnp.ndarray,
                                  consistency_weight: float = 0.01) -> jnp.ndarray:
    """
    Temporal consistency loss for spike patterns.
    
    Encourages smooth temporal evolution of spike patterns.
    
    Args:
        spike_trains: Spike trains [batch, time, neurons]
        consistency_weight: Consistency weight
        
    Returns:
        Temporal consistency loss
    """
    if spike_trains.shape[1] < 2:
        return jnp.array(0.0)
    
    # Calculate temporal differences
    temporal_diffs = jnp.diff(spike_trains, axis=1)
    
    # L2 penalty for large changes
    consistency_penalty = jnp.mean(temporal_diffs ** 2)
    
    return consistency_weight * consistency_penalty


def spike_entropy_loss(spike_trains: jnp.ndarray,
                      entropy_weight: float = 0.01,
                      target_entropy: Optional[float] = None) -> jnp.ndarray:
    """
    Spike entropy loss for encouraging diverse spike patterns.
    
    Args:
        spike_trains: Spike trains [batch, time, neurons]
        entropy_weight: Entropy loss weight
        target_entropy: Target entropy (None for automatic)
        
    Returns:
        Spike entropy loss
    """
    # Calculate spike probabilities per neuron
    spike_probs = jnp.mean(spike_trains, axis=(0, 1))  # [neurons]
    
    # Add small epsilon to avoid log(0)
    spike_probs = jnp.clip(spike_probs, 1e-8, 1.0 - 1e-8)
    
    # Calculate entropy
    entropy = -jnp.sum(spike_probs * jnp.log(spike_probs) + 
                      (1 - spike_probs) * jnp.log(1 - spike_probs))
    
    # Target entropy (maximum entropy for binary variables)
    if target_entropy is None:
        max_entropy = len(spike_probs) * (-0.5 * jnp.log(0.5) * 2)  # Binary entropy
        target_entropy = max_entropy * 0.8  # 80% of maximum
    
    # Penalty for deviation from target entropy
    entropy_penalty = (entropy - target_entropy) ** 2
    
    return entropy_weight * entropy_penalty


def create_surrogate_function(surrogate_type: str, beta: float = 1.0) -> Callable:
    """
    Factory function for creating surrogate gradient functions.
    
    Args:
        surrogate_type: Type of surrogate ('sigmoid', 'tanh', 'arctan', 'linear')
        beta: Steepness parameter
        
    Returns:
        Surrogate gradient function
    """
    surrogate_functions = {
        'sigmoid': lambda x: sigmoid_surrogate(x, beta),
        'tanh': lambda x: tanh_surrogate(x, beta),
        'arctan': lambda x: arctan_surrogate(x, beta),
        'linear': lambda x: linear_surrogate(x, beta)
    }
    
    if surrogate_type not in surrogate_functions:
        logger.warning(f"Unknown surrogate type: {surrogate_type}. Using sigmoid.")
        surrogate_type = 'sigmoid'
    
    return surrogate_functions[surrogate_type]


# Export spike bridge loss functions
__all__ = [
    "sigmoid_surrogate",
    "tanh_surrogate", 
    "arctan_surrogate",
    "linear_surrogate",
    "spike_rate_loss",
    "temporal_spike_consistency_loss",
    "spike_entropy_loss",
    "create_surrogate_function",
    "SpikeProjectionHead"
]

# Add the class for the plan
class SpikeProjectionHead(nn.Module):
    """Spike projection head for compatibility with plan."""
    
    output_dim: int = 256
    
    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.output_dim)(x)
