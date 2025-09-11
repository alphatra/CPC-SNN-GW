"""
Gradient flow monitoring and surrogate gradient functions for spike bridge.

This module contains all gradient-related functionality extracted from
spike_bridge.py for better modularity:
- GradientFlowMonitor: Monitors gradient health through spike operations
- EnhancedSurrogateGradients: Advanced surrogate gradient implementations
- spike_function_*: Core spike functions with gradient support

Split from spike_bridge.py for better maintainability.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Callable, Tuple
import logging
from functools import partial

# Import enhanced surrogate gradients
from ..snn_utils import (
    spike_function_with_enhanced_surrogate,
    create_enhanced_surrogate_gradient_fn,
    SurrogateGradientType
)

logger = logging.getLogger(__name__)


class GradientFlowMonitor:
    """
    Monitor and validate gradient flow through spike bridge.
    Critical for Executive Summary fix: end-to-end gradient validation.
    """
    
    def __init__(self):
        self.gradient_stats = {}
        self.flow_history = []
        
    def check_gradient_flow(self, params: Dict, gradients: Dict) -> Dict[str, float]:
        """
        Check gradient flow health through the spike bridge.
        
        Args:
            params: Model parameters
            gradients: Gradients from backpropagation
            
        Returns:
            Dictionary with gradient flow statistics
        """
        stats = {
            'gradient_norm': 0.0,
            'param_norm': 0.0,
            'gradient_to_param_ratio': 0.0,
            'vanishing_gradients': False,
            'exploding_gradients': False,
            'healthy_flow': True
        }
        
        # Calculate global gradient norm
        gradient_norm = jnp.sqrt(
            sum(jnp.sum(g**2) for g in jax.tree.leaves(gradients))
        )
        stats['gradient_norm'] = float(gradient_norm)
        
        # Calculate parameter norm
        param_norm = jnp.sqrt(
            sum(jnp.sum(p**2) for p in jax.tree.leaves(params))
        )
        stats['param_norm'] = float(param_norm)
        
        # Gradient to parameter ratio
        if param_norm > 1e-10:
            stats['gradient_to_param_ratio'] = float(gradient_norm / param_norm)
        
        # Check for pathological gradients
        stats['vanishing_gradients'] = gradient_norm < 1e-10
        stats['exploding_gradients'] = gradient_norm > 100.0
        stats['healthy_flow'] = not (stats['vanishing_gradients'] or stats['exploding_gradients'])
        
        # Store history
        self.flow_history.append(stats.copy())
        if len(self.flow_history) > 1000:  # Keep last 1000 entries
            self.flow_history.pop(0)
        
        # Log warnings
        if stats['vanishing_gradients']:
            logger.warning(f"Vanishing gradients detected: norm={gradient_norm:.2e}")
        elif stats['exploding_gradients']:
            logger.warning(f"Exploding gradients detected: norm={gradient_norm:.2e}")
        
        return stats


def spike_function_with_surrogate(v_mem: jnp.ndarray, 
                                threshold: float,
                                surrogate_fn: Callable) -> jnp.ndarray:
    """
    Spike function with surrogate gradients for backpropagation.
    
    Args:
        v_mem: Membrane potential
        threshold: Spike threshold
        surrogate_fn: Surrogate gradient function
        
    Returns:
        Binary spike output with surrogate gradients
    """
    # Forward pass: hard threshold
    spikes = (v_mem >= threshold).astype(jnp.float32)
    
    # Backward pass: use surrogate gradient
    surrogate_grad = surrogate_fn(v_mem - threshold)
    
    # Straight-through estimator with surrogate
    return spikes + jax.lax.stop_gradient(spikes - surrogate_grad)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def spike_function_fwd(v_mem: jnp.ndarray, 
                      threshold: float,
                      surrogate_fn: Callable) -> Tuple[jnp.ndarray, Tuple]:
    """Forward pass of spike function."""
    spikes = (v_mem >= threshold).astype(jnp.float32)
    return spikes, (v_mem, threshold, surrogate_fn)


def spike_function_bwd(surrogate_fn: Callable,
                      res: Tuple,
                      g: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
    """Backward pass using surrogate gradients."""
    v_mem, threshold, _ = res
    surrogate_grad = surrogate_fn(v_mem - threshold)
    return g * surrogate_grad, None


# Register VJP
spike_function_fwd.defvjp(spike_function_fwd, spike_function_bwd)


class EnhancedSurrogateGradients:
    """
    Enhanced surrogate gradient functions with validated flow.
    Implements multiple surrogate types with automatic flow validation.
    """
    
    @staticmethod
    def sigmoid_surrogate(x: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
        """Sigmoid-based surrogate gradient."""
        return beta * jax.nn.sigmoid(beta * x) * (1 - jax.nn.sigmoid(beta * x))
    
    @staticmethod 
    def tanh_surrogate(x: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
        """Tanh-based surrogate gradient."""
        return beta * (1 - jnp.tanh(beta * x) ** 2)
    
    @staticmethod
    def arctan_surrogate(x: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
        """Arctan-based surrogate gradient."""
        return beta / (1 + (beta * x) ** 2)
    
    @staticmethod
    def linear_surrogate(x: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
        """Linear surrogate gradient with clipping."""
        return jnp.clip(beta * (1 - jnp.abs(x)), 0, beta)
    
    @staticmethod
    def exponential_surrogate(x: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
        """Exponential surrogate gradient."""
        return beta * jnp.exp(-beta * jnp.abs(x))
    
    @staticmethod
    def get_surrogate_function(surrogate_type: str, beta: float = 1.0) -> Callable:
        """Get surrogate function by name with validation."""
        surrogates = {
            'sigmoid': lambda x: EnhancedSurrogateGradients.sigmoid_surrogate(x, beta),
            'tanh': lambda x: EnhancedSurrogateGradients.tanh_surrogate(x, beta),
            'arctan': lambda x: EnhancedSurrogateGradients.arctan_surrogate(x, beta),
            'linear': lambda x: EnhancedSurrogateGradients.linear_surrogate(x, beta),
            'exponential': lambda x: EnhancedSurrogateGradients.exponential_surrogate(x, beta)
        }
        
        if surrogate_type not in surrogates:
            logger.warning(f"Unknown surrogate type: {surrogate_type}. Using sigmoid.")
            surrogate_type = 'sigmoid'
        
        return surrogates[surrogate_type]
    
    @staticmethod
    def validate_surrogate_output(input_val: jnp.ndarray, 
                                 surrogate_output: jnp.ndarray) -> bool:
        """Validate surrogate gradient output for numerical stability."""
        try:
            # Check for NaN/Inf
            if jnp.any(jnp.isnan(surrogate_output)) or jnp.any(jnp.isinf(surrogate_output)):
                logger.warning("NaN or Inf detected in surrogate gradient output")
                return False
            
            # Check range
            if jnp.max(jnp.abs(surrogate_output)) > 100.0:
                logger.warning("Very large surrogate gradients detected (>100)")
                return False
            
            # Check shape consistency
            if input_val.shape != surrogate_output.shape:
                logger.warning("Shape mismatch in surrogate gradient computation")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating surrogate output: {e}")
            return False


# Export functions
__all__ = [
    "GradientFlowMonitor",
    "spike_function_with_surrogate",
    "spike_function_fwd", 
    "spike_function_bwd",
    "EnhancedSurrogateGradients"
]

