"""
Memory and gradient monitoring utilities.

This module contains monitoring functions extracted from
training_utils.py for better modularity.

Split from training_utils.py for better maintainability.
"""

import logging
from typing import Dict, Any, Union
import jax
import jax.numpy as jnp
import psutil

logger = logging.getLogger(__name__)


def monitor_memory_usage() -> Dict[str, float]:
    """
    Monitor current memory usage for training optimization.
    
    Returns:
        Dictionary with memory statistics
    """
    # System memory
    memory_info = psutil.virtual_memory()
    
    # Process memory  
    process = psutil.Process()
    process_memory = process.memory_info()
    
    # GPU memory (if available)
    gpu_memory_gb = 0.0
    try:
        devices = jax.devices()
        if devices and hasattr(devices[0], 'memory_stats'):
            memory_stats = devices[0].memory_stats()
            if memory_stats and 'bytes_in_use' in memory_stats:
                gpu_memory_gb = memory_stats['bytes_in_use'] / 1024**3
    except:
        pass
    
    memory_stats = {
        'system_memory_percent': memory_info.percent,
        'system_memory_available_gb': memory_info.available / 1024**3,
        'process_memory_gb': process_memory.rss / 1024**3,
        'gpu_memory_gb': gpu_memory_gb
    }
    
    # Log warnings for high memory usage
    if memory_info.percent > 85:
        logger.warning(f"High memory usage: {memory_info.percent:.1f}%")
    
    if gpu_memory_gb > 14:  # Warn if >14GB on typical 16GB GPU
        logger.warning(f"High GPU memory usage: {gpu_memory_gb:.1f}GB")
    
    return memory_stats


def compute_gradient_norm(grads: Dict) -> float:
    """
    Compute global gradient norm for gradient monitoring.
    
    Args:
        grads: Gradient dictionary (JAX pytree)
        
    Returns:
        Global gradient norm
    """
    # Compute L2 norm of all gradients
    squared_grads = jax.tree.map(lambda g: jnp.sum(g**2), grads)
    total_squared = jax.tree_util.tree_reduce(lambda x, y: x + y, squared_grads)
    grad_norm = jnp.sqrt(total_squared)
    
    return float(grad_norm)


def check_for_nans(values: Union[Dict, jnp.ndarray, float], name: str = "values") -> bool:
    """
    Check for NaN values in parameters, gradients, or losses.
    
    Args:
        values: Values to check (can be dict, array, or scalar)
        name: Name for logging
        
    Returns:
        True if NaNs found, False otherwise
    """
    def has_nan(x):
        return jnp.any(jnp.isnan(x))
    
    if isinstance(values, dict):
        # Check all values in dictionary
        for key, value in values.items():
            if has_nan(value):
                logger.error(f"NaN detected in {name}[{key}]")
                return True
    elif isinstance(values, jnp.ndarray):
        if has_nan(values):
            logger.error(f"NaN detected in {name}")
            return True
    elif isinstance(values, (float, int)):
        if jnp.isnan(values):
            logger.error(f"NaN detected in {name}")
            return True
    
    return False


def monitor_gradient_health(grads: Dict, 
                          step: int,
                          log_frequency: int = 100) -> Dict[str, Any]:
    """
    Monitor gradient health during training.
    
    Args:
        grads: Gradient dictionary
        step: Current training step
        log_frequency: How often to log detailed info
        
    Returns:
        Gradient health metrics
    """
    # Compute gradient statistics
    grad_norm = compute_gradient_norm(grads)
    has_nans = check_for_nans(grads, "gradients")
    
    # Gradient health metrics
    health_metrics = {
        'gradient_norm': grad_norm,
        'has_nans': has_nans,
        'healthy': not has_nans and 1e-10 < grad_norm < 100.0,
        'vanishing': grad_norm < 1e-10,
        'exploding': grad_norm > 100.0
    }
    
    # Detailed logging at specified frequency
    if step % log_frequency == 0:
        logger.info(f"Gradient health (step {step}): norm={grad_norm:.2e}, "
                   f"healthy={health_metrics['healthy']}")
    
    # Warning for problematic gradients
    if health_metrics['vanishing']:
        logger.warning(f"Vanishing gradients detected at step {step}: norm={grad_norm:.2e}")
    elif health_metrics['exploding']:
        logger.warning(f"Exploding gradients detected at step {step}: norm={grad_norm:.2e}")
    elif has_nans:
        logger.error(f"NaN gradients detected at step {step}")
    
    return health_metrics


def check_training_stability(loss_history: list, window_size: int = 20) -> Dict[str, bool]:
    """
    Check training stability based on loss history.
    
    Args:
        loss_history: List of recent loss values
        window_size: Window size for stability analysis
        
    Returns:
        Dictionary with stability indicators
    """
    if len(loss_history) < window_size:
        return {'insufficient_data': True}
    
    recent_losses = jnp.array(loss_history[-window_size:])
    
    # Stability metrics
    loss_variance = jnp.var(recent_losses)
    loss_mean = jnp.mean(recent_losses)
    loss_trend = recent_losses[-1] - recent_losses[0]  # Overall trend
    
    stability = {
        'stable': loss_variance < (loss_mean * 0.1)**2,  # Low relative variance
        'converging': loss_trend < 0,  # Decreasing trend
        'oscillating': loss_variance > (loss_mean * 0.5)**2,  # High variance
        'loss_variance': float(loss_variance),
        'loss_trend': float(loss_trend),
        'relative_variance': float(loss_variance / (loss_mean**2 + 1e-8))
    }
    
    return stability


# Export monitoring functions
__all__ = [
    "monitor_memory_usage",
    "compute_gradient_norm",
    "check_for_nans",
    "monitor_gradient_health",
    "check_training_stability"
]
