"""
Core training support utilities.

This module contains training support functions extracted from
training_utils.py for better modularity.

Split from training_utils.py for better maintainability.
"""

import time
import logging
import json
from typing import Dict, Any, Optional, Callable, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

logger = logging.getLogger(__name__)


@dataclass  
class ProgressTracker:
    """Track training progress with performance metrics."""
    
    total_steps: int
    log_interval: int = 50
    start_time: float = 0.0
    step_times: list = None
    
    def __post_init__(self):
        if self.step_times is None:
            self.step_times = []
        if self.start_time == 0.0:
            self.start_time = time.time()
    
    def update(self, step: int, metrics: Dict[str, Any]):
        """Update progress tracker with current step metrics."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Record step time
        if len(self.step_times) > 0:
            step_time = current_time - (self.start_time + sum(self.step_times))
            self.step_times.append(step_time)
        else:
            self.step_times.append(0.1)  # Initial estimate
        
        # Keep only recent step times
        if len(self.step_times) > 100:
            self.step_times = self.step_times[-50:]
        
        # Log progress at intervals
        if step % self.log_interval == 0:
            progress_percent = (step / self.total_steps) * 100
            avg_step_time = jnp.mean(jnp.array(self.step_times[-10:]))  # Recent average
            
            eta_seconds = (self.total_steps - step) * avg_step_time
            eta_formatted = format_training_time(0, eta_seconds)
            
            logger.info(f"Progress: {step}/{self.total_steps} ({progress_percent:.1f}%) - "
                       f"ETA: {eta_formatted}, step_time: {avg_step_time:.3f}s")


def format_training_time(current_time: float, total_time: Optional[float] = None) -> str:
    """Format training time for human-readable display."""
    if total_time is None:
        # Format single duration
        hours = int(current_time // 3600)
        minutes = int((current_time % 3600) // 60)
        seconds = int(current_time % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    else:
        # Format progress (current/total)
        current_formatted = format_training_time(current_time)
        total_formatted = format_training_time(total_time)
        percent = (current_time / total_time * 100) if total_time > 0 else 0
        
        return f"{current_formatted}/{total_formatted} ({percent:.1f}%)"


def fixed_gradient_accumulation(loss_fn: Callable,
                               params: Dict,
                               batch: Any,
                               accumulation_steps: int = 4) -> Tuple[jnp.ndarray, Dict]:
    """
    ✅ CRITICAL FIX: Fixed gradient accumulation with proper loss scaling.
    
    Previous bug: Divided gradients without scaling loss → wrong gradient magnitudes.
    Fix: Scale loss by accumulation steps, keep gradients unscaled.
    
    Args:
        loss_fn: Loss function to compute gradients for
        params: Model parameters
        batch: Training batch
        accumulation_steps: Number of accumulation steps
        
    Returns:
        Tuple of (scaled_loss, accumulated_gradients)
    """
    # Split batch into micro-batches (supports (x,y) tuple or single array)
    if hasattr(batch, 'shape'):
        batch_size = batch.shape[0]
        get_slice = lambda b, s, e: b[s:e]
    elif isinstance(batch, (tuple, list)) and hasattr(batch[0], 'shape'):
        batch_size = batch[0].shape[0]
        def get_slice(b, s, e):
            return (b[0][s:e], b[1][s:e])
    else:
        raise ValueError("Unsupported batch format for gradient accumulation")
    micro_batch_size = max(1, batch_size // accumulation_steps)
    
    accumulated_loss = 0.0
    accumulated_grads = None
    
    for i in range(accumulation_steps):
        # Get micro-batch
        start_idx = i * micro_batch_size
        end_idx = min(start_idx + micro_batch_size, batch_size)
        micro_batch = get_slice(batch, start_idx, end_idx)
        
        if len(micro_batch) == 0:
            continue
        
        # Compute loss and gradients for micro-batch
        def micro_loss_fn(p):
            return loss_fn(p, micro_batch)
        
        loss, grads = jax.value_and_grad(micro_loss_fn)(params)
        
        # ✅ CRITICAL FIX: Scale loss, not gradients
        scaled_loss = loss / accumulation_steps
        accumulated_loss += scaled_loss
        
        # Accumulate gradients (no scaling!)
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = jax.tree.map(
                lambda acc_g, new_g: acc_g + new_g,
                accumulated_grads, grads
            )
    
    # ✅ FINAL SCALING: Scale accumulated gradients by steps (not loss!)
    if accumulated_grads is not None:
        final_grads = jax.tree.map(
            lambda g: g / accumulation_steps,
            accumulated_grads
        )
    else:
        final_grads = jax.tree.map(jnp.zeros_like, params)
    
    return accumulated_loss, final_grads


def create_optimized_trainer_state(model, 
                                 learning_rate: float = 0.001,
                                 optimizer_type: str = "adamw") -> train_state.TrainState:
    """
    Create optimized training state with proper initialization.
    
    Args:
        model: Model to create training state for
        learning_rate: Learning rate
        optimizer_type: Type of optimizer ("adam", "adamw", "sgd")
        
    Returns:
        Initialized TrainState
    """
    # Create optimizer based on type
    if optimizer_type == "adamw":
        optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-5)
    elif optimizer_type == "adam":
        optimizer = optax.adam(learning_rate=learning_rate)
    elif optimizer_type == "sgd":
        optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)
    else:
        logger.warning(f"Unknown optimizer: {optimizer_type}. Using AdamW.")
        optimizer = optax.adamw(learning_rate=learning_rate)
    
    # Add gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optimizer
    )
    
    # Initialize dummy parameters (actual initialization happens in trainer)
    dummy_params = {'placeholder': jnp.array([1.0])}
    
    # Create training state
    trainer_state = train_state.TrainState.create(
        apply_fn=model.apply if hasattr(model, 'apply') else lambda params, x: x,
        params=dummy_params,
        tx=optimizer
    )
    
    logger.info(f"Optimized trainer state created: {optimizer_type}, lr={learning_rate}")
    return trainer_state


def validate_config(config: Any) -> bool:
    """
    Validate training configuration.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check if config has validate method
        if hasattr(config, 'validate'):
            return config.validate()
        
        # Basic validation for dict-like configs
        if hasattr(config, '__dict__') or isinstance(config, dict):
            config_dict = config.__dict__ if hasattr(config, '__dict__') else config
            
            # Check required fields
            required_fields = ['learning_rate', 'batch_size', 'num_epochs']
            for field in required_fields:
                if field not in config_dict:
                    logger.error(f"Missing required config field: {field}")
                    return False
            
            # Check value ranges
            if config_dict.get('learning_rate', 0) <= 0:
                logger.error("learning_rate must be positive")
                return False
                
            if config_dict.get('batch_size', 0) <= 0:
                logger.error("batch_size must be positive")
                return False
                
            if config_dict.get('num_epochs', 0) <= 0:
                logger.error("num_epochs must be positive")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Config validation error: {e}")
        return False


def save_config_to_file(config: Any, filepath: Union[str, Path]):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert config to dictionary
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    elif isinstance(config, dict):
        config_dict = config
    else:
        logger.warning("Config type not supported for saving")
        return
    
    # Convert JAX arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, jnp.ndarray):
            return obj.tolist()
        elif isinstance(obj, (jnp.int32, jnp.int64, jnp.float32, jnp.float64)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    serializable_config = convert_for_json(config_dict)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    logger.info(f"Configuration saved to {filepath}")


def estimate_training_time(num_epochs: int,
                         steps_per_epoch: int,
                         avg_step_time: float) -> Dict[str, Any]:
    """
    Estimate total training time.
    
    Args:
        num_epochs: Number of training epochs
        steps_per_epoch: Steps per epoch
        avg_step_time: Average time per step in seconds
        
    Returns:
        Dictionary with time estimates
    """
    total_steps = num_epochs * steps_per_epoch
    total_seconds = total_steps * avg_step_time
    
    estimates = {
        'total_steps': total_steps,
        'total_seconds': total_seconds,
        'total_formatted': format_training_time(total_seconds),
        'steps_per_hour': 3600 / avg_step_time if avg_step_time > 0 else 0,
        'epochs_per_hour': (3600 / avg_step_time) / steps_per_epoch if avg_step_time > 0 and steps_per_epoch > 0 else 0
    }
    
    return estimates


# Export training support functions
__all__ = [
    "ProgressTracker",
    "format_training_time",
    "fixed_gradient_accumulation",
    "create_optimized_trainer_state",
    "validate_config",
    "save_config_to_file",
    "estimate_training_time"
]
