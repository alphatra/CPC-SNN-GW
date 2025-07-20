"""
Training Utilities: Common Helper Functions

Shared utilities for training infrastructure:
- Logging setup with professional formatting
- Device optimization and memory management  
- Directory and path management
- Common validation and error handling
- JAX compilation utilities
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


def setup_professional_logging(log_level: int = logging.INFO, 
                              log_file: Optional[str] = None,
                              format_string: Optional[str] = None) -> logging.Logger:
    """
    Setup professional logging with consistent formatting across all trainers.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional file path for logging output
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    
    handlers = [console_handler]
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # Remove existing handlers
    root_logger.setLevel(log_level)
    
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Suppress verbose third-party loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return root_logger


def setup_directories(output_dir: str, 
                     checkpoint_dir: Optional[str] = None,
                     log_dir: Optional[str] = None) -> Dict[str, Path]:
    """
    Create and setup necessary directories for training.
    
    Args:
        output_dir: Main output directory
        checkpoint_dir: Checkpoint directory (default: output_dir/checkpoints)
        log_dir: Log directory (default: output_dir/logs)
        
    Returns:
        Dictionary with Path objects for each directory
    """
    output_path = Path(output_dir)
    
    dirs = {
        'output': output_path,
        'checkpoint': Path(checkpoint_dir) if checkpoint_dir else output_path / 'checkpoints',
        'log': Path(log_dir) if log_dir else output_path / 'logs',
        'plots': output_path / 'plots',
        'metrics': output_path / 'metrics'
    }
    
    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    return dirs


def optimize_jax_for_device(max_memory_gb: float = 8.0,
                           enable_x64: bool = False,
                           preallocate: bool = False) -> Dict[str, Any]:
    """
    Optimize JAX settings for the current device (CPU/Metal/CUDA).
    
    Args:
        max_memory_gb: Maximum memory to use (GB)
        enable_x64: Enable 64-bit precision
        preallocate: Preallocate GPU memory
        
    Returns:
        Dictionary with device info and settings
    """
    # Configure JAX memory settings
    if not preallocate:
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    # Set memory fraction
    memory_fraction = min(max_memory_gb / 16.0, 0.9)  # Assume 16GB max
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(memory_fraction)
    
    # Configure precision
    if enable_x64:
        jax.config.update('jax_enable_x64', True)
    
    # Get device information
    devices = jax.devices()
    platform = jax.lib.xla_bridge.get_backend().platform
    
    device_info = {
        'platform': platform,
        'devices': [str(d) for d in devices],
        'device_count': len(devices),
        'memory_fraction': memory_fraction,
        'x64_enabled': enable_x64,
        'preallocate': preallocate
    }
    
    logger.info(f"JAX Platform: {platform}")
    logger.info(f"Available devices: {device_info['devices']}")
    logger.info(f"Memory fraction: {memory_fraction:.2f}")
    
    return device_info


def validate_config(config: Any, required_fields: List[str]) -> bool:
    """
    Validate that a configuration object has all required fields.
    
    Args:
        config: Configuration object to validate
        required_fields: List of required field names
        
    Returns:
        True if valid, raises ValueError if not
    """
    missing_fields = []
    
    for field in required_fields:
        if not hasattr(config, field):
            missing_fields.append(field)
    
    if missing_fields:
        raise ValueError(f"Missing required config fields: {missing_fields}")
    
    # Validate specific field types and ranges
    if hasattr(config, 'batch_size') and config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    if hasattr(config, 'learning_rate') and config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    
    if hasattr(config, 'num_epochs') and config.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    
    return True


def save_config_to_file(config: Any, filepath: str) -> None:
    """
    Save configuration to JSON file for reproducibility.
    
    Args:
        config: Configuration object to save
        filepath: Path to save the config file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert dataclass to dict if needed
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = dict(config)
    
    # Handle non-serializable types
    def serialize_value(obj):
        if isinstance(obj, (np.ndarray, jnp.ndarray)):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return obj
    
    serializable_config = {k: serialize_value(v) for k, v in config_dict.items()}
    
    with open(filepath, 'w') as f:
        json.dump(serializable_config, f, indent=2, default=str)
    
    logger.info(f"Saved configuration to: {filepath}")


@jax.jit
def compute_gradient_norm(grads) -> jnp.ndarray:
    """
    Compute the L2 norm of gradients across all parameters.
    
    Args:
        grads: Gradient tree from JAX
        
    Returns:
        L2 norm of all gradients
    """
    leaves = jax.tree_leaves(grads)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


def check_for_nans(values: Dict[str, Any], step: int) -> bool:
    """
    Check for NaN values in training metrics and log warnings.
    
    Args:
        values: Dictionary of values to check
        step: Current training step
        
    Returns:
        True if NaNs found, False otherwise
    """
    nan_found = False
    
    for key, value in values.items():
        if isinstance(value, (float, int, np.ndarray, jnp.ndarray)):
            if jnp.any(jnp.isnan(jnp.asarray(value))):
                logger.warning(f"NaN detected in {key} at step {step}")
                nan_found = True
    
    return nan_found


def format_training_time(start_time: float, current_time: float) -> str:
    """
    Format elapsed training time in human-readable format.
    
    Args:
        start_time: Training start timestamp
        current_time: Current timestamp
        
    Returns:
        Formatted time string
    """
    elapsed = current_time - start_time
    
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    elif elapsed < 3600:
        return f"{elapsed/60:.1f}m"
    else:
        return f"{elapsed/3600:.1f}h"


def create_experiment_summary(config: Any, 
                            metrics: Dict[str, Any],
                            start_time: float,
                            end_time: float) -> Dict[str, Any]:
    """
    Create a comprehensive experiment summary for reporting.
    
    Args:
        config: Training configuration
        metrics: Final training metrics
        start_time: Experiment start time
        end_time: Experiment end time
        
    Returns:
        Dictionary with experiment summary
    """
    summary = {
        'experiment_info': {
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
            'end_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)),
            'duration': format_training_time(start_time, end_time),
            'platform': jax.lib.xla_bridge.get_backend().platform,
            'devices': [str(d) for d in jax.devices()]
        },
        'configuration': config.__dict__ if hasattr(config, '__dict__') else dict(config),
        'final_metrics': metrics,
        'success': True
    }
    
    return summary


class ProgressTracker:
    """Simple progress tracking utility for training loops."""
    
    def __init__(self, total_steps: int, log_interval: int = 100):
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.start_time = time.time()
        self.step_times = []
    
    def update(self, step: int, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Update progress and log if necessary."""
        current_time = time.time()
        self.step_times.append(current_time)
        
        if step % self.log_interval == 0 or step == self.total_steps - 1:
            progress_pct = (step + 1) / self.total_steps * 100
            elapsed_time = current_time - self.start_time
            
            if len(self.step_times) > 1:
                recent_times = self.step_times[-self.log_interval:]
                avg_step_time = (recent_times[-1] - recent_times[0]) / len(recent_times)
                eta = avg_step_time * (self.total_steps - step - 1)
                eta_str = format_training_time(0, eta)
            else:
                eta_str = "N/A"
            
            msg = f"Progress: {step+1}/{self.total_steps} ({progress_pct:.1f}%) - "
            msg += f"Elapsed: {format_training_time(self.start_time, current_time)} - "
            msg += f"ETA: {eta_str}"
            
            if metrics:
                metric_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items() 
                                       if isinstance(v, (int, float))])
                msg += f" - {metric_str}"
            
            logger.info(msg) 