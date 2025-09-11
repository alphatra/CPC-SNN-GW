"""
Environment and logging setup utilities.

This module contains setup functions extracted from
training_utils.py for better modularity.

Split from training_utils.py for better maintainability.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def setup_professional_logging(level=logging.INFO, log_file=None):
    """Setup professional logging configuration (idempotent, no duplicates)."""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to prevent duplicate logs
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.info("Professional logging setup completed")
    return root_logger


def setup_directories(output_dir: str) -> Dict[str, Path]:
    """Create and setup training directories."""
    base_dir = Path(output_dir)
    
    directories = {
        'base': base_dir,
        'log': base_dir / 'logs',
        'checkpoints': base_dir / 'checkpoints', 
        'models': base_dir / 'models',
        'config': base_dir / 'configs',
        'plots': base_dir / 'plots',
        'tensorboard': base_dir / 'tensorboard'
    }
    
    # Create all directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training directories setup: {len(directories)} directories in {base_dir}")
    return directories


def setup_optimized_environment(memory_fraction: float = 0.5) -> None:
    """
    Setup JAX environment with optimized settings.
    
    CRITICAL FIXES applied from Memory Bank:
    - Memory fraction: 0.9 → 0.5 (prevent swap on 16GB systems)
    - Pre-allocation disabled (prevent CUDA OOM)
    """
    # ✅ MEMORY FIX: Reduce memory fraction to prevent swap
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(memory_fraction)
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    # ✅ PERFORMANCE: Enable JIT compilation caching
    os.environ['XLA_FLAGS'] = '--xla_gpu_enable_async_collectives=true'
    
    # ✅ STABILITY: Additional stability flags
    os.environ['JAX_ENABLE_X64'] = 'false'  # Use float32 for memory efficiency
    
    logger.info(f"JAX environment optimized: memory_fraction={memory_fraction}")


def setup_training_environment(memory_fraction: float = 0.5) -> None:
    """
    Complete training environment setup.
    
    Combines all setup functions for one-call initialization.
    
    Args:
        memory_fraction: JAX memory fraction
    """
    logger.info("Setting up complete training environment...")
    
    # Setup JAX environment
    setup_optimized_environment(memory_fraction)
    
    # Setup logging (if not already done)
    if not logging.getLogger().handlers:
        setup_professional_logging()
    
    logger.info("✅ Complete training environment setup finished")


def get_system_info() -> Dict[str, Any]:
    """Get system information for training environment validation."""
    import psutil
    import jax
    
    # System info
    memory_info = psutil.virtual_memory()
    cpu_info = psutil.cpu_count()
    
    # JAX info
    jax_devices = jax.devices()
    jax_backend = jax.default_backend()
    
    system_info = {
        'cpu_cores': cpu_info,
        'memory_total_gb': memory_info.total / 1024**3,
        'memory_available_gb': memory_info.available / 1024**3,
        'memory_percent_used': memory_info.percent,
        'jax_backend': jax_backend,
        'jax_devices': len(jax_devices),
        'jax_device_types': [str(device.device_kind) for device in jax_devices]
    }
    
    return system_info


def validate_training_environment() -> bool:
    """Validate that training environment is properly configured."""
    try:
        # Check JAX setup
        import jax
        import jax.numpy as jnp
        
        # Test basic JAX operation
        test_array = jnp.array([1.0, 2.0, 3.0])
        test_result = jnp.sum(test_array)
        test_result.block_until_ready()
        
        # Check memory settings
        env_vars_ok = (
            'XLA_PYTHON_CLIENT_MEM_FRACTION' in os.environ and
            'XLA_PYTHON_CLIENT_PREALLOCATE' in os.environ
        )
        
        # Check directories can be created
        test_dir = Path("test_training_env")
        test_dir.mkdir(exist_ok=True)
        test_dir.rmdir()
        
        validation_result = env_vars_ok and test_result is not None
        
        if validation_result:
            logger.info("✅ Training environment validation PASSED")
        else:
            logger.error("❌ Training environment validation FAILED")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Training environment validation error: {e}")
        return False


# Export setup functions
__all__ = [
    "setup_professional_logging",
    "setup_directories",
    "setup_optimized_environment", 
    "setup_training_environment",
    "get_system_info",
    "validate_training_environment"
]
