"""
Utilities for CPC+SNN Neuromorphic GW Detection

Production-ready utilities following ML4GW standards.
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Optional, Union

import yaml
import jax
import jax.numpy as jnp


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    force: bool = False
) -> None:
    """Setup production-ready logging configuration.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional file path for log output
        format_string: Custom log format string
        force: If True, removes existing handlers before setup
    """
    if format_string is None:
        format_string = (
            "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s"
        )
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Clear any existing handlers only if force=True
    root_logger = logging.getLogger()
    if force:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set root level
    root_logger.setLevel(level)
    
    # Set JAX logging level to reduce noise
    jax_logger = logging.getLogger("jax")
    jax_logger.setLevel(logging.WARNING)


# Configuration utilities now available in ligo_cpc_snn.utils.config
# Use load_config() and save_config() from there for dataclass-based configuration


def get_jax_device_info() -> dict:
    """Get JAX device information for logging and debugging.
    
    Returns:
        Dictionary with device information
    """
    devices = jax.devices()
    
    device_info = {
        'num_devices': len(devices),
        'devices': [
            {
                'id': i,
                'device_kind': str(device.device_kind),
                'platform': str(device.platform),
            }
            for i, device in enumerate(devices)
        ],
        'default_backend': jax.default_backend(),
    }
    
    # Try to get memory info (may not be available on all platforms)
    try:
        if devices and hasattr(devices[0], 'memory_stats'):
            memory_stats = devices[0].memory_stats()
            device_info['memory_info'] = memory_stats
    except Exception:
        pass
    
    return device_info


def print_system_info() -> None:
    """Print system and JAX configuration information."""
    logger = logging.getLogger(__name__)
    
    # JAX information
    device_info = get_jax_device_info()
    
    logger.info("🖥️  System Information:")
    logger.info(f"   JAX backend: {device_info['default_backend']}")
    logger.info(f"   Available devices: {device_info['num_devices']}")
    
    for device in device_info['devices']:
        logger.info(
            f"     Device {device['id']}: {device['device_kind']} "
            f"({device['platform']})"
        )
    
    if 'memory_info' in device_info:
        memory_info = device_info['memory_info'] 
        if memory_info and 'bytes_in_use' in memory_info:
            memory_gb = memory_info['bytes_in_use'] / (1024**3)
            logger.info(f"   Memory in use: {memory_gb:.2f} GB")


def validate_array_shape(
    array: jnp.ndarray, 
    expected_shape: tuple,
    array_name: str = "array"
) -> None:
    """Validate array shape matches expected shape.
    
    Args:
        array: Input array to validate
        expected_shape: Expected shape tuple (use None or -1 for flexible dimensions)
        array_name: Name of array for error messages
        
    Raises:
        ValueError: If shape doesn't match
    """
    actual_shape = array.shape
    
    if len(actual_shape) != len(expected_shape):
        raise ValueError(
            f"{array_name} has {len(actual_shape)} dimensions, "
            f"expected {len(expected_shape)}"
        )
    
    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if expected is not None and expected != -1 and actual != expected:
            raise ValueError(
                f"{array_name} dimension {i} has size {actual}, "
                f"expected {expected}"
            )


def create_directory_structure(base_path: Union[str, Path], 
                             subdirs: list[str]) -> Path:
    """Create standardized directory structure for ML4GW projects.
    
    Args:
        base_path: Base directory path
        subdirs: List of subdirectory names to create
        
    Returns:
        Path to created base directory
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    for subdir in subdirs:
        (base_path / subdir).mkdir(exist_ok=True)
    
    return base_path


# Standard ML4GW project structure
ML4GW_PROJECT_STRUCTURE = [
    "data",
    "models", 
    "logs",
    "outputs",
    "configs",
    "checkpoints",
    "plots",
    "results",
]


__all__ = [
    "setup_logging",
    "get_jax_device_info",
    "print_system_info",
    "validate_array_shape",
    "create_directory_structure",
    "ML4GW_PROJECT_STRUCTURE",
    # Deprecated functions
    "safe_array_to_device",
    "validate_jax_array",
]


# ===== DEPRECATED COMPATIBILITY WRAPPERS =====
# For functions that were removed from deleted files

def safe_array_to_device(array, device=None, dtype=None):
    """Deprecated wrapper for safe_array_to_device. Use jax.device_put instead."""
    warnings.warn(
        "safe_array_to_device from jax_safety is deprecated. "
        "Use jax.device_put(jnp.array(...)) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    try:
        import numpy as np
        
        # Convert to array if needed
        if not isinstance(array, (jnp.ndarray, np.ndarray)):
            array = np.array(array, dtype=dtype)
        else:
            array = jnp.array(array, dtype=dtype) if dtype else jnp.array(array)
            
        # Put on device
        if device is not None:
            return jax.device_put(array, device)
        else:
            return jax.device_put(array)
    except Exception as e:
        logging.getLogger(__name__).error(f"safe_array_to_device fallback failed: {e}")
        raise


def validate_jax_array(array, min_dims=None, max_dims=None, dtypes=None):
    """Deprecated wrapper for validate_jax_array. Use validate_array_shape instead."""
    warnings.warn(
        "validate_jax_array from jax_safety is deprecated. "
        "Use validate_array_shape instead.",
        DeprecationWarning,
        stacklevel=2
    )
    try:
        # Basic validation using existing function
        if min_dims is not None and len(array.shape) < min_dims:
            raise ValueError(f"Array has {len(array.shape)} dimensions, minimum required: {min_dims}")
        
        if max_dims is not None and len(array.shape) > max_dims:
            raise ValueError(f"Array has {len(array.shape)} dimensions, maximum allowed: {max_dims}")
            
        if dtypes is not None:
            if not isinstance(dtypes, (list, tuple)):
                dtypes = [dtypes]
            if array.dtype not in dtypes:
                raise ValueError(f"Array dtype {array.dtype} not in allowed types: {dtypes}")
                
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"validate_jax_array fallback failed: {e}")
        raise
