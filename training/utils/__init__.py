"""
Training Utils Module: Training Utility Components

Modular implementation of training utilities split from 
training_utils.py for better maintainability.

Components:
- setup: Environment and logging setup utilities
- optimization: JAX optimization and caching functions
- monitoring: Memory and gradient monitoring
- training: Core training support functions
"""

from .setup import (
    setup_professional_logging,
    setup_directories,
    setup_optimized_environment,
    setup_training_environment
)
from .optimization import (
    optimize_jax_for_device,
    cached_cpc_forward,
    cached_spike_bridge,
    cached_snn_forward,
    precompile_training_functions
)
from .monitoring import (
    monitor_memory_usage,
    compute_gradient_norm,
    check_for_nans
)
from .training import (
    ProgressTracker,
    fixed_gradient_accumulation,
    create_optimized_trainer_state,
    format_training_time,
    validate_config,
    save_config_to_file
)

__all__ = [
    # Setup utilities
    "setup_professional_logging",
    "setup_directories", 
    "setup_optimized_environment",
    "setup_training_environment",
    
    # Optimization utilities
    "optimize_jax_for_device",
    "cached_cpc_forward",
    "cached_spike_bridge", 
    "cached_snn_forward",
    "precompile_training_functions",
    
    # Monitoring utilities
    "monitor_memory_usage",
    "compute_gradient_norm",
    "check_for_nans",
    
    # Training utilities
    "ProgressTracker",
    "fixed_gradient_accumulation",
    "create_optimized_trainer_state",
    "format_training_time",
    "validate_config",
    "save_config_to_file"
]
