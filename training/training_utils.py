"""
Training Utils (MODULAR)

This file delegates to modular training utility components for better maintainability.
The actual implementation has been split into:
- utils/setup.py: Environment and logging setup
- utils/optimization.py: JAX optimization and caching
- utils/monitoring.py: Memory and gradient monitoring  
- utils/training.py: Core training support functions

This file maintains backward compatibility through delegation.

Training Utilities: Performance-Optimized Training Infrastructure
"""

import logging
import warnings

# Import from new modular components
from .utils import (
    # Setup utilities
    setup_professional_logging,
    setup_directories,
    setup_optimized_environment,
    setup_training_environment,
    
    # Optimization utilities
    optimize_jax_for_device,
    cached_cpc_forward,
    cached_spike_bridge,
    cached_snn_forward,
    precompile_training_functions,
    
    # Monitoring utilities
    monitor_memory_usage,
    compute_gradient_norm,
    check_for_nans,
    
    # Training utilities
    ProgressTracker,
    fixed_gradient_accumulation,
    create_optimized_trainer_state,
    format_training_time,
    validate_config,
    save_config_to_file
)

logger = logging.getLogger(__name__)

# ===== BACKWARD COMPATIBILITY EXPORTS =====
# All functions are now imported from modular components

# Export everything for backward compatibility
__all__ = [
    # Setup utilities (now modular)
    "setup_professional_logging",
    "setup_directories",
    "setup_optimized_environment", 
    "setup_training_environment",
    
    # Optimization utilities (now modular)
    "optimize_jax_for_device",
    "cached_cpc_forward",
    "cached_spike_bridge",
    "cached_snn_forward", 
    "precompile_training_functions",
    
    # Monitoring utilities (now modular)
    "monitor_memory_usage", 
    "compute_gradient_norm",
    "check_for_nans",
    
    # Training utilities (now modular)
    "ProgressTracker",
    "fixed_gradient_accumulation",
    "create_optimized_trainer_state", 
    "format_training_time",
    "validate_config",
    "save_config_to_file"
]

# ===== DEPRECATION NOTICE =====
def _show_migration_notice():
    """Show migration notice for direct imports."""
    warnings.warn(
        "Direct imports from training_utils.py are deprecated. "
        "Use modular imports: from training.utils import setup_professional_logging, ProgressTracker",
        DeprecationWarning,
        stacklevel=3
    )

# Show notice when module is imported directly
logger.info("📦 Using modular training utility components (training_utils.py → utils/)")
_show_migration_notice()
