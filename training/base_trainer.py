"""
Base Trainer (MODULAR)

This file delegates to modular base training components for better maintainability.
The actual implementation has been split into:
- base/config.py: TrainingConfig and configuration utilities
- base/trainer.py: TrainerBase + CPCSNNTrainer
- base/factory.py: Factory functions

This file maintains backward compatibility through delegation.

Base Trainer: Abstract Training Interface
"""

import logging
import warnings

# Import from new modular components
from .base import (
    TrainingConfig,
    TrainerBase,
    CPCSNNTrainer,
    create_cpc_snn_trainer
)

logger = logging.getLogger(__name__)

# ===== BACKWARD COMPATIBILITY EXPORTS =====
# All classes and functions are now imported from modular components

# Export everything for backward compatibility
__all__ = [
    # Configuration (now modular)
    "TrainingConfig",
    
    # Base trainers (now modular)
    "TrainerBase",
    "CPCSNNTrainer",
    
    # Factory functions (now modular)
    "create_cpc_snn_trainer"
]

# ===== DEPRECATION NOTICE =====
def _show_migration_notice():
    """Show migration notice for direct imports."""
    warnings.warn(
        "Direct imports from base_trainer.py are deprecated. "
        "Use modular imports: from training.base import TrainerBase, TrainingConfig, CPCSNNTrainer",
        DeprecationWarning,
        stacklevel=3
    )

# Show notice when module is imported directly
logger.info("📦 Using modular base training components (base_trainer.py → base/)")
_show_migration_notice()
