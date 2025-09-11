"""
Advanced Training (MODULAR)

This file delegates to modular advanced training components for better maintainability.
The actual implementation has been split into:
- advanced/attention.py: AttentionCPCEncoder 
- advanced/snn_deep.py: DeepSNN + LIFLayer
- advanced/trainer.py: RealAdvancedGWTrainer + factory

This file maintains backward compatibility through delegation.

Advanced Training with Real Gradient Updates
Addresses Executive Summary Priority 5: Replace Mock Training
"""

import logging
import warnings

# Import from new modular components
from .advanced import (
    AttentionCPCEncoder,
    DeepSNN,
    LIFLayer,
    RealAdvancedGWTrainer,
    create_real_advanced_trainer
)

logger = logging.getLogger(__name__)

# ===== BACKWARD COMPATIBILITY EXPORTS =====
# All classes and functions are now imported from modular components

# Export everything for backward compatibility
__all__ = [
    # Attention components (now modular)
    "AttentionCPCEncoder",
    
    # Deep SNN components (now modular)
    "DeepSNN",
    "LIFLayer",
    
    # Advanced trainer (now modular)
    "RealAdvancedGWTrainer",
    "create_real_advanced_trainer"
]

# ===== DEPRECATION NOTICE =====
def _show_migration_notice():
    """Show migration notice for direct imports."""
    warnings.warn(
        "Direct imports from advanced_training.py are deprecated. "
        "Use modular imports: from training.advanced import RealAdvancedGWTrainer, AttentionCPCEncoder",
        DeprecationWarning,
        stacklevel=3
    )

# Show notice when module is imported directly
logger.info("📦 Using modular advanced training components (advanced_training.py → advanced/)")
_show_migration_notice()
