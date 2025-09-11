"""
SNN Classifier (MODULAR)

This file delegates to modular SNN classifier components for better maintainability.
The actual implementation has been split into:
- snn/core.py: Main classifier implementations
- snn/layers.py: LIF layer implementations
- snn/config.py: Configuration classes
- snn/trainer.py: Training utilities
- snn/factory.py: Factory functions

This file maintains backward compatibility through delegation.
"""

import logging
import warnings

# Import from new modular components
from .snn import (
    SNNClassifier,
    EnhancedSNNClassifier,
    LIFLayer,
    VectorizedLIFLayer,
    EnhancedLIFWithMemory,
    SNNConfig,
    EnhancedSNNConfig,
    SNNTrainer,
    create_snn_classifier,
    create_enhanced_snn_classifier,
    create_snn_config,
    create_enhanced_snn_config,
    create_lif_layer,
    create_snn_trainer
)

logger = logging.getLogger(__name__)

# ===== BACKWARD COMPATIBILITY EXPORTS =====
# All classes and functions are now imported from modular components

# Export everything for backward compatibility
__all__ = [
    # Core classifiers (now modular)
    "SNNClassifier",
    "EnhancedSNNClassifier",
    
    # Layer implementations (now modular)
    "LIFLayer",
    "VectorizedLIFLayer",
    "EnhancedLIFWithMemory",
    
    # Configuration (now modular)
    "SNNConfig",
    "EnhancedSNNConfig",
    
    # Training (now modular)
    "SNNTrainer",
    
    # Factory functions (now modular)
    "create_snn_classifier",
    "create_enhanced_snn_classifier",
    "create_snn_config",
    "create_enhanced_snn_config",
    "create_lif_layer",
    "create_snn_trainer"
]

# ===== DEPRECATION NOTICE =====
def _show_migration_notice():
    """Show migration notice for direct imports."""
    warnings.warn(
        "Direct imports from snn_classifier.py are deprecated. "
        "Use modular imports: from models.snn import SNNClassifier, EnhancedSNNClassifier, LIFLayer",
        DeprecationWarning,
        stacklevel=3
    )

# Show notice when module is imported directly
logger.info("📦 Using modular SNN classifier components (snn_classifier.py → snn/)")
_show_migration_notice()
