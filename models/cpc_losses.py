"""
CPC Losses (MODULAR)

This file delegates to modular CPC loss components for better maintainability.
The actual implementation has been split into:
- cpc/losses.py: InfoNCE implementations
- cpc/miners.py: Hard negative mining
- cpc/metrics.py: Contrastive evaluation metrics

This file maintains backward compatibility through delegation.
"""

import logging
import warnings

# Import from new modular components
from .cpc import (
    enhanced_info_nce_loss,
    info_nce_loss,
    temporal_info_nce_loss,
    advanced_info_nce_loss_with_momentum,
    momentum_enhanced_info_nce_loss,
    MomentumHardNegativeMiner,
    AdaptiveTemperatureController,
    contrastive_accuracy,
    cosine_similarity_matrix,
    compute_contrastive_metrics,
    evaluate_representation_quality
)

logger = logging.getLogger(__name__)

# ===== BACKWARD COMPATIBILITY EXPORTS =====
# All functions and classes are now imported from modular components

# Export everything for backward compatibility
__all__ = [
    # Loss functions (now modular)
    "enhanced_info_nce_loss",
    "info_nce_loss",
    "temporal_info_nce_loss",
    "advanced_info_nce_loss_with_momentum",
    "momentum_enhanced_info_nce_loss",
    
    # Mining and control (now modular)
    "MomentumHardNegativeMiner",
    "AdaptiveTemperatureController",
    
    # Metrics (now modular)
    "contrastive_accuracy",
    "cosine_similarity_matrix",
    "compute_contrastive_metrics",
    "evaluate_representation_quality"
]

# ===== DEPRECATION NOTICE =====
def _show_migration_notice():
    """Show migration notice for direct imports."""
    warnings.warn(
        "Direct imports from cpc_losses.py are deprecated. "
        "Use modular imports: from models.cpc import enhanced_info_nce_loss, MomentumHardNegativeMiner",
        DeprecationWarning,
        stacklevel=3
    )

# Show notice when module is imported directly
logger.info("📦 Using modular CPC loss components (cpc_losses.py → cpc/)")
_show_migration_notice()
