"""
Complete Enhanced Training (MODULAR)

This file delegates to modular enhanced training components for better maintainability.
The actual implementation has been split into:
- enhanced/config.py: CompleteEnhancedConfig + TrainStateWithBatchStats
- enhanced/model.py: CompleteEnhancedModel
- enhanced/trainer.py: CompleteEnhancedTrainer
- enhanced/factory.py: Factory functions

This file maintains backward compatibility through delegation.

🚀 COMPLETE ENHANCED TRAINING - ALL 5 REVOLUTIONARY IMPROVEMENTS INTEGRATED

World's first complete neuromorphic gravitational wave detection system with:
✅ 1. Adaptive Multi-Scale Surrogate Gradients (better than ETSformer ESA)
✅ 2. Temporal Transformer with Multi-Scale Convolution (GW-optimized)
✅ 3. Learnable Multi-Threshold Spike Encoding (biologically realistic)
✅ 4. Enhanced LIF with Memory and Refractory Period (neuromorphic advantages)
✅ 5. Momentum-based InfoNCE with Hard Negative Mining (superior contrastive learning)
"""

import logging
import warnings

# Import from new modular components
from .enhanced import (
    CompleteEnhancedTrainer,
    CompleteEnhancedConfig,
    CompleteEnhancedModel,
    TrainStateWithBatchStats,
    create_complete_enhanced_trainer,
    run_complete_enhanced_experiment
)

logger = logging.getLogger(__name__)

# ===== BACKWARD COMPATIBILITY EXPORTS =====
# All classes and functions are now imported from modular components

# Export everything for backward compatibility
__all__ = [
    # Configuration (now modular)
    "CompleteEnhancedConfig",
    "TrainStateWithBatchStats",
    
    # Model (now modular)
    "CompleteEnhancedModel",
    
    # Trainer (now modular)
    "CompleteEnhancedTrainer",
    
    # Factory functions (now modular)
    "create_complete_enhanced_trainer",
    "run_complete_enhanced_experiment"
]

# ===== DEPRECATION NOTICE =====
def _show_migration_notice():
    """Show migration notice for direct imports."""
    warnings.warn(
        "Direct imports from complete_enhanced_training.py are deprecated. "
        "Use modular imports: from training.enhanced import CompleteEnhancedTrainer, CompleteEnhancedConfig",
        DeprecationWarning,
        stacklevel=3
    )

# Show notice when module is imported directly
logger.info("📦 Using modular enhanced training components (complete_enhanced_training.py → enhanced/)")
_show_migration_notice()
