"""
CPC Encoder (MODULAR)

This file delegates to modular CPC encoder components for better maintainability.
The actual implementation has been split into:
- cpc/core.py: Main encoder implementations
- cpc/transformer.py: Transformer-based encoders
- cpc/config.py: Configuration classes
- cpc/trainer.py: Training utilities
- cpc/factory.py: Factory functions

This file maintains backward compatibility through delegation.
"""

import logging
import warnings

# Import from new modular components
from .cpc import (
    CPCEncoder,
    RealCPCEncoder,
    EnhancedCPCEncoder,
    TemporalTransformerCPC,
    TemporalTransformerConfig,
    RealCPCConfig,
    ExperimentConfig,
    CPCTrainer,
    create_cpc_encoder,
    create_enhanced_cpc_encoder,
    create_real_cpc_encoder,
    create_real_cpc_trainer,
    create_standard_cpc_encoder,
    create_experiment_config
)

logger = logging.getLogger(__name__)

# ===== BACKWARD COMPATIBILITY EXPORTS =====
# All classes and functions are now imported from modular components

# Export everything for backward compatibility
__all__ = [
    # Core encoders (now modular)
    "CPCEncoder",
    "RealCPCEncoder",
    "EnhancedCPCEncoder",
    
    # Transformer components (now modular)
    "TemporalTransformerCPC",
    "TemporalTransformerConfig",
    
    # Configuration (now modular)
    "RealCPCConfig",
    "ExperimentConfig",
    
    # Training (now modular)
    "CPCTrainer",
    
    # Factory functions (now modular)
    "create_cpc_encoder",
    "create_enhanced_cpc_encoder",
    "create_real_cpc_encoder", 
    "create_real_cpc_trainer",
    "create_standard_cpc_encoder",
    "create_experiment_config"
]

# ===== DEPRECATION NOTICE =====
def _show_migration_notice():
    """Show migration notice for direct imports."""
    warnings.warn(
        "Direct imports from cpc_encoder.py are deprecated. "
        "Use modular imports: from models.cpc import CPCEncoder, RealCPCEncoder, EnhancedCPCEncoder",
        DeprecationWarning,
        stacklevel=3
    )

# Show notice when module is imported directly
logger.info("📦 Using modular CPC encoder components (cpc_encoder.py → cpc/)")
_show_migration_notice()
