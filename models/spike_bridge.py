"""
Spike Bridge (MODULAR)

This file delegates to modular spike bridge components for better maintainability.
The actual implementation has been split into:
- bridge/core.py: ValidatedSpikeBridge main class
- bridge/encoders.py: Encoding strategies
- bridge/gradients.py: Gradient flow monitoring
- bridge/testing.py: Test utilities

This file maintains backward compatibility through delegation.

Enhanced Spike Bridge with Gradient Flow Validation
"""

import logging
import warnings

# Import from new modular components
from .bridge import (
    ValidatedSpikeBridge,
    TemporalContrastEncoder,
    LearnableMultiThresholdEncoder,
    PhasePreservingEncoder,
    GradientFlowMonitor,
    EnhancedSurrogateGradients,
    spike_function_with_surrogate,
    spike_function_fwd,
    spike_function_bwd,
    test_gradient_flow,
    create_validated_spike_bridge
)

logger = logging.getLogger(__name__)

# ===== BACKWARD COMPATIBILITY EXPORTS =====
# All classes and functions are now imported from modular components

# Export everything for backward compatibility
__all__ = [
    # Main bridge (now modular)
    "ValidatedSpikeBridge",
    "create_validated_spike_bridge",
    
    # Encoders (now modular)
    "TemporalContrastEncoder",
    "LearnableMultiThresholdEncoder",
    "PhasePreservingEncoder",
    
    # Gradient components (now modular)
    "GradientFlowMonitor",
    "EnhancedSurrogateGradients",
    "spike_function_with_surrogate",
    "spike_function_fwd",
    "spike_function_bwd",
    
    # Testing (now modular)
    "test_gradient_flow"
]

# ===== DEPRECATION NOTICE =====
def _show_migration_notice():
    """Show migration notice for direct imports."""
    warnings.warn(
        "Direct imports from spike_bridge.py are deprecated. "
        "Use modular imports: from models.bridge import ValidatedSpikeBridge, TemporalContrastEncoder",
        DeprecationWarning,
        stacklevel=3
    )

# Show notice when module is imported directly
logger.info("📦 Using modular spike bridge components (spike_bridge.py → bridge/)")
_show_migration_notice()
