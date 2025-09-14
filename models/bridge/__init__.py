"""
Bridge Module: Spike Bridge Components for CPC-SNN Integration

Modular implementation of spike encoding and gradient flow validation
split from the large spike_bridge.py for better maintainability.

Components:
- core: ValidatedSpikeBridge main class
- encoders: Various spike encoding strategies
- gradients: Gradient flow monitoring and surrogate functions
- testing: Test utilities for gradient flow validation
"""

from .core import ValidatedSpikeBridge, create_validated_spike_bridge
from .encoders import (
    TemporalContrastEncoder,
    LearnableMultiThresholdEncoder, 
    PhasePreservingEncoder
)
from .gradients import (
    GradientFlowMonitor,
    EnhancedSurrogateGradients,
    spike_function_with_surrogate,
    spike_function_fwd,
    spike_function_bwd
)
from .losses import (
    sigmoid_surrogate,
    spike_rate_loss,
    create_surrogate_function
)
from .heads import SpikeProjectionHead, MultiChannelProjection, AdaptiveProjection

__all__ = [
    # Core bridge components
    "ValidatedSpikeBridge",
    "create_validated_spike_bridge",
    
    # Encoding strategies
    "TemporalContrastEncoder",
    "LearnableMultiThresholdEncoder", 
    "PhasePreservingEncoder",
    
    # Gradient flow components
    "GradientFlowMonitor",
    "EnhancedSurrogateGradients", 
    "spike_function_with_surrogate",
    "spike_function_fwd",
    "spike_function_bwd",
    
    # Loss functions (NEW)
    "sigmoid_surrogate",
    "spike_rate_loss",
    "create_surrogate_function",
    
    # Head components (NEW)
    "SpikeProjectionHead",
    "MultiChannelProjection",
    "AdaptiveProjection",
    
    # Testing utilities (kept internal; not exported)
]

