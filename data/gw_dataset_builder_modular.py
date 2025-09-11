"""
GW Dataset Builder (MODULAR)

This file delegates to modular dataset builder components for better maintainability.
The actual implementation has been split into:
- builders/core.py: GWDatasetBuilder
- builders/factory.py: Factory functions
- builders/testing.py: Test utilities

This file maintains backward compatibility through delegation.
"""

import logging
import warnings

# Import from new modular components
from .builders.core import GWDatasetBuilder
from .builders.factory import (
    create_mixed_gw_dataset,
    create_evaluation_dataset,
    create_training_dataset
)
from .builders.testing import test_dataset_builder

# Re-export dependencies for compatibility
from .gw_signal_params import GeneratorSettings
from .gw_synthetic_generator import ContinuousGWGenerator

logger = logging.getLogger(__name__)

# ===== BACKWARD COMPATIBILITY EXPORTS =====
__all__ = [
    # Main builder (now modular)
    "GWDatasetBuilder",
    
    # Factory functions (now modular)
    "create_mixed_gw_dataset",
    "create_evaluation_dataset", 
    "create_training_dataset",
    
    # Testing (now modular)
    "test_dataset_builder"
]

# ===== DEPRECATION NOTICE =====
warnings.warn(
    "Direct imports from gw_dataset_builder.py are deprecated. "
    "Use modular imports: from data.builders import GWDatasetBuilder, create_mixed_gw_dataset",
    DeprecationWarning,
    stacklevel=2
)

logger.info("📦 Using modular dataset builder components (gw_dataset_builder.py → builders/)")

