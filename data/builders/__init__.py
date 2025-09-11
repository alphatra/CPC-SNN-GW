"""
Builders Module: Dataset Building Components

Modular implementation of dataset builders split from 
gw_dataset_builder.py for better maintainability.

Components:
- core: GWDatasetBuilder main class
- factory: Dataset creation factory functions
- testing: Test utilities for dataset builders
"""

from .core import GWDatasetBuilder
from .factory import create_mixed_gw_dataset, create_evaluation_dataset
from .testing import test_dataset_builder

__all__ = [
    # Main builder
    "GWDatasetBuilder",
    
    # Factory functions
    "create_mixed_gw_dataset",
    "create_evaluation_dataset",
    
    # Testing utilities
    "test_dataset_builder"
]

