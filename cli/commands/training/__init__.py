"""
Training Commands Module

Modular implementation of training command components
split from CLI for better maintainability.

Components:
- initializer: Training environment setup
- data_loader: Data loading strategies  
- standard: Standard CPC+SNN training
- enhanced: Enhanced training with improvements
"""

from .initializer import (
    setup_training_environment,
    validate_training_setup, 
    get_recommended_training_config
)
from .data_loader import (
    load_training_data,
    _load_mlgwsc_data,
    _load_synthetic_data,
    _load_real_ligo_data
)
from .standard import run_standard_training
from .enhanced import run_enhanced_training, run_complete_enhanced_training

__all__ = [
    # Environment setup
    "setup_training_environment",
    "validate_training_setup",
    "get_recommended_training_config",
    
    # Data loading
    "load_training_data",
    "_load_mlgwsc_data", 
    "_load_synthetic_data",
    "_load_real_ligo_data",
    
    # Training implementations
    "run_standard_training",
    "run_enhanced_training",
    "run_complete_enhanced_training"
]
