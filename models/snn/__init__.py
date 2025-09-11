"""
SNN Module: Spiking Neural Network Components

Modular implementation of SNN (Spiking Neural Network) components
split from large files for better maintainability.

Components:
- core: Main SNN classifier implementations
- layers: LIF and enhanced spiking layers
- config: Configuration classes for SNN
- trainer: SNN training utilities  
- factory: Factory functions for creating SNN components
"""

from .core import SNNClassifier, EnhancedSNNClassifier
from .layers import LIFLayer, VectorizedLIFLayer, EnhancedLIFWithMemory
from .config import SNNConfig
from .trainer import SNNTrainer
from .factory import (
    create_snn_classifier,
    create_enhanced_snn_classifier,
    create_snn_config
)

__all__ = [
    # Core classifiers
    "SNNClassifier",
    "EnhancedSNNClassifier",
    
    # Layer implementations
    "LIFLayer",
    "VectorizedLIFLayer", 
    "EnhancedLIFWithMemory",
    
    # Configuration
    "SNNConfig",
    
    # Training
    "SNNTrainer",
    
    # Factory functions
    "create_snn_classifier",
    "create_enhanced_snn_classifier",
    "create_snn_config"
]

