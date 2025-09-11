"""
Enhanced Training Module: Complete Enhanced Training Components

Modular implementation of the complete enhanced training system
split from complete_enhanced_training.py (1052 LOC) for better maintainability.

This module contains the world's first complete neuromorphic gravitational 
wave detection training system with all 5 revolutionary improvements.

Components:
- config: Configuration classes and data structures
- model: Complete enhanced model implementation
- trainer: Main enhanced trainer class
- factory: Factory functions for creating enhanced components
- optimizations: Advanced optimization utilities
"""

from .config import CompleteEnhancedConfig, TrainStateWithBatchStats
from .model import CompleteEnhancedModel
from .trainer import CompleteEnhancedTrainer
from .factory import create_complete_enhanced_trainer, run_complete_enhanced_experiment

__all__ = [
    # Configuration
    "CompleteEnhancedConfig",
    "TrainStateWithBatchStats",
    
    # Model
    "CompleteEnhancedModel",
    
    # Trainer
    "CompleteEnhancedTrainer",
    
    # Factory functions
    "create_complete_enhanced_trainer",
    "run_complete_enhanced_experiment"
]

