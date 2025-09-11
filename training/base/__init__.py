"""
Base Training Module: Core Training Components

Modular implementation of base training components
split from base_trainer.py for better maintainability.

Components:
- config: TrainingConfig and related configuration classes
- trainer: TrainerBase abstract class and CPCSNNTrainer implementation  
- factory: Factory functions for creating base trainers
"""

from .config import TrainingConfig
from .trainer import TrainerBase, CPCSNNTrainer
from .factory import create_cpc_snn_trainer

__all__ = [
    # Configuration
    "TrainingConfig",
    
    # Base trainers
    "TrainerBase",
    "CPCSNNTrainer",
    
    # Factory functions
    "create_cpc_snn_trainer"
]
