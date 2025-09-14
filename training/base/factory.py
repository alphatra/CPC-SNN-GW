"""
Factory functions for base training components.

This module contains factory functions extracted from
base_trainer.py for better modularity.

Split from base_trainer.py for better maintainability.
"""

import logging
from typing import Optional, Dict, Any

from .config import TrainingConfig
from .trainer import CPCSNNTrainer

logger = logging.getLogger(__name__)


def create_cpc_snn_trainer(config: Optional[TrainingConfig] = None) -> CPCSNNTrainer:
    """
    Factory function for creating standard CPC+SNN trainer.
    
    Args:
        config: Training configuration (uses defaults if None)
        
    Returns:
        Configured CPCSNNTrainer instance
    """
    if config is None:
        config = TrainingConfig()
    
    # Validate configuration
    if not config.validate():
        raise ValueError("Invalid training configuration provided")
    
    logger.info(f"Creating CPCSNNTrainer with config: batch_size={config.batch_size}, "
               f"learning_rate={config.learning_rate}, num_epochs={config.num_epochs}")
    
    return CPCSNNTrainer(config)


def create_training_config(**kwargs) -> TrainingConfig:
    """
    Factory function for creating training configuration with overrides.
    
    Args:
        **kwargs: Configuration parameter overrides
        
    Returns:
        TrainingConfig instance with applied overrides
    """
    config = TrainingConfig()
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config parameter: {key}")
    
    # Validate final configuration
    if not config.validate():
        raise ValueError("Invalid training configuration after applying overrides")
    
    logger.info(f"Created training config with {len(kwargs)} overrides")
    
    return config


def create_default_cpc_snn_trainer() -> CPCSNNTrainer:
    """
    Create CPC+SNN trainer with default configuration optimized for most use cases.
    
    Returns:
        CPCSNNTrainer with sensible defaults
    """
    config = TrainingConfig(
        # Optimized defaults
        batch_size=1,  # Memory-safe
        learning_rate=5e-5,  # Conservative LR
        num_epochs=50,  # Reasonable training length
        optimizer="sgd",  # Memory-efficient
        
        # Enable key features
        use_focal_loss=True,
        use_cpc_aux_loss=True,
        use_real_ligo_data=True,
        
        # Memory optimization
        mixed_precision=True,
        grad_accum_steps=4,  # Simulate larger batch
        
        # Monitoring
        use_wandb=False,  # Disable by default
        enable_profiling=False  # Disable for speed
    )
    
    return create_cpc_snn_trainer(config)


def get_trainer_recommendations(use_case: str = "general") -> Dict[str, Any]:
    """
    Get recommended configuration for different use cases.
    
    Args:
        use_case: Use case type ("general", "memory_limited", "high_performance", "debugging")
        
    Returns:
        Dictionary with recommended configuration parameters
    """
    recommendations = {
        "general": {
            "batch_size": 1,
            "learning_rate": 5e-5,
            "num_epochs": 50,
            "optimizer": "sgd",
            "use_focal_loss": True,
            "use_cpc_aux_loss": True,
            "mixed_precision": True
        },
        
        "memory_limited": {
            "batch_size": 1,
            "learning_rate": 1e-5,
            "num_epochs": 30,
            "optimizer": "sgd",
            "grad_accum_steps": 8,  # Simulate larger batch
            "mixed_precision": True,
            "enable_profiling": False,
            "sequence_length": 1024  # Shorter sequences
        },
        
        "high_performance": {
            "batch_size": 4,
            "learning_rate": 1e-4,
            "num_epochs": 100,
            "optimizer": "adamw",
            "use_focal_loss": True,
            "use_cpc_aux_loss": True,
            "mixed_precision": True,
            "enable_profiling": True
        },
        
        "debugging": {
            "batch_size": 1,
            "learning_rate": 1e-4,
            "num_epochs": 5,
            "log_every": 1,
            "eval_every": 5,
            "enable_profiling": True,
            "use_wandb": False,
            "use_tensorboard": False
        }
    }
    
    if use_case not in recommendations:
        logger.warning(f"Unknown use case: {use_case}. Using 'general'.")
        use_case = "general"
    
    return recommendations[use_case]


# Export factory functions
__all__ = [
    "create_cpc_snn_trainer",
    "create_training_config",
    "create_default_cpc_snn_trainer",
    "get_trainer_recommendations"
]
