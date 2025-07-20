"""
Training Module: Model Training Infrastructure

Provides comprehensive training capabilities for CPC+SNN pipeline:
- TrainerBase: Unified training interface with W&B/TensorBoard integration
- Enhanced training with attention mechanisms
- Advanced training with progressive learning
- Hydra CLI configuration system

All trainers use unified TrainingConfig for consistent configuration.
"""

from .enhanced_gw_training import EnhancedGWTrainer, create_enhanced_gw_trainer
from .advanced_training import AdvancedGWTrainer, run_advanced_training_experiment
from .pretrain_cpc import CPCPretrainer, main as pretrain_cpc_main
from .base_trainer import (
    TrainerBase,
    TrainingConfig,
    TrainingMetrics,
    HydraTrainerMixin,
    CPCSNNTrainer,
    create_training_config,
    create_cpc_snn_trainer,
    create_cpc_snn_cli_app,
    create_hydra_cli_app
)

__all__ = [
    # Legacy trainers
    "EnhancedGWTrainer",
    "create_enhanced_gw_trainer",
    "AdvancedGWTrainer", 
    "run_advanced_training_experiment",
    "CPCPretrainer",
    "pretrain_cpc_main",
    
    # New unified training system
    "TrainerBase",
    "TrainingConfig",
    "TrainingMetrics",
    "HydraTrainerMixin",
    "CPCSNNTrainer",
    "create_training_config",
    "create_cpc_snn_trainer",
    "create_cpc_snn_cli_app",
    "create_hydra_cli_app"
]

# Lazy import system to avoid circular dependencies
def __getattr__(name):
    if name == "EnhancedGWTrainer":
        from .enhanced_gw_training import EnhancedGWTrainer
        return EnhancedGWTrainer
    elif name == "AdvancedGWTrainer":
        from .advanced_training import AdvancedGWTrainer
        return AdvancedGWTrainer
    elif name == "CPCPretrainer":
        from .pretrain_cpc import CPCPretrainer
        return CPCPretrainer
    elif name == "TrainerBase":
        from .base_trainer import TrainerBase
        return TrainerBase
    elif name == "TrainingConfig":
        from .base_trainer import TrainingConfig
        return TrainingConfig
    elif name == "TrainingMetrics":
        from .base_trainer import TrainingMetrics
        return TrainingMetrics
    elif name == "HydraTrainerMixin":
        from .base_trainer import HydraTrainerMixin
        return HydraTrainerMixin
    elif name == "CPCSNNTrainer":
        from .base_trainer import CPCSNNTrainer
        return CPCSNNTrainer
    elif name == "create_training_config":
        from .base_trainer import create_training_config
        return create_training_config
    elif name == "create_cpc_snn_trainer":
        from .base_trainer import create_cpc_snn_trainer
        return create_cpc_snn_trainer
    elif name == "create_cpc_snn_cli_app":
        from .base_trainer import create_cpc_snn_cli_app
        return create_cpc_snn_cli_app
    elif name == "create_hydra_cli_app":
        from .base_trainer import create_hydra_cli_app
        return create_hydra_cli_app
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
