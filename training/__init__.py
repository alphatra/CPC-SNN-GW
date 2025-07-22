"""
Training Module: Neuromorphic Training Pipeline

Core training infrastructure for CPC+SNN gravitational wave detection:
- Base trainer with unified interface
- Specialized trainers (CPC pretraining, unified multi-stage, advanced, enhanced)
- Training utilities and metrics
- Production-ready training experiments
"""

import logging
from typing import Dict, Any, Optional

# Module version
__version__ = "1.0.0"

# Module logger
logger = logging.getLogger(__name__)

# Core trainer exports
from .base_trainer import (
    TrainerBase,
    TrainingConfig, 
    CPCSNNTrainer,
    create_cpc_snn_trainer
)

# Specialized trainers
from .unified_trainer import (
    UnifiedTrainer,
    UnifiedTrainingConfig,
    create_unified_trainer
)

from .advanced_training import (
    RealAdvancedGWTrainer as AdvancedGWTrainer,  # Use alias for compatibility
    create_real_advanced_trainer as create_advanced_trainer  # Use alias for compatibility
)

from .enhanced_gw_training import (
    EnhancedGWTrainer,
    EnhancedGWConfig,
    create_enhanced_trainer,
    run_enhanced_training_experiment
)

from .pretrain_cpc import (
    CPCPretrainer,
    CPCPretrainConfig,
    create_cpc_pretrainer,
    run_cpc_pretraining_experiment
)

# Training utilities
from .training_utils import (
    setup_professional_logging,
    setup_directories,
    optimize_jax_for_device,
    validate_config,
    save_config_to_file,
    compute_gradient_norm,
    check_for_nans,
    ProgressTracker,
    format_training_time
)

from .training_metrics import (
    TrainingMetrics,
    ExperimentTracker,
    EarlyStoppingMonitor,
    PerformanceProfiler,
    create_training_metrics
)

# All available trainers
AVAILABLE_TRAINERS = {
    'base': CPCSNNTrainer,
    'unified': UnifiedTrainer,
    'advanced': AdvancedGWTrainer,
    'enhanced': EnhancedGWTrainer,
    'cpc_pretrain': CPCPretrainer
}

# All available configs
AVAILABLE_CONFIGS = {
    'base': TrainingConfig,
    'unified': UnifiedTrainingConfig,
    'advanced': TrainingConfig,  # Use base config as fallback
    'enhanced': EnhancedGWConfig,
    'cpc_pretrain': CPCPretrainConfig
}


def create_trainer(trainer_type: str, config: Optional[Any] = None):
    """
    Factory function to create any trainer type.
    
    Args:
        trainer_type: Type of trainer ('base', 'unified', 'advanced', 'enhanced', 'cpc_pretrain')
        config: Optional configuration object
    
    Returns:
        Configured trainer instance
    """
    if trainer_type not in AVAILABLE_TRAINERS:
        raise ValueError(f"Unknown trainer type: {trainer_type}. Available: {list(AVAILABLE_TRAINERS.keys())}")
    
    trainer_class = AVAILABLE_TRAINERS[trainer_type]
    
    if config is None:
        config_class = AVAILABLE_CONFIGS[trainer_type]
        config = config_class()
    
    return trainer_class(config)


def run_training_experiment(experiment_type: str = 'base'):
    """
    Run a complete training experiment.
    
    Args:
        experiment_type: Type of experiment to run
    
    Returns:
        Experiment results
    """
    experiment_runners = {
        'enhanced': run_enhanced_training_experiment,
        'cpc_pretrain': run_cpc_pretraining_experiment
    }
    
    if experiment_type in experiment_runners:
        return experiment_runners[experiment_type]()
    else:
        logger.info(f"No predefined experiment for {experiment_type}. Use create_trainer() instead.")
        return None


def get_trainer_info() -> Dict[str, Any]:
    """Get information about available trainers and their capabilities."""
    return {
        'base': {
            'class': 'CPCSNNTrainer',
            'description': 'Basic CPC+SNN trainer with standard pipeline',
            'features': ['CPC encoder', 'Spike bridge', 'SNN classifier']
        },
        'unified': {
            'class': 'UnifiedTrainer', 
            'description': 'Multi-stage training (CPC -> SNN -> Joint)',
            'features': ['Multi-stage training', 'Progressive training', 'Stage-wise optimization']
        },
        'advanced': {
            'class': 'AdvancedGWTrainer',
            'description': 'Advanced techniques for 85%+ accuracy',
            'features': ['Attention mechanism', 'Focal loss', 'Mixup augmentation', 'Deep SNN']
        },
        'enhanced': {
            'class': 'EnhancedGWTrainer',
            'description': 'Production-ready with real data integration',
            'features': ['GWOSC data', 'Mixed datasets', 'Detailed metrics', 'Gradient accumulation']
        },
        'cpc_pretrain': {
            'class': 'CPCPretrainer',
            'description': 'Self-supervised CPC pretraining',
            'features': ['InfoNCE loss', 'Self-supervised learning', 'Representation learning']
        }
    }


# Module exports
__all__ = [
    # Core trainers
    'TrainerBase',
    'TrainingConfig',
    'CPCSNNTrainer',
    'create_cpc_snn_trainer',
    
    # Specialized trainers
    'UnifiedTrainer',
    'UnifiedTrainingConfig', 
    'create_unified_trainer',
    'AdvancedGWTrainer',
    'create_advanced_trainer',
    'EnhancedGWTrainer', 
    'EnhancedGWConfig',
    'create_enhanced_trainer',
    'CPCPretrainer',
    'CPCPretrainConfig',
    'create_cpc_pretrainer',
    
    # Utilities
    'setup_professional_logging',
    'setup_directories',
    'optimize_jax_for_device',
    'validate_config',
    'save_config_to_file',
    'compute_gradient_norm',
    'check_for_nans',
    'ProgressTracker',
    'format_training_time',
    
    # Metrics
    'TrainingMetrics',
    'ExperimentTracker',
    'EarlyStoppingMonitor',
    'PerformanceProfiler',
    'create_training_metrics',
    
    # Experiments
    'run_enhanced_training_experiment', 
    'run_cpc_pretraining_experiment',
    
    # Factory functions
    'create_trainer',
    'run_training_experiment',
    'get_trainer_info',
    
    # Constants
    'AVAILABLE_TRAINERS',
    'AVAILABLE_CONFIGS'
]

logger.info(f"Training module initialized (v{__version__}) with {len(AVAILABLE_TRAINERS)} trainer types")
