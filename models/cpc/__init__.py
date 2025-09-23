"""
CPC Module: Contrastive Predictive Coding Components

Modular implementation of CPC (Contrastive Predictive Coding) components
split from large files for better maintainability.

Components:
- core: Main CPCEncoder implementations
- transformer: Transformer-based CPC encoders  
- config: Configuration classes
- trainer: CPC training utilities
- factory: Factory functions for creating CPC components
- losses: InfoNCE and contrastive loss implementations
- miners: Hard negative mining and temperature control
- metrics: Contrastive learning evaluation metrics
"""

from .core import CPCEncoder, RealCPCEncoder, EnhancedCPCEncoder
from .transformer import TemporalTransformerCPC, TemporalTransformerConfig
from .config import RealCPCConfig, ExperimentConfig
from .trainer import CPCTrainer
from .factory import (
    create_cpc_encoder,
    create_enhanced_cpc_encoder,
    create_real_cpc_encoder,
    create_real_cpc_trainer,
    create_standard_cpc_encoder,
    create_experiment_config
)

from .blocks import ConvBlock, GRUContext, ProjectionHead, FeatureEncoder
from .losses import (
    enhanced_info_nce_loss, 
    info_nce_loss,
    temporal_info_nce_loss,
    advanced_info_nce_loss_with_momentum,
    momentum_enhanced_info_nce_loss,
    gw_twins_inspired_loss  # ✅ NEW: GW Twins inspired loss
)
from .miners import MomentumHardNegativeMiner, AdaptiveTemperatureController  
from .metrics import (
    contrastive_accuracy, 
    cosine_similarity_matrix,
    compute_contrastive_metrics,
    evaluate_representation_quality
)

__all__ = [
    # Core encoders
    "CPCEncoder",
    "RealCPCEncoder", 
    "EnhancedCPCEncoder",
    
    # Building blocks (NEW)
    "ConvBlock",
    "GRUContext",
    "ProjectionHead",
    "FeatureEncoder",
    
    # Transformer components
    "TemporalTransformerCPC",
    "TemporalTransformerConfig",
    
    # Configuration
    "RealCPCConfig",
    "ExperimentConfig",
    
    # Training
    "CPCTrainer",
    
    # Factory functions
    "create_cpc_encoder",
    "create_enhanced_cpc_encoder",
    "create_real_cpc_encoder",
    "create_real_cpc_trainer",
    "create_standard_cpc_encoder", 
    "create_experiment_config",
    
    # Loss functions
    "enhanced_info_nce_loss",
    "info_nce_loss",
    "temporal_info_nce_loss",
    "advanced_info_nce_loss_with_momentum",
    "momentum_enhanced_info_nce_loss",
    "gw_twins_inspired_loss",  # ✅ NEW
    
    # Mining and control
    "MomentumHardNegativeMiner",
    "AdaptiveTemperatureController",
    
    # Metrics
    "contrastive_accuracy",
    "cosine_similarity_matrix",
    "compute_contrastive_metrics",
    "evaluate_representation_quality"
]
