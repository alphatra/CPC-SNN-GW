"""
Factory functions for creating CPC components.

This module contains factory functions extracted from
cpc_encoder.py for better modularity.

Split from cpc_encoder.py for better maintainability.
"""

import logging
from typing import Optional

from .config import RealCPCConfig, ExperimentConfig, CPCConfig
from .core import RealCPCEncoder, CPCEncoder, EnhancedCPCEncoder
from .trainer import CPCTrainer

logger = logging.getLogger(__name__)


def create_real_cpc_encoder(config: Optional[RealCPCConfig] = None) -> RealCPCEncoder:
    """Create production-ready CPC encoder with real data capabilities."""
    if config is None:
        config = RealCPCConfig()
    
    logger.info(f"Creating RealCPCEncoder with latent_dim={config.latent_dim}")
    return RealCPCEncoder(config=config)


def create_real_cpc_trainer(config: Optional[RealCPCConfig] = None) -> CPCTrainer:
    """Create CPC trainer with real data configuration."""
    if config is None:
        config = RealCPCConfig()
    
    logger.info(f"Creating CPCTrainer with config: {config}")
    return CPCTrainer(config=config)


def create_enhanced_cpc_encoder(config: Optional[ExperimentConfig] = None) -> EnhancedCPCEncoder:
    """Create enhanced CPC encoder with transformer and advanced features."""
    if config is None:
        config = ExperimentConfig()
    
    logger.info(f"Creating EnhancedCPCEncoder with transformer={config.use_transformer}")
    return EnhancedCPCEncoder(config=config)


def create_standard_cpc_encoder(latent_dim: int = 256,
                               context_length: int = 32,
                               prediction_steps: int = 8) -> CPCEncoder:
    """Create standard CPC encoder with basic configuration."""
    logger.info(f"Creating CPCEncoder with latent_dim={latent_dim}")
    return CPCEncoder(
        latent_dim=latent_dim,
        context_length=context_length,
        prediction_steps=prediction_steps
    )


def create_cpc_encoder(encoder_type: str = "standard", **kwargs) -> CPCEncoder:
    """
    Universal factory for creating any CPC encoder type.
    
    Args:
        encoder_type: Type of encoder ("standard", "real", "enhanced")
        **kwargs: Configuration parameters
        
    Returns:
        Configured CPC encoder
    """
    if encoder_type == "standard":
        return create_standard_cpc_encoder(**kwargs)
    elif encoder_type == "real":
        config = RealCPCConfig(**kwargs) if kwargs else None
        return create_real_cpc_encoder(config)
    elif encoder_type == "enhanced":
        config = ExperimentConfig(**kwargs) if kwargs else None
        return create_enhanced_cpc_encoder(config)
    else:
        logger.warning(f"Unknown encoder type: {encoder_type}. Using standard.")
        return create_standard_cpc_encoder(**kwargs)


def create_experiment_config(**kwargs) -> RealCPCConfig:
    """Create experiment configuration with overrides."""
    config = RealCPCConfig()
    
    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config parameter: {key}")
    
    # Validate the configuration
    if not config.validate():
        raise ValueError("Invalid experiment configuration")
    
    return config


def get_cpc_encoder_info() -> dict:
    """Get information about available CPC encoder types."""
    return {
        "standard": {
            "class": "CPCEncoder",
            "description": "Basic CPC encoder with conv+GRU architecture",
            "features": ["Temporal convolution", "GRU context", "InfoNCE loss"]
        },
        "real": {
            "class": "RealCPCEncoder", 
            "description": "Production-ready CPC with advanced features",
            "features": ["Multi-layer encoder", "Attention support", "Advanced normalization"]
        },
        "enhanced": {
            "class": "EnhancedCPCEncoder",
            "description": "Advanced CPC with transformer and multi-scale processing",
            "features": ["Multi-scale conv", "Transformer attention", "Feature fusion"]
        }
    }


# Export factory functions
__all__ = [
    "create_real_cpc_encoder",
    "create_real_cpc_trainer", 
    "create_enhanced_cpc_encoder",
    "create_standard_cpc_encoder",
    "create_cpc_encoder",
    "create_experiment_config",
    "get_cpc_encoder_info"
]

