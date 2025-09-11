"""
Factory functions for creating SNN components.

This module contains factory functions extracted from
snn_classifier.py for better modularity.

Split from snn_classifier.py for better maintainability.
"""

import logging
from typing import Optional

from .config import SNNConfig, EnhancedSNNConfig, LIFConfig
from .core import SNNClassifier, EnhancedSNNClassifier
from .layers import LIFLayer, VectorizedLIFLayer, EnhancedLIFWithMemory
from .trainer import SNNTrainer

logger = logging.getLogger(__name__)


def create_snn_classifier(hidden_size: int = 128, 
                         num_classes: int = 2,
                         num_layers: int = 3) -> SNNClassifier:
    """
    Create standard SNN classifier with basic configuration.
    
    Args:
        hidden_size: Size of hidden layers
        num_classes: Number of output classes
        num_layers: Number of SNN layers
        
    Returns:
        Configured SNNClassifier
    """
    logger.info(f"Creating SNNClassifier: {num_layers} layers, {hidden_size} units, {num_classes} classes")
    
    return SNNClassifier(
        hidden_size=hidden_size,
        num_classes=num_classes,
        num_layers=num_layers
    )


def create_enhanced_snn_classifier(config: Optional[EnhancedSNNConfig] = None) -> EnhancedSNNClassifier:
    """
    Create enhanced SNN classifier with advanced features.
    
    Args:
        config: Enhanced SNN configuration
        
    Returns:
        Configured EnhancedSNNClassifier
    """
    if config is None:
        config = EnhancedSNNConfig()
    
    logger.info(f"Creating EnhancedSNNClassifier with {len(config.hidden_sizes)} layers")
    logger.info(f"Features: attention={config.use_attention}, memory={config.use_long_term_memory}")
    
    return EnhancedSNNClassifier(config=config)


def create_snn_config(
    hidden_sizes: tuple = (256, 128, 64),
    num_classes: int = 2,
    tau_mem: float = 20e-3,
    tau_syn: float = 5e-3,
    threshold: float = 1.0,
    use_layer_norm: bool = True,
    surrogate_beta: float = 10.0,
    **kwargs
) -> SNNConfig:
    """
    Create SNN configuration with custom parameters.
    
    Args:
        hidden_sizes: Tuple of hidden layer sizes
        num_classes: Number of output classes
        tau_mem: Membrane time constant
        tau_syn: Synaptic time constant
        threshold: Spike threshold
        use_layer_norm: Whether to use layer normalization
        surrogate_beta: Surrogate gradient beta parameter
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured SNNConfig
    """
    config = SNNConfig(
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        num_layers=len(hidden_sizes),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        threshold=threshold,
        use_layer_norm=use_layer_norm,
        surrogate_beta=surrogate_beta,
        **kwargs
    )
    
    # Validate configuration
    if not config.validate():
        raise ValueError("Invalid SNN configuration")
    
    logger.info(f"Created SNNConfig: {config.num_layers} layers, classes={config.num_classes}")
    
    return config


def create_enhanced_snn_config(
    hidden_sizes: tuple = (256, 128, 64),
    num_classes: int = 2,
    use_attention: bool = False,
    use_long_term_memory: bool = True,
    use_adaptive_threshold: bool = False,
    **kwargs
) -> EnhancedSNNConfig:
    """
    Create enhanced SNN configuration with advanced features.
    
    Args:
        hidden_sizes: Tuple of hidden layer sizes
        num_classes: Number of output classes  
        use_attention: Whether to use attention mechanisms
        use_long_term_memory: Whether to use long-term memory
        use_adaptive_threshold: Whether to use adaptive thresholds
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured EnhancedSNNConfig
    """
    config = EnhancedSNNConfig(
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        num_layers=len(hidden_sizes),
        use_attention=use_attention,
        use_long_term_memory=use_long_term_memory,
        use_adaptive_threshold=use_adaptive_threshold,
        **kwargs
    )
    
    # Validate configuration
    if not config.validate():
        raise ValueError("Invalid enhanced SNN configuration")
    
    logger.info(f"Created EnhancedSNNConfig: {config.num_layers} layers, "
               f"attention={use_attention}, memory={use_long_term_memory}")
    
    return config


def create_lif_layer(features: int,
                    tau_mem: float = 20e-3,
                    tau_syn: float = 5e-3,
                    threshold: float = 1.0,
                    layer_type: str = "standard") -> LIFLayer:
    """
    Create LIF layer with specified configuration.
    
    Args:
        features: Number of neurons
        tau_mem: Membrane time constant
        tau_syn: Synaptic time constant  
        threshold: Spike threshold
        layer_type: Type of LIF layer ("standard", "vectorized", "enhanced")
        
    Returns:
        Configured LIF layer
    """
    if layer_type == "vectorized":
        return VectorizedLIFLayer(
            features=features,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            threshold=threshold
        )
    elif layer_type == "enhanced":
        return EnhancedLIFWithMemory(
            features=features,
            tau_mem=tau_mem,
            tau_syn=tau_syn,  
            threshold=threshold
        )
    else:  # standard
        return LIFLayer(
            features=features,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            threshold=threshold
        )


def create_snn_trainer(config: Optional[SNNConfig] = None) -> SNNTrainer:
    """
    Create SNN trainer with configuration.
    
    Args:
        config: SNN configuration
        
    Returns:
        Configured SNNTrainer
    """
    if config is None:
        config = SNNConfig()
    
    logger.info(f"Creating SNNTrainer with {config.num_layers} layers")
    return SNNTrainer(config=config)


def get_snn_info() -> dict:
    """Get information about available SNN components."""
    return {
        "classifiers": {
            "standard": {
                "class": "SNNClassifier",
                "description": "Basic multi-layer SNN with LIF neurons",
                "features": ["3-layer architecture", "LayerNorm", "Adaptive dropout"]
            },
            "enhanced": {
                "class": "EnhancedSNNClassifier", 
                "description": "Advanced SNN with memory and attention",
                "features": ["Memory mechanisms", "Attention", "Adaptive thresholds"]
            }
        },
        "layers": {
            "standard": "Basic LIF layer with sequential processing",
            "vectorized": "Vectorized LIF for performance optimization",
            "enhanced": "Advanced LIF with memory and adaptation"
        }
    }


# Export factory functions
__all__ = [
    "create_snn_classifier",
    "create_enhanced_snn_classifier", 
    "create_snn_config",
    "create_enhanced_snn_config",
    "create_lif_layer",
    "create_snn_trainer",
    "get_snn_info"
]

