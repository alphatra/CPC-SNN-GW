"""
Models Module: Neural Network Architectures

Implements the 3-component neuromorphic pipeline:
1. CPC Encoder - Self-supervised representation learning
2. Spike Bridge - Continuous to spike conversion  
3. SNN Classifier - Neuromorphic binary classification
"""

import importlib
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, Any, List

# Module version
__version__ = "1.0.0"

# Module logger
logger = logging.getLogger(__name__)

# Core component imports - simplified lazy loading
_LAZY_IMPORTS = {
    # CPC Encoder - main components only
    "CPCEncoder": ("cpc_encoder", "CPCEncoder"),
    "EnhancedCPCEncoder": ("cpc_encoder", "EnhancedCPCEncoder"),
    "ExperimentConfig": ("cpc_encoder", "ExperimentConfig"),
    
    # CPC Components and Losses
    "RMSNorm": ("cpc_components", "RMSNorm"),
    "WeightNormDense": ("cpc_components", "WeightNormDense"),
    "enhanced_info_nce_loss": ("cpc_losses", "enhanced_info_nce_loss"),
    "info_nce_loss": ("cpc_losses", "info_nce_loss"),
    
    # SNN Classifier - main components only
    "SNNClassifier": ("snn_classifier", "SNNClassifier"),
    "EnhancedSNNClassifier": ("snn_classifier", "EnhancedSNNClassifier"),
    "VectorizedLIFLayer": ("snn_classifier", "VectorizedLIFLayer"),
    "SNNConfig": ("snn_classifier", "SNNConfig"),
    "SNNTrainer": ("snn_classifier", "SNNTrainer"),
    "LIFLayer": ("snn_classifier", "LIFLayer"),
    
    # SNN Utils
    "SurrogateGradientType": ("snn_utils", "SurrogateGradientType"),
    "BatchedSNNValidator": ("snn_utils", "BatchedSNNValidator"),
    "create_surrogate_gradient_fn": ("snn_utils", "create_surrogate_gradient_fn"),
    
    # Spike Bridge - main components only
    "SpikeBridge": ("spike_bridge", "SpikeBridge"),
    "OptimizedSpikeBridge": ("spike_bridge", "OptimizedSpikeBridge"),
    "SpikeBridgeConfig": ("spike_bridge", "SpikeBridgeConfig"),
    "SpikeEncodingStrategy": ("spike_bridge", "SpikeEncodingStrategy"),
    "ThroughputMetrics": ("spike_bridge", "ThroughputMetrics"),
}

# Factory functions mapping
_FACTORY_FUNCTIONS = {
    # CPC Encoder factories
    "create_cpc_encoder": ("cpc_encoder", "create_cpc_encoder"),
    "create_enhanced_cpc_encoder": ("cpc_encoder", "create_enhanced_cpc_encoder"),
    
    # SNN Classifier factories
    "create_snn_classifier": ("snn_classifier", "create_snn_classifier"),
    "create_enhanced_snn_classifier": ("snn_classifier", "create_enhanced_snn_classifier"),
    "create_snn_config": ("snn_classifier", "create_snn_config"),
    
    # Spike Bridge factories
    "create_optimized_spike_bridge": ("spike_bridge", "create_optimized_spike_bridge"),
    "create_int8_spike_bridge": ("spike_bridge", "create_int8_spike_bridge"),
    "create_cosine_spike_bridge": ("spike_bridge", "create_cosine_spike_bridge"),
    "create_default_spike_bridge": ("spike_bridge", "create_default_spike_bridge"),
}

# Combine all imports
_ALL_IMPORTS = {**_LAZY_IMPORTS, **_FACTORY_FUNCTIONS}


@dataclass
class ModelsConfig:
    """Central configuration for the models module."""
    
    # CPC Configuration
    cpc_latent_dim: int = 128
    cpc_num_layers: int = 4
    cpc_hidden_dim: int = 256
    
    # SNN Configuration
    snn_hidden_size: int = 128
    snn_num_classes: int = 3
    snn_tau_mem: float = 20e-3
    snn_tau_syn: float = 5e-3
    snn_threshold: float = 1.0
    
    # Spike Bridge Configuration
    spike_time_steps: int = 100
    spike_max_rate: float = 100.0
    spike_dt: float = 1e-3
    spike_encoding: str = "poisson_rate"
    
    # Performance
    use_mixed_precision: bool = True
    enable_jit: bool = True
    memory_efficient: bool = True


def create_models_config(**kwargs) -> ModelsConfig:
    """Create models configuration with overrides."""
    return ModelsConfig(**kwargs)


def get_available_models() -> List[str]:
    """Get list of available model classes."""
    model_classes = [
        name for name, (module, cls) in _LAZY_IMPORTS.items()
        if any(keyword in name for keyword in ["Encoder", "Classifier", "Bridge", "Layer"])
    ]
    return sorted(model_classes)


def get_available_factories() -> List[str]:
    """Get list of available factory functions."""
    return sorted(_FACTORY_FUNCTIONS.keys())


def validate_model_config(config: ModelsConfig) -> bool:
    """Validate models configuration."""
    try:
        # Validate CPC config
        assert config.cpc_latent_dim > 0, "CPC latent_dim must be positive"
        assert config.cpc_num_layers > 0, "CPC num_layers must be positive"
        assert config.cpc_hidden_dim > 0, "CPC hidden_dim must be positive"
        
        # Validate SNN config
        assert config.snn_hidden_size > 0, "SNN hidden_size must be positive"
        assert config.snn_num_classes > 1, "SNN num_classes must be > 1"
        assert config.snn_tau_mem > 0, "SNN tau_mem must be positive"
        assert config.snn_tau_syn > 0, "SNN tau_syn must be positive"
        assert config.snn_threshold > 0, "SNN threshold must be positive"
        
        # Validate Spike Bridge config
        assert config.spike_time_steps > 0, "spike_time_steps must be positive"
        assert config.spike_max_rate > 0, "spike_max_rate must be positive"
        assert config.spike_dt > 0, "spike_dt must be positive"
        
        return True
        
    except AssertionError as e:
        logger.error(f"Model configuration validation failed: {e}")
        return False


def __getattr__(name: str) -> Any:
    """Lazy loading of model components."""
    if name in _ALL_IMPORTS:
        module_name, attr_name = _ALL_IMPORTS[name]
        
        try:
            # Try relative import first
            try:
                module = importlib.import_module(f".{module_name}", package=__name__)
            except (ImportError, ValueError):
                # Fallback to absolute import
                module = importlib.import_module(f"models.{module_name}")
            
            attr = getattr(module, attr_name)
            
            # Cache the attribute for future use
            globals()[name] = attr
            return attr
            
        except Exception as e:
            raise ImportError(f"Cannot import {name} from {module_name}: {str(e)}")
    
    raise AttributeError(f"Module 'models' has no attribute '{name}'")


def __dir__() -> List[str]:
    """Return available attributes for tab completion."""
    return list(_ALL_IMPORTS.keys()) + [
        "ModelsConfig", "create_models_config", "get_available_models",
        "get_available_factories", "validate_model_config", "__version__"
    ]


# Module initialization
logger.info(f"Models module v{__version__} initialized with lazy loading")
logger.info(f"Available models: {len(_LAZY_IMPORTS)} classes, {len(_FACTORY_FUNCTIONS)} factories") 