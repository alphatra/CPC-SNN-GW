"""
Models Module: Neural Network Architectures

Implements the 3-component neuromorphic pipeline:
1. CPC Encoder - Self-supervised representation learning
2. Spike Bridge - Continuous to spike conversion  
3. SNN Classifier - Neuromorphic binary classification
"""

import importlib
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Union, Any

# Module version
__version__ = "1.0.0"

# Module logger
logger = logging.getLogger(__name__)

# Lazy import mappings
_LAZY_IMPORTS = {
    # CPC Encoder components
    "CPCEncoder": ("cpc_encoder", "CPCEncoder"),
    "EnhancedCPCEncoder": ("cpc_encoder", "EnhancedCPCEncoder"),
    "ExperimentConfig": ("cpc_encoder", "ExperimentConfig"),
    "RMSNorm": ("cpc_encoder", "RMSNorm"),
    "WeightNormDense": ("cpc_encoder", "WeightNormDense"),
    "EquinoxGRUWrapper": ("cpc_encoder", "EquinoxGRUWrapper"),
    "info_nce_loss": ("cpc_encoder", "info_nce_loss"),
    "enhanced_info_nce_loss": ("cpc_encoder", "enhanced_info_nce_loss"),
    "create_enhanced_cpc_encoder": ("cpc_encoder", "create_enhanced_cpc_encoder"),
    "create_standard_cpc_encoder": ("cpc_encoder", "create_standard_cpc_encoder"),
    "create_experiment_config": ("cpc_encoder", "create_experiment_config"),
    
    # SNN Classifier components
    "SNNClassifier": ("snn_classifier", "SNNClassifier"),
    "create_snn_classifier": ("snn_classifier", "create_snn_classifier"),
    "SNNTrainer": ("snn_classifier", "SNNTrainer"),
    "LIFLayer": ("snn_classifier", "LIFLayer"),
    "EnhancedSNNClassifier": ("snn_classifier", "EnhancedSNNClassifier"),
    "VectorizedLIFLayer": ("snn_classifier", "VectorizedLIFLayer"),
    "BatchedSNNValidator": ("snn_classifier", "BatchedSNNValidator"),
    "SNNConfig": ("snn_classifier", "SNNConfig"),
    "SurrogateGradientType": ("snn_classifier", "SurrogateGradientType"),
    "create_enhanced_snn_classifier": ("snn_classifier", "create_enhanced_snn_classifier"),
    "create_snn_config": ("snn_classifier", "create_snn_config"),
    "create_surrogate_gradient_fn": ("snn_classifier", "create_surrogate_gradient_fn"),
    "spike_function_with_surrogate": ("snn_classifier", "spike_function_with_surrogate"),
    
    # Spike Bridge components
    "SpikeBridge": ("spike_bridge", "SpikeBridge"),
    "SpikeEncodingStrategy": ("spike_bridge", "SpikeEncodingStrategy"),
    "OptimizedSpikeBridge": ("spike_bridge", "OptimizedSpikeBridge"),
    "SpikeBridgeConfig": ("spike_bridge", "SpikeBridgeConfig"),
    "ThroughputMetrics": ("spike_bridge", "ThroughputMetrics"),
    "create_default_spike_bridge": ("spike_bridge", "create_default_spike_bridge"),
    "create_fast_spike_bridge": ("spike_bridge", "create_fast_spike_bridge"),
    "create_robust_spike_bridge": ("spike_bridge", "create_robust_spike_bridge"),
    "create_spike_bridge_from_string": ("spike_bridge", "create_spike_bridge_from_string"),
    "create_optimized_spike_bridge": ("spike_bridge", "create_optimized_spike_bridge"),
    "create_int8_spike_bridge": ("spike_bridge", "create_int8_spike_bridge"),
    "create_cosine_spike_bridge": ("spike_bridge", "create_cosine_spike_bridge"),
    "create_benchmark_config": ("spike_bridge", "create_benchmark_config"),
}

# Dependency requirements for specific modules
_DEPENDENCY_REQUIREMENTS = {
    "cpc_encoder": {
        "equinox": "pip install equinox",
        "flax": "pip install flax",
        "optax": "pip install optax"
    },
    "snn_classifier": {
        "jax": "pip install jax jaxlib",
        "flax": "pip install flax"
    },
    "spike_bridge": {
        "jax": "pip install jax jaxlib",
        "numpy": "pip install numpy"
    }
}

__all__ = [
    # Module version
    "__version__",
    
    # CPC Encoder components
    "CPCEncoder",
    "EnhancedCPCEncoder",
    "ExperimentConfig",
    "RMSNorm",
    "WeightNormDense",
    "EquinoxGRUWrapper",
    "info_nce_loss",
    "enhanced_info_nce_loss",
    "create_enhanced_cpc_encoder",
    "create_standard_cpc_encoder",
    "create_experiment_config",
    
    # SNN Classifier components
    "SNNClassifier",
    "create_snn_classifier", 
    "SNNTrainer",
    "LIFLayer",
    "EnhancedSNNClassifier",
    "VectorizedLIFLayer",
    "BatchedSNNValidator",
    "SNNConfig",
    "SurrogateGradientType",
    "create_enhanced_snn_classifier",
    "create_snn_config",
    "create_surrogate_gradient_fn",
    "spike_function_with_surrogate",
    
    # Spike Bridge components
    "SpikeBridge",
    "SpikeEncodingStrategy",
    "OptimizedSpikeBridge",
    "SpikeBridgeConfig",
    "ThroughputMetrics",
    "create_default_spike_bridge",
    "create_fast_spike_bridge",
    "create_robust_spike_bridge",
    "create_spike_bridge_from_string",
    "create_optimized_spike_bridge",
    "create_int8_spike_bridge",
    "create_cosine_spike_bridge",
    "create_benchmark_config",
]

# Global models module configuration
@dataclass
class ModelsConfig:
    """Global configuration for models module."""
    # CPC Encoder settings
    cpc_hidden_size: int = 128
    cpc_num_layers: int = 2
    cpc_prediction_steps: int = 12
    cpc_temperature: float = 0.1
    
    # SNN Classifier settings
    snn_hidden_size: int = 256
    snn_num_layers: int = 2
    snn_threshold: float = 1.0
    snn_tau_mem: float = 20.0
    snn_tau_syn: float = 5.0
    
    # Spike Bridge settings
    spike_encoding_strategy: str = "linear"
    spike_time_steps: int = 16
    spike_threshold: float = 0.5
    
    # Training settings
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    
    # Hardware optimization
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = False
    use_jit_compilation: bool = True
    
    # Logging and monitoring
    log_level: str = "INFO"
    wandb_project: str = "cpc-snn-gw"
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Mixed precision requirements
        if self.use_mixed_precision:
            if self.cpc_hidden_size % 8 != 0:
                raise ValueError(f"cpc_hidden_size must be divisible by 8 for mixed precision, got {self.cpc_hidden_size}")
            if self.snn_hidden_size % 8 != 0:
                raise ValueError(f"snn_hidden_size must be divisible by 8 for mixed precision, got {self.snn_hidden_size}")
        
        # Positive values validation
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs}")
            
        # SNN parameters validation
        if self.snn_threshold <= 0:
            raise ValueError(f"snn_threshold must be positive, got {self.snn_threshold}")
        if self.snn_tau_mem <= 0:
            raise ValueError(f"snn_tau_mem must be positive, got {self.snn_tau_mem}")
        if self.snn_tau_syn <= 0:
            raise ValueError(f"snn_tau_syn must be positive, got {self.snn_tau_syn}")
            
        # Spike encoding validation
        valid_strategies = ["linear", "poisson", "threshold", "rate"]
        if self.spike_encoding_strategy not in valid_strategies:
            raise ValueError(f"spike_encoding_strategy must be one of {valid_strategies}, got {self.spike_encoding_strategy}")
        if self.spike_time_steps <= 0:
            raise ValueError(f"spike_time_steps must be positive, got {self.spike_time_steps}")
            
        # Temperature validation
        if self.cpc_temperature <= 0:
            raise ValueError(f"cpc_temperature must be positive, got {self.cpc_temperature}")
            
        # Prediction steps validation
        if self.cpc_prediction_steps <= 0:
            raise ValueError(f"cpc_prediction_steps must be positive, got {self.cpc_prediction_steps}")

# Global config instance
_global_config = ModelsConfig()

def get_models_config() -> ModelsConfig:
    """Get global models module configuration."""
    return _global_config

def set_models_config(config: ModelsConfig):
    """Set global models module configuration."""
    global _global_config
    _global_config = config
    logger.info(f"Models module config updated: CPC hidden_size={config.cpc_hidden_size}, SNN hidden_size={config.snn_hidden_size}")

# Enhanced lazy import system with dependency checking
def __getattr__(name):
    """
    Enhanced lazy import with comprehensive error handling and dependency checking.
    
    Args:
        name: Name of the attribute to import
        
    Returns:
        The imported attribute
        
    Raises:
        AttributeError: If the attribute doesn't exist
        ImportError: If required dependencies are missing
    """
    if name == "__version__":
        return __version__
    
    # Check if name is in lazy imports
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        
        try:
            # Check for missing dependencies
            if module_name in _DEPENDENCY_REQUIREMENTS:
                missing_deps = []
                for dep, install_cmd in _DEPENDENCY_REQUIREMENTS[module_name].items():
                    try:
                        importlib.import_module(dep)
                    except ImportError:
                        missing_deps.append((dep, install_cmd))
                
                if missing_deps:
                    dep_names = [dep for dep, _ in missing_deps]
                    install_cmds = [cmd for _, cmd in missing_deps]
                    logger.warning(
                        f"Missing optional dependencies for {module_name}: {dep_names}. "
                        f"Install with: {'; '.join(install_cmds)}"
                    )
            
            # Dynamic import with better error handling
            try:
                module = importlib.import_module(module_name, package=__name__)
            except ImportError:
                # Fallback to absolute import for better compatibility
                try:
                    full_module_name = f"ligo_cpc_snn.models.{module_name}"
                    module = importlib.import_module(full_module_name)
                except ImportError as e:
                    raise ImportError(
                        f"Failed to import {module_name} from models package. "
                        f"Ensure the module exists and dependencies are installed. "
                        f"Original error: {e}"
                    )
            
            # Get the attribute from the module
            if hasattr(module, attr_name):
                attr = getattr(module, attr_name)
                
                # Cache the attribute for faster future access
                globals()[name] = attr
                
                logger.debug(f"Lazy loaded: {name} from {module_name}")
                return attr
            else:
                raise AttributeError(f"Module {module_name} has no attribute '{attr_name}'")
                
        except ImportError as e:
            # More informative error message
            error_msg = f"Cannot import {name} from {module_name}: {str(e)}"
            
            if module_name in _DEPENDENCY_REQUIREMENTS:
                deps = _DEPENDENCY_REQUIREMENTS[module_name]
                error_msg += f"\nMissing dependencies? Try: {'; '.join(deps.values())}"
            
            raise ImportError(error_msg)
        
        except Exception as e:
            logger.error(f"Unexpected error importing {name}: {str(e)}")
            raise AttributeError(f"Failed to import {name}: {str(e)}")
    
    # Handle special config attributes
    elif name == "models_config":
        return get_models_config()
    
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Convenience functions for unified model creation
def create_unified_model(config: Optional[ModelsConfig] = None) -> Dict[str, Any]:
    """
    Create a unified model with all components.
    
    Args:
        config: Optional configuration override
        
    Returns:
        Dictionary containing all model components
    """
    if config is None:
        config = get_models_config()
    
    # Lazy import the creation functions
    cpc_encoder = create_enhanced_cpc_encoder(
        hidden_size=config.cpc_hidden_size,
        num_layers=config.cpc_num_layers,
        prediction_steps=config.cpc_prediction_steps,
        temperature=config.cpc_temperature
    )
    
    spike_bridge = create_default_spike_bridge(
        encoding_strategy=config.spike_encoding_strategy,
        time_steps=config.spike_time_steps,
        threshold=config.spike_threshold
    )
    
    snn_classifier = create_enhanced_snn_classifier(
        hidden_size=config.snn_hidden_size,
        num_layers=config.snn_num_layers,
        threshold=config.snn_threshold,
        tau_mem=config.snn_tau_mem,
        tau_syn=config.snn_tau_syn
    )
    
    return {
        "cpc_encoder": cpc_encoder,
        "spike_bridge": spike_bridge,
        "snn_classifier": snn_classifier,
        "config": config
    }

def check_model_dependencies(module_name: Optional[str] = None) -> Dict[str, bool]:
    """
    Check which model dependencies are available.
    
    Args:
        module_name: Specific module to check (optional)
        
    Returns:
        Dictionary mapping dependency names to availability
    """
    results = {}
    
    modules_to_check = [module_name] if module_name else _DEPENDENCY_REQUIREMENTS.keys()
    
    for mod_name in modules_to_check:
        if mod_name in _DEPENDENCY_REQUIREMENTS:
            for dep_name in _DEPENDENCY_REQUIREMENTS[mod_name].keys():
                try:
                    importlib.import_module(dep_name)
                    results[dep_name] = True
                except ImportError:
                    results[dep_name] = False
    
    return results

def get_missing_model_dependencies(module_name: Optional[str] = None) -> Dict[str, str]:
    """
    Get missing model dependencies and their install commands.
    
    Args:
        module_name: Specific module to check (optional)
        
    Returns:
        Dictionary mapping missing dependency names to install commands
    """
    missing = {}
    
    modules_to_check = [module_name] if module_name else _DEPENDENCY_REQUIREMENTS.keys()
    
    for mod_name in modules_to_check:
        if mod_name in _DEPENDENCY_REQUIREMENTS:
            for dep_name, install_cmd in _DEPENDENCY_REQUIREMENTS[mod_name].items():
                try:
                    importlib.import_module(dep_name)
                except ImportError:
                    missing[dep_name] = install_cmd
    
    return missing 