"""
CPC+SNN Neuromorphic Gravitational Wave Detection

World's first neuromorphic gravitational wave detector using 
Contrastive Predictive Coding + Spiking Neural Networks.

Designed for production deployment following ML4GW standards.
"""

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.1.0-dev"
    __version_tuple__ = (0, 1, 0, "dev")

# Extended version information
__author__ = "CPC-SNN-GW Team"
__email__ = "contact@cpc-snn-gw.dev"
__license__ = "MIT"
__description__ = "Neuromorphic Gravitational Wave Detection using CPC+SNN"
__url__ = "https://github.com/cpc-snn-gw/ligo-cpc-snn"

# Build information
import datetime
__build_date__ = datetime.datetime.now().isoformat()
__build_info__ = {
    "version": __version__,
    "version_tuple": __version_tuple__,
    "build_date": __build_date__,
    "python_version": None,  # Will be set later
    "jax_version": None,
    "dependencies": {}
}

# Runtime version checking
def get_version_info():
    """Get comprehensive version information."""
    import sys
    import platform
    
    try:
        import jax
        jax_version = jax.__version__
    except ImportError:
        jax_version = "not available"
    
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = "not available"
    
    version_info = {
        "ligo_cpc_snn": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "jax": jax_version,
        "numpy": numpy_version,
        "build_date": __build_date__
    }
    
    return version_info

def print_version_info():
    """Print comprehensive version information."""
    info = get_version_info()
    print("="*60)
    print("LIGO CPC-SNN Neuromorphic GW Detection")
    print("="*60)
    print(f"Version: {info['ligo_cpc_snn']}")
    print(f"Build Date: {info['build_date']}")
    print(f"Python: {info['python']}")
    print(f"Platform: {info['platform']}")
    print(f"JAX: {info['jax']}")
    print(f"NumPy: {info['numpy']}")
    print("="*60)

# Unified export/import functions for easier usage
def export_dataset(dataset, output_path, format='hdf5', **kwargs):
    """
    Unified dataset export function supporting multiple formats.
    
    Args:
        dataset: Dataset to export
        output_path: Output file path
        format: Export format ('hdf5', 'json', 'numpy')
        **kwargs: Additional format-specific arguments
        
    Returns:
        True if successful
    """
    from .data.gw_synthetic_generator import ContinuousGWGenerator
    
    if format.lower() == 'hdf5':
        generator = ContinuousGWGenerator()
        return generator.export_dataset_to_hdf5(dataset, output_path, **kwargs)
    elif format.lower() == 'json':
        import json
        try:
            # Convert JAX arrays to lists for JSON serialization
            json_data = {}
            for key, value in dataset.items():
                if hasattr(value, 'tolist'):
                    json_data[key] = value.tolist()
                else:
                    json_data[key] = value
            
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            return True
        except Exception as e:
            print(f"JSON export failed: {e}")
            return False
    elif format.lower() == 'numpy':
        import numpy as np
        try:
            np.savez_compressed(output_path, **dataset)
            return True
        except Exception as e:
            print(f"NumPy export failed: {e}")
            return False
    else:
        raise ValueError(f"Unsupported format: {format}")

def import_dataset(input_path, format='auto'):
    """
    Unified dataset import function supporting multiple formats.
    
    Args:
        input_path: Input file path
        format: Import format ('auto', 'hdf5', 'json', 'numpy')
        
    Returns:
        Loaded dataset or None if failed
    """
    from pathlib import Path
    
    input_path = Path(input_path)
    
    # Auto-detect format
    if format == 'auto':
        if input_path.suffix == '.h5':
            format = 'hdf5'
        elif input_path.suffix == '.json':
            format = 'json'
        elif input_path.suffix in ['.npz', '.npy']:
            format = 'numpy'
        else:
            raise ValueError(f"Cannot auto-detect format for {input_path}")
    
    if format.lower() == 'hdf5':
        # ✅ Fixed: Direct HDF5 loading instead of non-existent method
        try:
            import h5py
            import numpy as np
            with h5py.File(input_path, 'r') as f:
                dataset = {}
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        dataset[key] = np.array(f[key])
                    elif isinstance(f[key], h5py.Group):
                        # Handle groups (nested structure)
                        group_data = {}
                        for subkey in f[key].keys():
                            group_data[subkey] = np.array(f[key][subkey])
                        dataset[key] = group_data
                return dataset
        except ImportError:
            print("h5py not available. Install with: pip install h5py")
            return None
        except Exception as e:
            print(f"HDF5 import failed: {e}")
            return None
    elif format.lower() == 'json':
        import json
        try:
            with open(input_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"JSON import failed: {e}")
            return None
    elif format.lower() == 'numpy':
        import numpy as np
        try:
            return dict(np.load(input_path))
        except Exception as e:
            print(f"NumPy import failed: {e}")
            return None
    else:
        raise ValueError(f"Unsupported format: {format}")

def create_training_pipeline(config=None):
    """
    Create unified training pipeline with CPC+SNN+Spike components.
    
    Args:
        config: Training configuration (optional)
        
    Returns:
        Training pipeline object
    """
    from .training import create_cpc_snn_trainer
    
    return create_cpc_snn_trainer(config)

def quick_start_demo():
    """
    Quick start demonstration of the CPC-SNN-GW system.
    
    Generates sample data, trains a model, and shows results.
    """
    print("🚀 CPC-SNN-GW Quick Start Demo")
    print("="*50)
    
    # Generate sample data
    from .data import ContinuousGWGenerator
    generator = ContinuousGWGenerator(duration=1.0)
    
    print("1. Generating sample GW data...")
    dataset = generator.generate_training_dataset(num_signals=5)
    print(f"   Generated {len(dataset['data'])} samples")
    
    # Create training pipeline
    print("2. Creating training pipeline...")
    pipeline = create_training_pipeline()
    print("   Pipeline created successfully")
    
    # Show version info
    print("3. System information:")
    print_version_info()
    
    print("\n✅ Demo completed successfully!")
    print("Next steps: Use export_dataset() to save your data")

# Complete alphabetized __all__ list - Single source of truth
__all__ = [
    # Core metadata
    "__version__",
    "__version_tuple__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
    "__url__",
    "__build_date__",
    "__build_info__",
    "get_version_info",
    "print_version_info",
    "export_dataset",
    "import_dataset", 
    "create_training_pipeline",
    "quick_start_demo",
    
    # Data module exports (alphabetical)
    "AdvancedDataPreprocessor",
    "CANONICAL_LABELS",
    "CacheMetadata",
    "COLOR_SCHEMES",
    "ContinuousGWGenerator",
    "ContinuousGWParams",
    "DataPreprocessor",
    "GWOSCDownloader",
    "GWSignalType",
    "LABEL_COLORS",
    "LABEL_COLORS_COLORBLIND",
    "LABEL_COLORS_SCIENTIFIC",
    "LABEL_DESCRIPTIONS",
    "LABEL_NAMES",
    "LabelError",
    "LabelValidationResult",
    "ProductionGWOSCDownloader",
    "ProfessionalCacheManager",
    "ProcessingResult",
    "QualityMetrics",
    "SegmentSampler",
    "SignalConfiguration",
    "cache_decorator",
    "convert_legacy_labels",
    "create_label_report",
    "create_label_visualization_config",
    "create_mixed_gw_dataset",
    "dataset_to_canonical",
    "get_cache_manager",
    "get_class_weights",
    "get_cmap_colors",
    "log_dataset_info",
    "normalize_labels",
    "validate_dataset_labels",
    "validate_labels",
    
    # Models module exports (alphabetical)
    "BatchedSNNValidator",
    "CPCEncoder", 
    "CPCPretrainer",
    "EnhancedCPCEncoder",
    "EnhancedSNNClassifier",
    "EquinoxGRUWrapper",
    "ExperimentConfig",
    "LIFLayer",
    "OptimizedSpikeBridge",
    "RMSNorm",
    "SNNClassifier", 
    "SNNConfig",
    "SNNTrainer",
    "SpikeBridge",
    "SpikeBridgeConfig",
    "SpikeEncodingStrategy",
    "SurrogateGradientType",
    "ThroughputMetrics",
    "VectorizedLIFLayer",
    "WeightNormDense",
    "create_benchmark_config",
    "create_cosine_spike_bridge",
    "create_default_spike_bridge",
    "create_enhanced_cpc_encoder",
    "create_enhanced_snn_classifier",
    "create_experiment_config",
    "create_fast_spike_bridge", 
    "create_int8_spike_bridge",
    "create_optimized_spike_bridge",
    "create_robust_spike_bridge",
    "create_snn_classifier",
    "create_snn_config",
    "create_spike_bridge_from_string",
    "create_standard_cpc_encoder",
    "create_surrogate_gradient_fn",
    "enhanced_info_nce_loss",
    "info_nce_loss",
    "spike_function_with_surrogate",
    
    # Training module exports (alphabetical)
    "AdvancedGWTrainer",
    "CPCPretrainer",
    "CPCSNNTrainer",
    "EnhancedGWTrainer",
    "HydraTrainerMixin",
    "TrainerBase",
    "TrainingConfig",
    "TrainingMetrics",
    "create_cpc_snn_cli_app",
    "create_cpc_snn_trainer",
    "create_enhanced_gw_trainer",
    "create_hydra_cli_app",
    "create_training_config",
    "pretrain_cpc_main",
    "run_advanced_training_experiment",
    
    # Utils module exports (alphabetical)
    "ML4GW_PROJECT_STRUCTURE",
    "create_directory_structure",
    "get_jax_device_info",
    "print_system_info",
    "setup_logging",
    "validate_array_shape",
]

# Lazy import system with __getattr__ fallback
def __getattr__(name):
    """Lazy import system for optional dependencies and compatibility."""
    
    # Data module imports
    if name == "AdvancedDataPreprocessor":
        from .data.gw_download import AdvancedDataPreprocessor
        return AdvancedDataPreprocessor
    elif name == "CANONICAL_LABELS":
        from .data.label_utils import CANONICAL_LABELS
        return CANONICAL_LABELS
    elif name == "CacheMetadata":
        from .data.cache_manager import CacheMetadata
        return CacheMetadata
    elif name == "COLOR_SCHEMES":
        from .data.label_utils import COLOR_SCHEMES
        return COLOR_SCHEMES
    elif name == "ContinuousGWGenerator":
        from .data.gw_synthetic_generator import ContinuousGWGenerator
        return ContinuousGWGenerator
    elif name == "ContinuousGWParams":
        from .data.gw_signal_params import ContinuousGWParams
        return ContinuousGWParams
    elif name == "DataPreprocessor":
        from .data.gw_download import DataPreprocessor
        return DataPreprocessor
    elif name == "GWOSCDownloader":
        from .data.gw_download import GWOSCDownloader
        return GWOSCDownloader
    elif name == "GWSignalType":
        from .data.label_utils import GWSignalType
        return GWSignalType
    elif name == "LABEL_COLORS":
        from .data.label_utils import LABEL_COLORS
        return LABEL_COLORS
    elif name == "LABEL_COLORS_COLORBLIND":
        from .data.label_utils import LABEL_COLORS_COLORBLIND
        return LABEL_COLORS_COLORBLIND
    elif name == "LABEL_COLORS_SCIENTIFIC":
        from .data.label_utils import LABEL_COLORS_SCIENTIFIC
        return LABEL_COLORS_SCIENTIFIC
    elif name == "LABEL_DESCRIPTIONS":
        from .data.label_utils import LABEL_DESCRIPTIONS
        return LABEL_DESCRIPTIONS
    elif name == "LABEL_NAMES":
        from .data.label_utils import LABEL_NAMES
        return LABEL_NAMES
    elif name == "LabelError":
        from .data.label_utils import LabelError
        return LabelError
    elif name == "LabelValidationResult":
        from .data.label_utils import LabelValidationResult
        return LabelValidationResult
    elif name == "ProductionGWOSCDownloader":
        from .data.gw_download import ProductionGWOSCDownloader
        return ProductionGWOSCDownloader
    elif name == "ProfessionalCacheManager":
        from .data.cache_manager import ProfessionalCacheManager
        return ProfessionalCacheManager
    elif name == "ProcessingResult":
        from .data.gw_download import ProcessingResult
        return ProcessingResult
    elif name == "QualityMetrics":
        from .data.gw_download import QualityMetrics
        return QualityMetrics
    elif name == "SegmentSampler":
        from .data.gw_download import SegmentSampler
        return SegmentSampler
    elif name == "SignalConfiguration":
        from .data.gw_signal_params import SignalConfiguration
        return SignalConfiguration
    elif name == "cache_decorator":
        from .data.cache_manager import cache_decorator
        return cache_decorator
    elif name == "convert_legacy_labels":
        from .data.label_utils import convert_legacy_labels
        return convert_legacy_labels
    elif name == "create_label_report":
        from .data.label_utils import create_label_report
        return create_label_report
    elif name == "create_label_visualization_config":
        from .data.label_utils import create_label_visualization_config
        return create_label_visualization_config
    elif name == "create_mixed_gw_dataset":
        from .data.gw_dataset_builder import create_mixed_gw_dataset
        return create_mixed_gw_dataset
    elif name == "dataset_to_canonical":
        from .data.label_utils import dataset_to_canonical
        return dataset_to_canonical
    elif name == "get_cache_manager":
        from .data.cache_manager import get_cache_manager
        return get_cache_manager
    elif name == "get_class_weights":
        from .data.label_utils import get_class_weights
        return get_class_weights
    elif name == "get_cmap_colors":
        from .data.label_utils import get_cmap_colors
        return get_cmap_colors
    elif name == "log_dataset_info":
        from .data.label_utils import log_dataset_info
        return log_dataset_info
    elif name == "normalize_labels":
        from .data.label_utils import normalize_labels
        return normalize_labels
    elif name == "validate_dataset_labels":
        from .data.label_utils import validate_dataset_labels
        return validate_dataset_labels
    elif name == "validate_labels":
        from .data.label_utils import validate_labels
        return validate_labels
    
    # Models module imports
    elif name == "CPCEncoder":
        from .models.cpc.core import CPCEncoder
        return CPCEncoder
    elif name == "EnhancedCPCEncoder":
        from .models.cpc.core import EnhancedCPCEncoder
        return EnhancedCPCEncoder
    elif name == "EquinoxGRUWrapper":
        from .models.cpc_components import EquinoxGRUWrapper
        return EquinoxGRUWrapper
    elif name == "ExperimentConfig":
        from .models.cpc.config import ExperimentConfig
        return ExperimentConfig
    elif name == "EnhancedSNNClassifier":
        from .models.snn.core import EnhancedSNNClassifier
        return EnhancedSNNClassifier
    elif name == "LIFLayer":
        from .models.snn.layers import LIFLayer
        return LIFLayer
    elif name == "OptimizedSpikeBridge":
        from .models.bridge.core import ValidatedSpikeBridge
        return ValidatedSpikeBridge
    elif name == "RMSNorm":
        from .models.cpc_components import RMSNorm
        return RMSNorm
    elif name == "SNNClassifier":
        from .models.snn.core import SNNClassifier
        return SNNClassifier
    elif name == "SNNConfig":
        from .models.snn.config import SNNConfig
        return SNNConfig
    elif name == "SNNTrainer":
        from .models.snn.trainer import SNNTrainer
        return SNNTrainer
    elif name == "SpikeBridge":
        from .models.bridge.core import ValidatedSpikeBridge
        return ValidatedSpikeBridge
    elif name == "SpikeEncodingStrategy":
        from .models.bridge.encoders import SpikeEncodingStrategy
        return SpikeEncodingStrategy
    elif name == "SurrogateGradientType":
        from .models.snn_utils import SurrogateGradientType
        return SurrogateGradientType
    elif name == "WeightNormDense":
        from .models.cpc_components import WeightNormDense
        return WeightNormDense
    elif name == "VectorizedLIFLayer":
        from .models.snn.layers import VectorizedLIFLayer
        return VectorizedLIFLayer
    elif name == "create_default_spike_bridge":
        from .models.bridge.core import create_default_spike_bridge
        return create_default_spike_bridge
    elif name == "create_enhanced_cpc_encoder":
        from .models.cpc.factory import create_enhanced_cpc_encoder
        return create_enhanced_cpc_encoder
    elif name == "create_enhanced_snn_classifier":
        from .models.snn.factory import create_enhanced_snn_classifier
        return create_enhanced_snn_classifier
    elif name == "create_experiment_config":
        from .models.cpc.factory import create_experiment_config
        return create_experiment_config
    elif name == "create_fast_spike_bridge":
        from .models.bridge.core import create_fast_spike_bridge
        return create_fast_spike_bridge
    elif name == "create_robust_spike_bridge":
        from .models.bridge.core import create_robust_spike_bridge
        return create_robust_spike_bridge
    elif name == "create_snn_classifier":
        from .models.snn.factory import create_snn_classifier
        return create_snn_classifier
    elif name == "create_snn_config":
        from .models.snn.factory import create_snn_config
        return create_snn_config
    elif name == "create_spike_bridge_from_string":
        from .models.bridge.core import create_spike_bridge_from_string
        return create_spike_bridge_from_string
    elif name == "create_standard_cpc_encoder":
        from .models.cpc.factory import create_standard_cpc_encoder
        return create_standard_cpc_encoder
    elif name == "create_surrogate_gradient_fn":
        from .models.snn_utils import create_surrogate_gradient_fn
        return create_surrogate_gradient_fn
    elif name == "enhanced_info_nce_loss":
        from .models.cpc.losses import enhanced_info_nce_loss
        return enhanced_info_nce_loss
    elif name == "info_nce_loss":
        from .models.cpc.losses import info_nce_loss
        return info_nce_loss
    elif name == "spike_function_with_surrogate":
        from .models.snn_utils import spike_function_with_surrogate
        return spike_function_with_surrogate
    elif name == "BatchedSNNValidator":
        from .models.snn_utils import BatchedSNNValidator
        return BatchedSNNValidator
    
    # Training module imports
    elif name == "AdvancedGWTrainer":
        # Backward-compatibility alias to modular advanced trainer
        from .training.advanced import RealAdvancedGWTrainer as AdvancedGWTrainer
        return AdvancedGWTrainer
    elif name == "CPCPretrainer":
        from .training.pretrain_cpc import CPCPretrainer
        return CPCPretrainer
    elif name == "CPCSNNTrainer":
        from .training.base_trainer import CPCSNNTrainer
        return CPCSNNTrainer
    elif name == "HydraTrainerMixin":
        from .training.base_trainer import HydraTrainerMixin
        return HydraTrainerMixin
    elif name == "TrainerBase":
        from .training.base_trainer import TrainerBase
        return TrainerBase
    elif name == "TrainingConfig":
        from .training.base_trainer import TrainingConfig
        return TrainingConfig
    elif name == "TrainingMetrics":
        from .training.base_trainer import TrainingMetrics
        return TrainingMetrics
    elif name == "create_cpc_snn_cli_app":
        from .training.base_trainer import create_cpc_snn_cli_app
        return create_cpc_snn_cli_app
    elif name == "create_cpc_snn_trainer":
        from .training.base_trainer import create_cpc_snn_trainer
        return create_cpc_snn_trainer
    elif name == "create_enhanced_gw_trainer":
        from .training.enhanced_gw_training import create_enhanced_gw_trainer
        return create_enhanced_gw_trainer
    elif name == "create_hydra_cli_app":
        from .training.base_trainer import create_hydra_cli_app
        return create_hydra_cli_app
    elif name == "create_training_config":
        from .training.base_trainer import create_training_config
        return create_training_config
    elif name == "pretrain_cpc_main":
        from .training.pretrain_cpc import main as pretrain_cpc_main
        return pretrain_cpc_main
    elif name == "run_advanced_training_experiment":
        # Deprecated: no direct experiment runner in modular API
        import warnings
        warnings.warn(
            "run_advanced_training_experiment is deprecated. "
            "Use training.advanced.create_real_advanced_trainer and invoke your run loop.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise AttributeError(
            "run_advanced_training_experiment has been removed; "
            "construct a trainer via training.advanced.create_real_advanced_trainer(config)"
        )
    
    # Utils module imports
    elif name == "ML4GW_PROJECT_STRUCTURE":
        from .utils import ML4GW_PROJECT_STRUCTURE
        return ML4GW_PROJECT_STRUCTURE
    elif name == "create_directory_structure":
        from .utils import create_directory_structure
        return create_directory_structure
    elif name == "get_jax_device_info":
        from .utils import get_jax_device_info
        return get_jax_device_info
    elif name == "print_system_info":
        from .utils import print_system_info
        return print_system_info
    elif name == "setup_logging":
        from .utils import setup_logging
        return setup_logging
    elif name == "validate_array_shape":
        from .utils import validate_array_shape
        return validate_array_shape
    
    # Compatibility fallback
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Package metadata
__title__ = "ligo-cpc-snn"
__description__ = "Neuromorphic gravitational wave detection using CPC + Spiking Neural Networks"
__author__ = "Gracjan"
__email__ = "contact@ml4gw-neuromorphic.org"
__license__ = "MIT"
__copyright__ = "Copyright 2025 CPC+SNN Neuromorphic GW Detection Project" 

# Optional dependencies check with informative error messages
def _check_optional_dependencies():
    """Check availability of optional dependencies."""
    try:
        import optax
        import equinox
        import haiku
    except ImportError as e:
        import warnings
        warnings.warn(
            f"Optional dependency not available: {e}. "
            "Some advanced features may be limited. "
            "Install with: pip install ligo-cpc-snn[full]",
            UserWarning,
            stacklevel=2
        )

# Auto-check on import (non-blocking)
try:
    _check_optional_dependencies()
except Exception:
    pass  # Silent fallback for production environments 