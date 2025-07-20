"""
Data Module: GWOSC Gravitational Wave Data Pipeline

Handles downloading, preprocessing, and quality validation of
LIGO gravitational wave data from GWOSC (Gravitational Wave Open Science Center).
Includes continuous GW generation and unified label management.
"""

import importlib
import logging

# Module version
__version__ = "1.0.0"

# Module logger
logger = logging.getLogger(__name__)

# Lazy import mappings
_LAZY_IMPORTS = {
    # GWOSC downloaders and preprocessing
    "GWOSCDownloader": ("gw_download", "GWOSCDownloader"),
    "DataPreprocessor": ("gw_download", "DataPreprocessor"),
    "ProductionGWOSCDownloader": ("gw_download", "ProductionGWOSCDownloader"),
    "AdvancedDataPreprocessor": ("gw_download", "AdvancedDataPreprocessor"),
    "QualityMetrics": ("gw_download", "QualityMetrics"),
    "ProcessingResult": ("gw_download", "ProcessingResult"),
    "SegmentSampler": ("gw_download", "SegmentSampler"),
    
    # Professional cache management
    "ProfessionalCacheManager": ("cache_manager", "ProfessionalCacheManager"),
    "CacheMetadata": ("cache_manager", "CacheMetadata"),
    "get_cache_manager": ("cache_manager", "get_cache_manager"),
    "cache_decorator": ("cache_manager", "cache_decorator"),
    
    # Continuous GW generation
    "ContinuousGWGenerator": ("continuous_gw_generator", "ContinuousGWGenerator"),
    "ContinuousGWParams": ("continuous_gw_generator", "ContinuousGWParams"),
    "SignalConfiguration": ("continuous_gw_generator", "SignalConfiguration"),
    "create_mixed_gw_dataset": ("continuous_gw_generator", "create_mixed_gw_dataset"),
    
    # Label utilities
    "GWSignalType": ("label_utils", "GWSignalType"),
    "CANONICAL_LABELS": ("label_utils", "CANONICAL_LABELS"),
    "LABEL_NAMES": ("label_utils", "LABEL_NAMES"),
    "LABEL_DESCRIPTIONS": ("label_utils", "LABEL_DESCRIPTIONS"),
    "LABEL_COLORS": ("label_utils", "LABEL_COLORS"),
    "LABEL_COLORS_SCIENTIFIC": ("label_utils", "LABEL_COLORS_SCIENTIFIC"),
    "LABEL_COLORS_COLORBLIND": ("label_utils", "LABEL_COLORS_COLORBLIND"),
    "COLOR_SCHEMES": ("label_utils", "COLOR_SCHEMES"),
    "LabelError": ("label_utils", "LabelError"),
    "LabelValidationResult": ("label_utils", "LabelValidationResult"),
    "normalize_labels": ("label_utils", "normalize_labels"),
    "convert_legacy_labels": ("label_utils", "convert_legacy_labels"),
    "validate_labels": ("label_utils", "validate_labels"),
    "validate_dataset_labels": ("label_utils", "validate_dataset_labels"),
    "get_class_weights": ("label_utils", "get_class_weights"),
    "get_cmap_colors": ("label_utils", "get_cmap_colors"),
    "create_label_visualization_config": ("label_utils", "create_label_visualization_config"),
    "create_label_report": ("label_utils", "create_label_report"),
    "dataset_to_canonical": ("label_utils", "dataset_to_canonical"),
    "log_dataset_info": ("label_utils", "log_dataset_info"),
}

# Dependency requirements for specific modules
_DEPENDENCY_REQUIREMENTS = {
    "continuous_gw_generator": {
        "h5py": "pip install h5py",
        "tensorflow": "pip install tensorflow",
        "tensorflow_datasets": "pip install tensorflow-datasets",
        "pyfstat": "pip install pyfstat"
    },
    "cache_manager": {
        "h5py": "pip install h5py"
    },
    "gw_download": {
        "gwpy": "pip install gwpy",
        "ligo.segments": "pip install ligo-segments"
    }
}

__all__ = [
    # Module version
    "__version__",
    
    # GWOSC downloaders and preprocessing
    "GWOSCDownloader",
    "DataPreprocessor", 
    "ProductionGWOSCDownloader",
    "AdvancedDataPreprocessor",
    "QualityMetrics",
    "ProcessingResult",
    "SegmentSampler",
    
    # Professional cache management
    "ProfessionalCacheManager",
    "CacheMetadata",
    "get_cache_manager",
    "cache_decorator",
    
    # Continuous GW generation
    "ContinuousGWGenerator",
    "ContinuousGWParams", 
    "SignalConfiguration",
    "create_mixed_gw_dataset",
    
    # Label utilities
    "GWSignalType",
    "CANONICAL_LABELS",
    "LABEL_NAMES", 
    "LABEL_DESCRIPTIONS",
    "LABEL_COLORS",
    "LABEL_COLORS_SCIENTIFIC",
    "LABEL_COLORS_COLORBLIND",
    "COLOR_SCHEMES",
    "LabelError",
    "LabelValidationResult",
    "normalize_labels",
    "convert_legacy_labels",
    "validate_labels",
    "validate_dataset_labels",
    "get_class_weights",
    "get_cmap_colors",
    "create_label_visualization_config",
    "create_label_report",
    "dataset_to_canonical",
    "log_dataset_info",
    
    # Health check function
    "run_health_check",
]

# Global data module configuration
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class DataConfig:
    """Global configuration for data module."""
    sampling_rate: int = 4096
    default_cache_dir: str = "data_cache"
    max_cache_size_gb: float = 10.0
    default_timeout: int = 30
    enable_quality_checks: bool = True
    gwosc_urls: Dict[str, str] = None
    
    def __post_init__(self):
        if self.gwosc_urls is None:
            self.gwosc_urls = {
                "main": "https://gwosc.org/",
                "api": "https://gwosc.org/api/",
                "data": "https://gwosc.org/data/"
            }

# Global config instance
_global_config = DataConfig()

def get_data_config() -> DataConfig:
    """Get global data module configuration."""
    return _global_config

def set_data_config(config: DataConfig):
    """Set global data module configuration."""
    global _global_config
    _global_config = config
    logger.info(f"Data module config updated: sampling_rate={config.sampling_rate}")

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
            
            # Dynamic import
            full_module_name = f"{__name__}.{module_name}"
            module = importlib.import_module(full_module_name)
            
            # Get the attribute from the module
            if hasattr(module, attr_name):
                attr = getattr(module, attr_name)
                
                # Cache the attribute for faster future access
                globals()[name] = attr
                
                return attr
            else:
                raise AttributeError(f"Module {full_module_name} has no attribute '{attr_name}'")
                
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
    elif name == "data_config":
        return get_data_config()
    
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Add some convenience functions
def check_dependencies(module_name: Optional[str] = None) -> Dict[str, bool]:
    """
    Check which dependencies are available.
    
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

def get_missing_dependencies(module_name: Optional[str] = None) -> Dict[str, str]:
    """
    Get missing dependencies and their install commands.
    
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

def run_health_check():
    """
    Runs a health check on the data module to verify imports and dependencies.
    
    Returns:
        bool: True if all checks pass, False otherwise
    """
    logger.info("Running data module health check...")
    all_ok = True
    
    # 1. Check all lazy imports
    for name in __all__:
        if name == "__version__":
            continue
        try:
            # This will trigger the lazy import mechanism in __getattr__
            # Use globals() to access the current module 
            import sys
            current_module = sys.modules[__name__]
            getattr(current_module, name)
            logger.info(f"✅ Successfully imported '{name}'")
        except (AttributeError, ImportError) as e:
            logger.error(f"❌ Failed to import '{name}': {e}")
            all_ok = False
            
    # 2. Report missing dependencies
    missing = get_missing_dependencies()
    if missing:
        logger.warning("🔥 Missing dependencies detected:")
        for dep, cmd in missing.items():
            logger.warning(f"  - {dep} (install with: {cmd})")
        all_ok = False
    else:
        logger.info("✅ All optional dependencies are installed.")
        
    # 3. Test basic functionality
    try:
        # Test global config
        config = get_data_config()
        logger.info(f"✅ Global config loaded: sampling_rate={config.sampling_rate}")
        
        # Test dependency checking
        deps = check_dependencies()
        logger.info(f"✅ Dependency check completed: {len(deps)} dependencies checked")
        
    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        all_ok = False
        
    if all_ok:
        logger.info("🎉 Health check passed. Data module is ready.")
    else:
        logger.error("❌ Health check failed. Please install missing dependencies.")
        
    return all_ok 