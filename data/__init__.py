"""
Data Module: GWOSC Gravitational Wave Data Pipeline

Handles downloading, preprocessing, and quality validation of
LIGO gravitational wave data from GWOSC (Gravitational Wave Open Science Center).
Includes continuous GW generation and unified label management.
"""

import importlib
import logging
import warnings

# Module version
__version__ = "1.0.0"

# Module logger
logger = logging.getLogger(__name__)

# ✅ FIXED: Lazy import mappings (ReadLIGO instead of GWOSC)
_LAZY_IMPORTS = {
    # ✅ ReadLIGO downloaders and preprocessing  
    "ReadLIGODownloader": ("gw_download", "ReadLIGODownloader"),
    "LIGODownloader": ("gw_download", "LIGODownloader"),
    "DataPreprocessor": ("gw_download", "DataPreprocessor"),
    "AdvancedDataPreprocessor": ("gw_download", "AdvancedDataPreprocessor"),
    "QualityMetrics": ("gw_download", "QualityMetrics"),
    "ProcessingResult": ("gw_download", "ProcessingResult"),
    "SegmentSampler": ("gw_download", "SegmentSampler"),
    
    # ✅ MLGWSC-1 data loader (NEW)
    "MLGWSCDataLoader": ("mlgwsc_data_loader", "MLGWSCDataLoader"),
    "create_mlgwsc_loader": ("mlgwsc_data_loader", "create_mlgwsc_loader"),
    
    # ✅ ReadLIGO data sources
    "LIGOEventData": ("gw_download", "LIGOEventData"),
    "LIGODataQuality": ("gw_download", "LIGODataQuality"),
    "ReadLIGODataFetcher": ("gw_download", "ReadLIGODataFetcher"),
    "LIGODataValidator": ("gw_download", "LIGODataValidator"),
    
    # ✅ Real LIGO data integration (migrated to gw_synthetic_generator)
    "download_gw150914_data": ("gw_synthetic_generator", "download_gw150914_data"),
    "create_proper_windows": ("gw_synthetic_generator", "create_proper_windows"),
    "create_real_ligo_dataset": ("gw_synthetic_generator", "create_real_ligo_dataset"),
    "create_simulated_gw150914_strain": ("gw_synthetic_generator", "create_simulated_gw150914_strain"),
    
    # ⚠️ DEPRECATED: GWOSC (with warnings)
    "GWOSCDownloader": ("gw_download", "GWOSCDownloader"),
    
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
    
    # Label utilities (from remaining files)
    "GWSignalType": ("label_enums", "GWSignalType"),
    "CANONICAL_LABELS": ("label_enums", "CANONICAL_LABELS"),
    "LABEL_NAMES": ("label_enums", "LABEL_NAMES"),
    "LABEL_DESCRIPTIONS": ("label_enums", "LABEL_DESCRIPTIONS"),
    "LABEL_COLORS": ("label_enums", "LABEL_COLORS"),
    
    # ✅ NEW: Modular preprocessing components
    "SegmentSampler": ("preprocessing.sampler", "SegmentSampler"),
    "AdvancedDataPreprocessor": ("preprocessing.core", "AdvancedDataPreprocessor"),
    "PreprocessingConfig": ("preprocessing.core", "PreprocessingConfig"),
    "preprocess_strain_data": ("preprocessing.utils", "preprocess_strain_data"),
    "validate_data_quality": ("preprocessing.utils", "validate_data_quality"),
    
    # ✅ NEW: Modular dataset builder components
    "GWDatasetBuilder": ("builders.core", "GWDatasetBuilder"),
    "create_mixed_gw_dataset": ("builders.factory", "create_mixed_gw_dataset"),
    "create_evaluation_dataset": ("builders.factory", "create_evaluation_dataset"),
    "test_dataset_builder": ("builders.testing", "test_dataset_builder"),
    
    # ✅ NEW: Modular cache components  
    "ProfessionalCacheManager": ("cache.manager", "ProfessionalCacheManager"),
    "CacheMetadata": ("cache.manager", "CacheMetadata"),
    "CacheStatistics": ("cache.manager", "CacheStatistics"),
    "get_cache_manager": ("cache.manager", "get_cache_manager"),
    "create_professional_cache": ("cache.manager", "create_professional_cache"),
    "cache_decorator": ("cache.operations", "cache_decorator"),
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
    
    # ✅ FIXED: ReadLIGO downloaders and preprocessing
    "ReadLIGODownloader",
    "LIGODownloader",
    "DataPreprocessor", 
    "AdvancedDataPreprocessor",
    "QualityMetrics",
    "ProcessingResult",
    "SegmentSampler",
    
    # ✅ ReadLIGO data sources
    "LIGOEventData",
    "LIGODataQuality", 
    "ReadLIGODataFetcher",
    "LIGODataValidator",
    
    # ✅ Real LIGO data integration
    "download_gw150914_data",
    "create_proper_windows", 
    "create_real_ligo_dataset",
    "create_simulated_gw150914_strain",
    
    # ⚠️ DEPRECATED: GWOSC (with warnings)
    "GWOSCDownloader",
    
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
    
    # Label utilities (remaining)
    "GWSignalType",
    "CANONICAL_LABELS",
    "LABEL_NAMES", 
    "LABEL_DESCRIPTIONS",
    "LABEL_COLORS",
    
    # ✅ NEW: Modular preprocessing components
    "SegmentSampler",
    "AdvancedDataPreprocessor",
    "PreprocessingConfig", 
    "preprocess_strain_data",
    "validate_data_quality",
    
    # ✅ NEW: Modular dataset builder components
    "GWDatasetBuilder",
    "create_mixed_gw_dataset",
    "create_evaluation_dataset", 
    "test_dataset_builder",
    
    # ✅ NEW: Modular cache components
    "ProfessionalCacheManager",
    "CacheMetadata",
    "CacheStatistics",
    "get_cache_manager",
    "create_professional_cache",
    "cache_decorator",
    
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


# ===== DEPRECATED COMPATIBILITY WRAPPERS =====
# For functions that were removed from deleted files

def create_real_ligo_dataset_deprecated(*args, **kwargs):
    """Deprecated wrapper for create_real_ligo_dataset. Use gw_synthetic_generator instead."""
    warnings.warn(
        "create_real_ligo_dataset from real_ligo_integration is deprecated. "
        "Use ContinuousGWGenerator.create_real_ligo_dataset instead.",
        DeprecationWarning,
        stacklevel=2
    )
    try:
        from .gw_synthetic_generator import ContinuousGWGenerator
        generator = ContinuousGWGenerator()
        return generator.create_real_ligo_dataset(*args, **kwargs)
    except Exception as e:
        logger.error(f"Fallback to gw_synthetic_generator failed: {e}")
        raise ImportError("real_ligo_integration module was removed. Use gw_synthetic_generator instead.")


def download_gw150914_data_deprecated(*args, **kwargs):
    """Deprecated wrapper for download_gw150914_data. Use gw_synthetic_generator instead."""
    warnings.warn(
        "download_gw150914_data from real_ligo_integration is deprecated. "
        "Use ContinuousGWGenerator.download_gw150914_data instead.",
        DeprecationWarning,
        stacklevel=2
    )
    try:
        from .gw_synthetic_generator import ContinuousGWGenerator
        generator = ContinuousGWGenerator()
        return generator.download_gw150914_data(*args, **kwargs)
    except Exception as e:
        logger.error(f"Fallback to gw_synthetic_generator failed: {e}")
        raise ImportError("real_ligo_integration module was removed. Use gw_synthetic_generator instead.") 