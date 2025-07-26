"""
✅ FIXED ReadLIGO Data Download Pipeline

Unified interface for gravitational wave data downloading and preprocessing.
REPLACED problematic GWOSC API with working ReadLIGO solution.

Updated modules:
- readligo_data_sources.py: ReadLIGO data sources and quality metrics ✅
- readligo_downloader.py: ReadLIGO downloader implementation ✅
- gw_preprocessor.py: Data preprocessing and segment sampling ✅
"""

# ✅ FIXED: Import ReadLIGO functionality instead of GWOSC
from .readligo_data_sources import (
    LIGOEventData,
    QualityMetrics,
    ProcessingResult,
    LIGODataQuality,
    ReadLIGODataFetcher,
    LIGODataValidator,
    create_readligo_fetcher,
    create_ligo_validator
)

from .readligo_downloader import (
    ReadLIGODownloader,
    _safe_jax_cpu_context,
    _compute_kurtosis,
    _safe_array_to_jax,
    _generate_cache_key,
    create_readligo_downloader
)

from .gw_preprocessor import (
    AdvancedDataPreprocessor,
    SegmentSampler
)

# ✅ FIXED: Legacy aliases now point to ReadLIGO
LIGODownloader = ReadLIGODownloader
DataPreprocessor = AdvancedDataPreprocessor

# ⚠️ DEPRECATED but kept for backward compatibility
# These will issue warnings when used
import warnings

class _DeprecatedGWOSCDownloader:
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "GWOSCDownloader is deprecated and replaced with ReadLIGODownloader. "
            "Please update your code to use ReadLIGODownloader for reliable LIGO data access.",
            DeprecationWarning,
            stacklevel=2
        )
        self._readligo = ReadLIGODownloader(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self._readligo, name)

GWOSCDownloader = _DeprecatedGWOSCDownloader

__all__ = [
    # ✅ FIXED: Core ReadLIGO classes
    'ReadLIGODownloader',
    'LIGODownloader', 
    'AdvancedDataPreprocessor', 
    'SegmentSampler',
    
    # ✅ FIXED: ReadLIGO data sources
    'LIGOEventData',
    'QualityMetrics',
    'ProcessingResult',
    'LIGODataQuality',
    'ReadLIGODataFetcher',
    'LIGODataValidator',
    
    # ✅ FIXED: Factory functions
    'create_readligo_fetcher',
    'create_ligo_validator',
    'create_readligo_downloader',
    
    # Utility functions
    '_safe_jax_cpu_context',
    '_compute_kurtosis',
    '_safe_array_to_jax',
    '_generate_cache_key',
    
    # ⚠️ DEPRECATED: Legacy aliases (with warnings)
    'GWOSCDownloader',
    'DataPreprocessor'
] 