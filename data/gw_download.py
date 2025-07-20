"""
GWOSC Data Download Pipeline

Unified interface for gravitational wave data downloading and preprocessing.
This module provides backward compatibility while delegating functionality
to specialized modules.

Refactored into:
- gw_data_sources.py: Abstract data sources and quality metrics
- gw_downloader.py: GWOSC downloader implementation  
- gw_preprocessor.py: Data preprocessing and segment sampling
"""

# Import core functionality from specialized modules
from .gw_data_sources import (
    GWDataSource,
    QualityMetrics,
    ProcessingResult,
    SmartSegmentSampler,
    validate_strain_data
)

from .gw_downloader import (
    ProductionGWOSCDownloader,
    _safe_jax_cpu_context,
    _compute_kurtosis,
    _safe_array_to_jax,
    _generate_cache_key
)

from .gw_preprocessor import (
    AdvancedDataPreprocessor,
    SegmentSampler
)

# Legacy aliases for backward compatibility
GWOSCDownloader = ProductionGWOSCDownloader
DataPreprocessor = AdvancedDataPreprocessor

__all__ = [
    # Core classes
    'ProductionGWOSCDownloader',
    'AdvancedDataPreprocessor', 
    'SegmentSampler',
    
    # Data sources
    'GWDataSource',
    'QualityMetrics',
    'ProcessingResult',
    'SmartSegmentSampler',
    
    # Utility functions
    'validate_strain_data',
    '_safe_jax_cpu_context',
    '_compute_kurtosis',
    '_safe_array_to_jax',
    '_generate_cache_key',
    
    # Legacy aliases
    'GWOSCDownloader',
    'DataPreprocessor'
] 