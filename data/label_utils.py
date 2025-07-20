"""
Enhanced Label Utilities for GW Detection - Unified Interface

Refactored label utilities system providing backward compatibility while 
delegating functionality to specialized modules for better maintainability.

This module serves as the main entry point and provides backward compatibility
for existing code while the actual implementation is distributed across:

- label_enums.py: Core enumerations, constants, and color schemes
- label_validation.py: Validation functions and error handling
- label_correction.py: Auto-correction algorithms (fuzzy, ML)
- label_analytics.py: Statistics, visualization, and reporting

All original functionality is preserved through imports and legacy aliases.
"""

# Import core enumerations and constants
from .label_enums import (
    GWSignalType,
    CANONICAL_LABELS,
    LABEL_NAMES,
    LABEL_DESCRIPTIONS,
    LABEL_COLORS,
    LABEL_COLORS_SCIENTIFIC,
    LABEL_COLORS_COLORBLIND,
    COLOR_SCHEMES,
    LEGACY_LABEL_MAPPINGS,
    MAX_LABEL_VALUE,
    MIN_LABEL_VALUE,
    NUM_CLASSES,
    VALID_LABEL_VALUES,
    DEFAULT_FUZZY_THRESHOLD,
    DEFAULT_ML_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    ML_MODEL_CONFIGS,
    get_cmap_colors,
    get_label_info,
    is_valid_label,
    normalize_label_value
)

# Import validation functionality
from .label_validation import (
    LabelError,
    LabelValidationResult,
    validate_labels,
    validate_dataset_labels,
    convert_legacy_labels,
    normalize_labels,
    _fast_validity_check,
    _fast_class_distribution
)

# Import correction algorithms
from .label_correction import (
    fuzzy_match_labels,
    ml_embedding_similarity,
    auto_correct_labels,
    ml_auto_correct_labels,
    smart_label_correction,
    benchmark_correction_methods,
    _cached_similarity_score,
    _cached_ml_embedding_similarity,
    _cached_canonical_embeddings
)

# Import analytics and statistics
from .label_analytics import (
    DatasetStatistics,
    get_class_weights,
    calculate_dataset_statistics,
    create_dataset_summary,
    log_dataset_info,
    create_label_visualization_config,
    dataset_to_canonical,
    create_label_report
)

# Legacy function aliases for backward compatibility
def log_dataset_info_legacy(labels, dataset_name="Dataset"):
    """Legacy wrapper for log_dataset_info function."""
    log_dataset_info(labels, dataset_name)

def create_dataset_summary_legacy(labels, dataset_name="Dataset"):
    """Legacy wrapper for create_dataset_summary function."""
    return create_dataset_summary(labels, dataset_name, include_details=True)

# Export all symbols for backward compatibility
__all__ = [
    # Core enumerations and constants
    'GWSignalType',
    'CANONICAL_LABELS',
    'LABEL_NAMES',
    'LABEL_DESCRIPTIONS', 
    'LABEL_COLORS',
    'LABEL_COLORS_SCIENTIFIC',
    'LABEL_COLORS_COLORBLIND',
    'COLOR_SCHEMES',
    'LEGACY_LABEL_MAPPINGS',
    'MAX_LABEL_VALUE',
    'MIN_LABEL_VALUE',
    'NUM_CLASSES',
    'VALID_LABEL_VALUES',
    'DEFAULT_FUZZY_THRESHOLD',
    'DEFAULT_ML_THRESHOLD',
    'DEFAULT_CONFIDENCE_THRESHOLD',
    'ML_MODEL_CONFIGS',
    
    # Validation classes and functions
    'LabelError',
    'LabelValidationResult',
    'validate_labels',
    'validate_dataset_labels',
    'convert_legacy_labels',
    'normalize_labels',
    
    # Correction algorithms
    'fuzzy_match_labels',
    'ml_embedding_similarity',
    'auto_correct_labels',
    'ml_auto_correct_labels',
    'smart_label_correction',
    'benchmark_correction_methods',
    
    # Analytics and statistics
    'DatasetStatistics',
    'get_class_weights',
    'calculate_dataset_statistics',
    'create_dataset_summary',
    'log_dataset_info',
    'create_label_visualization_config',
    'dataset_to_canonical',
    'create_label_report',
    
    # Utility functions
    'get_cmap_colors',
    'get_label_info', 
    'is_valid_label',
    'normalize_label_value',
    
    # Legacy aliases
    'log_dataset_info_legacy',
    'create_dataset_summary_legacy',
    
    # Internal functions (for advanced users)
    '_fast_validity_check',
    '_fast_class_distribution',
    '_cached_similarity_score',
    '_cached_ml_embedding_similarity',
    '_cached_canonical_embeddings'
]

# Module metadata
__version__ = "2.0.0"
__author__ = "LIGO CPC+SNN Team"
__description__ = "Enhanced label utilities for gravitational wave detection" 