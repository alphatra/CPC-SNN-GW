"""
Label Enumerations and Constants

Defines core enumerations, canonical labels, color schemes and constants
for the GW detection label system. Single source of truth for all
label-related constants across the CPC-SNN-GW project.

Features:
- GWSignalType enumeration with clear integer mappings
- Canonical label definitions and descriptions
- Multiple color schemes (default, scientific, colorblind-friendly)
- Backward compatibility support
"""

from typing import Dict, List
from enum import IntEnum
import numpy as np


class GWSignalType(IntEnum):
    """
    Gravitational wave signal type enumeration.
    
    Uses clear integer values for compatibility with neural networks.
    """
    NOISE = 0           # Pure noise / background
    CONTINUOUS_GW = 1   # Continuous gravitational waves
    BINARY_MERGER = 2   # Binary black hole / neutron star mergers
    

# Canonical label definitions - single source of truth
CANONICAL_LABELS = {
    GWSignalType.NOISE: "noise",
    GWSignalType.CONTINUOUS_GW: "continuous_gw", 
    GWSignalType.BINARY_MERGER: "binary_merger"
}

# Human-readable label names
LABEL_NAMES = {
    GWSignalType.NOISE: "Noise",
    GWSignalType.CONTINUOUS_GW: "Continuous GW",
    GWSignalType.BINARY_MERGER: "Binary Merger"
}

# Detailed descriptions for documentation
LABEL_DESCRIPTIONS = {
    GWSignalType.NOISE: "Background noise without gravitational wave signals",
    GWSignalType.CONTINUOUS_GW: "Continuous gravitational wave signals from rotating neutron stars",
    GWSignalType.BINARY_MERGER: "Transient signals from binary black hole or neutron star mergers"
}

# Default color scheme for visualizations
LABEL_COLORS = {
    GWSignalType.NOISE: "#2E86C1",          # Blue
    GWSignalType.CONTINUOUS_GW: "#28B463",   # Green  
    GWSignalType.BINARY_MERGER: "#E74C3C"    # Red
}

# Scientific publication color scheme (grayscale-friendly)
LABEL_COLORS_SCIENTIFIC = {
    GWSignalType.NOISE: "#1B2631",          # Dark gray
    GWSignalType.CONTINUOUS_GW: "#566573",   # Medium gray
    GWSignalType.BINARY_MERGER: "#D5D8DC"    # Light gray
}

# Colorblind-friendly scheme (Viridis-inspired)
LABEL_COLORS_COLORBLIND = {
    GWSignalType.NOISE: "#440154",          # Purple
    GWSignalType.CONTINUOUS_GW: "#31688E",   # Teal
    GWSignalType.BINARY_MERGER: "#FDE725"    # Yellow
}

# Available color schemes
COLOR_SCHEMES = {
    'default': LABEL_COLORS,
    'scientific': LABEL_COLORS_SCIENTIFIC,
    'colorblind': LABEL_COLORS_COLORBLIND
}

# Legacy label mappings for backward compatibility
LEGACY_LABEL_MAPPINGS = {
    # Old string labels to new enum values
    "background": GWSignalType.NOISE,
    "noise_only": GWSignalType.NOISE,
    "no_signal": GWSignalType.NOISE,
    "continuous": GWSignalType.CONTINUOUS_GW,
    "cw": GWSignalType.CONTINUOUS_GW,
    "pulsar": GWSignalType.CONTINUOUS_GW,
    "merger": GWSignalType.BINARY_MERGER,
    "bbh": GWSignalType.BINARY_MERGER,
    "bns": GWSignalType.BINARY_MERGER,
    "transient": GWSignalType.BINARY_MERGER,
    
    # Numeric mappings
    0: GWSignalType.NOISE,
    1: GWSignalType.CONTINUOUS_GW,
    2: GWSignalType.BINARY_MERGER
}

# Validation constants
MAX_LABEL_VALUE = max(GWSignalType)
MIN_LABEL_VALUE = min(GWSignalType)
NUM_CLASSES = len(GWSignalType)
VALID_LABEL_VALUES = list(GWSignalType)

# Default parameters for various operations
DEFAULT_FUZZY_THRESHOLD = 0.8
DEFAULT_ML_THRESHOLD = 0.7
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

# ML model configurations
ML_MODEL_CONFIGS = {
    "sentence_transformer": {
        "model_name": "all-MiniLM-L6-v2",
        "cache_size": 1000,
        "similarity_threshold": 0.7
    },
    "sklearn": {
        "metric": "cosine",
        "cache_size": 500
    }
}


def get_cmap_colors(scheme: str = 'default') -> List[str]:
    """
    Get color list for matplotlib colormap creation.
    
    Args:
        scheme: Color scheme name ('default', 'scientific', 'colorblind')
        
    Returns:
        List of hex color codes
    """
    if scheme not in COLOR_SCHEMES:
        raise ValueError(f"Unknown color scheme: {scheme}. Available: {list(COLOR_SCHEMES.keys())}")
    
    colors = COLOR_SCHEMES[scheme]
    return [colors[label] for label in sorted(GWSignalType)]


def get_label_info(label_value: int) -> Dict[str, str]:
    """
    Get comprehensive information about a label.
    
    Args:
        label_value: Integer label value
        
    Returns:
        Dictionary with name, canonical form, description, and color
    """
    if label_value not in GWSignalType:
        raise ValueError(f"Invalid label value: {label_value}")
    
    signal_type = GWSignalType(label_value)
    
    return {
        'canonical': CANONICAL_LABELS[signal_type],
        'name': LABEL_NAMES[signal_type],
        'description': LABEL_DESCRIPTIONS[signal_type],
        'color': LABEL_COLORS[signal_type],
        'enum_value': int(signal_type)
    }


def is_valid_label(label_value) -> bool:
    """
    Check if a label value is valid.
    
    Args:
        label_value: Value to check
        
    Returns:
        True if valid, False otherwise
    """
    try:
        return int(label_value) in GWSignalType
    except (ValueError, TypeError):
        return False


def normalize_label_value(label) -> int:
    """
    Normalize various label formats to integer enum values.
    
    Args:
        label: Label in various formats (string, int, enum)
        
    Returns:
        Normalized integer label value
        
    Raises:
        ValueError: If label cannot be normalized
    """
    # Handle enum instances
    if isinstance(label, GWSignalType):
        return int(label)
    
    # Handle direct integer values
    if isinstance(label, (int, np.integer)):
        if label in GWSignalType:
            return int(label)
        else:
            raise ValueError(f"Invalid integer label: {label}")
    
    # Handle string labels
    if isinstance(label, str):
        label_lower = label.lower().strip()
        
        # Check canonical labels
        for signal_type, canonical in CANONICAL_LABELS.items():
            if label_lower == canonical:
                return int(signal_type)
        
        # Check legacy mappings
        if label_lower in LEGACY_LABEL_MAPPINGS:
            return int(LEGACY_LABEL_MAPPINGS[label_lower])
        
        raise ValueError(f"Unknown string label: {label}")
    
    raise ValueError(f"Cannot normalize label of type {type(label)}: {label}")


# Export all important symbols
__all__ = [
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
    'get_cmap_colors',
    'get_label_info',
    'is_valid_label',
    'normalize_label_value'
] 