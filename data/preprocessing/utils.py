"""
Preprocessing utility functions.

This module contains standalone utility functions extracted from
gw_preprocessor.py for better modularity.

Split from gw_preprocessor.py for better maintainability.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def preprocess_strain_data(strain_data: jnp.ndarray,
                          sample_rate: int = 4096,
                          bandpass: Tuple[float, float] = (20.0, 1024.0),
                          apply_whitening: bool = True) -> jnp.ndarray:
    """
    Quick preprocessing utility for strain data.
    
    Simplified interface for common preprocessing operations.
    
    Args:
        strain_data: Raw strain data [time_samples]
        sample_rate: Sampling rate in Hz
        bandpass: Band-pass filter frequencies (low, high) in Hz
        apply_whitening: Whether to apply whitening
        
    Returns:
        Preprocessed strain data
    """
    from .core import AdvancedDataPreprocessor, PreprocessingConfig
    
    # Create preprocessor with provided config
    config = PreprocessingConfig(
        sample_rate=sample_rate,
        bandpass=bandpass,
        apply_whitening=apply_whitening
    )
    
    preprocessor = AdvancedDataPreprocessor(config)
    
    # Process (without quality metrics for simplicity)
    processed_strain = preprocessor.preprocess_strain(strain_data, return_quality=False)
    
    return processed_strain


def validate_data_quality(strain_data: jnp.ndarray,
                         quality_threshold: float = 0.7) -> Dict[str, any]:
    """
    Quick quality validation for strain data.
    
    Args:
        strain_data: Strain data to validate
        quality_threshold: Minimum quality threshold
        
    Returns:
        Dictionary with quality assessment
    """
    from .core import AdvancedDataPreprocessor, PreprocessingConfig
    
    # Create preprocessor for quality assessment
    config = PreprocessingConfig(quality_threshold=quality_threshold)
    preprocessor = AdvancedDataPreprocessor(config)
    
    # Get quality metrics
    _, quality_metrics = preprocessor.preprocess_strain(strain_data, return_quality=True)
    
    return {
        "is_good_quality": quality_metrics.is_good_quality,
        "quality_score": quality_metrics.quality_score,
        "snr_estimate": quality_metrics.snr_estimate,
        "spectral_coherence": quality_metrics.spectral_coherence,
        "glitch_score": quality_metrics.glitch_score
    }


def batch_preprocess_strains(batch_strain_data: jnp.ndarray,
                           config: Optional[Dict] = None) -> jnp.ndarray:
    """
    Batch preprocessing utility for multiple strain data.
    
    Args:
        batch_strain_data: Batch of strain data [batch_size, time_samples]
        config: Optional configuration dictionary
        
    Returns:
        Batch of preprocessed strain data
    """
    from .core import AdvancedDataPreprocessor, PreprocessingConfig
    
    # Create config from dict if provided
    if config is None:
        preprocessing_config = PreprocessingConfig()
    else:
        preprocessing_config = PreprocessingConfig(**config)
    
    preprocessor = AdvancedDataPreprocessor(preprocessing_config)
    
    # Process batch
    processed_batch, _ = preprocessor.process_batch(batch_strain_data)
    
    return processed_batch


def create_preprocessing_pipeline(config_dict: Optional[Dict] = None):
    """
    Factory function for creating preprocessing pipeline.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Configured AdvancedDataPreprocessor
    """
    from .core import AdvancedDataPreprocessor, PreprocessingConfig
    
    if config_dict is None:
        config = PreprocessingConfig()
    else:
        config = PreprocessingConfig(**config_dict)
    
    return AdvancedDataPreprocessor(config)


def get_default_preprocessing_config() -> Dict[str, any]:
    """Get default preprocessing configuration as dictionary."""
    from .core import PreprocessingConfig
    
    config = PreprocessingConfig()
    
    return {
        "sample_rate": config.sample_rate,
        "bandpass": config.bandpass,
        "apply_whitening": config.apply_whitening,
        "psd_length": config.psd_length,
        "quality_threshold": config.quality_threshold,
        "filter_order": config.filter_order
    }


# Export utility functions
__all__ = [
    "preprocess_strain_data",
    "validate_data_quality",
    "batch_preprocess_strains", 
    "create_preprocessing_pipeline",
    "get_default_preprocessing_config"
]

