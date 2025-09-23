"""
Unified filtering implementations for CPC-SNN-GW system.

This module consolidates all filtering functionality to eliminate redundancy
between data/preprocessing/core.py and cli/runners/standard.py

UNIFIED METHODS:
- design_windowed_sinc_bandpass: Professional bandpass filter design
- design_windowed_sinc_lowpass: Anti-aliasing lowpass filter design  
- antialias_downsample: Complete downsampling with anti-aliasing
- apply_bandpass_filter: Apply bandpass filtering to signals
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Union
import logging

logger = logging.getLogger(__name__)


def design_windowed_sinc_bandpass(
    low_freq: float, 
    high_freq: float, 
    order: int = 8,
    min_taps: int = 129
) -> jnp.ndarray:
    """
    ✅ UNIFIED: Professional windowed-sinc bandpass filter design
    
    This is the single source of truth for bandpass filtering across the system.
    Replaces redundant implementations in core.py and standard.py.
    
    Args:
        low_freq: Low cutoff frequency (normalized 0-1)
        high_freq: High cutoff frequency (normalized 0-1)
        order: Filter order (affects length)
        min_taps: Minimum number of filter taps
        
    Returns:
        Filter coefficients as JAX array
    """
    # ✅ ADAPTIVE FILTER LENGTH: Ensures adequate selectivity
    taps = int(max(min_taps, 16 * order + 1))
    
    # ✅ WINDOWED-SINC DESIGN: Proven method
    n = jnp.arange(taps)
    m = (taps - 1) / 2.0
    
    # Design as difference of two lowpass filters
    h_high = 2.0 * high_freq * jnp.sinc(2.0 * high_freq * (n - m))
    h_low = 2.0 * low_freq * jnp.sinc(2.0 * low_freq * (n - m))
    h_bandpass = h_high - h_low
    
    # ✅ HANN WINDOW: Optimal spectral characteristics
    hann_window = 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * n / (taps - 1))
    h_windowed = h_bandpass * hann_window
    
    # ✅ UNITY GAIN NORMALIZATION: Preserve signal amplitude
    h_normalized = h_windowed / (jnp.sum(jnp.abs(h_windowed)) + 1e-8)
    
    logger.debug(f"✅ Unified {taps}-tap bandpass filter designed")
    logger.debug(f"   Range: [{low_freq:.3f}, {high_freq:.3f}] (normalized)")
    
    return h_normalized.astype(jnp.float32)


def design_windowed_sinc_lowpass(
    cutoff_freq: float,
    decim_factor: int = 1, 
    taps: int = 97
) -> jnp.ndarray:
    """
    ✅ UNIFIED: Professional windowed-sinc lowpass filter design
    
    Used for anti-aliasing in downsampling operations.
    Replaces _design_lowpass_kernel from cli/runners/standard.py
    
    Args:
        cutoff_freq: Cutoff frequency (normalized 0-1)
        decim_factor: Decimation factor for anti-aliasing
        taps: Number of filter taps
        
    Returns:
        Filter coefficients as JAX array
    """
    # ✅ ANTI-ALIASING CUTOFF: Prevent aliasing in decimation
    if decim_factor > 1:
        fc = 0.5 / float(max(1, decim_factor))
    else:
        fc = cutoff_freq
        
    # ✅ WINDOWED-SINC LOWPASS DESIGN
    n = jnp.arange(taps)
    m = (taps - 1) / 2.0
    
    # Sinc lowpass kernel
    h = 2.0 * fc * jnp.sinc(2.0 * fc * (n - m))
    
    # ✅ HANN WINDOW: Reduce spectral leakage
    hann_window = 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * n / (taps - 1))
    h_windowed = h * hann_window
    
    # ✅ UNITY GAIN NORMALIZATION: DC gain = 1
    h_normalized = h_windowed / (jnp.sum(h_windowed) + 1e-8)
    
    logger.debug(f"✅ Unified {taps}-tap lowpass filter designed (fc={fc:.3f})")
    
    return h_normalized


def antialias_downsample(
    x: jnp.ndarray, 
    target_length: int = 512,
    max_taps: int = 97
) -> jnp.ndarray:
    """
    ✅ UNIFIED: Professional anti-aliasing downsampling
    
    Replaces _antialias_downsample from cli/runners/standard.py
    with consistent implementation across the system.
    
    Args:
        x: Input signal [batch, time, features]
        target_length: Target length after downsampling
        max_taps: Maximum filter taps for anti-aliasing
        
    Returns:
        Downsampled signal with same shape but shorter time dimension
    """
    current_length = x.shape[1]
    
    # No downsampling needed
    if current_length <= target_length:
        return x
    
    # Calculate decimation factor
    decim_factor = int(jnp.ceil(current_length / target_length))
    
    # ✅ ADAPTIVE FILTER LENGTH: Based on decimation factor
    taps = int(min(max_taps, max(31, 6 * decim_factor + 1)))
    
    # ✅ DESIGN ANTI-ALIASING FILTER
    kernel = design_windowed_sinc_lowpass(
        cutoff_freq=0.5,  # Will be adjusted for decimation
        decim_factor=decim_factor,
        taps=taps
    )
    
    # ✅ APPLY FILTERING WITH REFLECTION PADDING
    pad = taps // 2
    x_padded = jnp.pad(x, ((0, 0), (pad, pad), (0, 0)), mode='reflect')
    
    # ✅ CONVOLUTION PER FEATURE (efficient implementation)
    def conv_feature(feature_idx):
        x_feat = x_padded[:, :, feature_idx]
        y_feat = jax.vmap(lambda row: jnp.convolve(row, kernel, mode='valid'))(x_feat)
        return y_feat
    
    # Process all features
    num_features = x.shape[-1]
    filtered_features = [conv_feature(f) for f in range(num_features)]
    y_filtered = jnp.stack(filtered_features, axis=-1)
    
    # ✅ DECIMATION: Keep every decim_factor-th sample
    y_decimated = y_filtered[:, ::decim_factor, :]
    
    # ✅ ENSURE TARGET LENGTH: Pad or truncate as needed
    actual_length = y_decimated.shape[1]
    if actual_length < target_length:
        pad_needed = target_length - actual_length
        y_decimated = jnp.pad(y_decimated, ((0, 0), (0, pad_needed), (0, 0)), mode='edge')
    elif actual_length > target_length:
        y_decimated = y_decimated[:, :target_length, :]
    
    logger.debug(f"✅ Unified downsampling: {current_length} → {y_decimated.shape[1]} (factor={decim_factor})")
    
    return y_decimated


def apply_bandpass_filter(
    signal: jnp.ndarray,
    filter_coeffs: jnp.ndarray,
    mode: str = 'same'
) -> jnp.ndarray:
    """
    ✅ UNIFIED: Apply bandpass filter to signal
    
    Consistent filtering application across the system.
    
    Args:
        signal: Input signal to filter
        filter_coeffs: Filter coefficients from design_windowed_sinc_bandpass
        mode: Convolution mode ('same', 'valid', 'full')
        
    Returns:
        Filtered signal
    """
    # Handle different input shapes
    original_shape = signal.shape
    
    if signal.ndim == 1:
        # Single signal
        filtered = jnp.convolve(signal, filter_coeffs, mode=mode)
    elif signal.ndim == 2:
        # Batch of signals [batch, time] or [time, features]
        if original_shape[0] < original_shape[1]:
            # Assume [batch, time] - filter each signal
            filtered = jax.vmap(lambda s: jnp.convolve(s, filter_coeffs, mode=mode))(signal)
        else:
            # Assume [time, features] - filter along time axis
            filtered = jax.vmap(lambda s: jnp.convolve(s, filter_coeffs, mode=mode), in_axes=1, out_axes=1)(signal)
    elif signal.ndim == 3:
        # [batch, time, features] - filter each signal and feature
        def filter_one_sample(sample):
            return jax.vmap(lambda s: jnp.convolve(s, filter_coeffs, mode=mode), in_axes=1, out_axes=1)(sample)
        filtered = jax.vmap(filter_one_sample)(signal)
    else:
        raise ValueError(f"Unsupported signal shape: {original_shape}")
    
    logger.debug(f"✅ Unified filtering applied: {original_shape} → {filtered.shape}")
    
    return filtered


# ✅ CONVENIENCE FUNCTIONS for backward compatibility

def create_butterworth_filter(sample_rate: int, low_freq: float, high_freq: float, order: int = 8) -> jnp.ndarray:
    """
    ✅ CONVENIENCE: Create Butterworth-style bandpass filter
    
    Provides backward compatibility for existing code.
    """
    # Normalize frequencies
    nyquist = sample_rate / 2.0
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist
    
    # Clip to valid range
    low_norm = jnp.clip(low_norm, 0.001, 0.999)
    high_norm = jnp.clip(high_norm, 0.001, 0.999)
    
    return design_windowed_sinc_bandpass(low_norm, high_norm, order)


def create_antialias_filter(decim_factor: int, taps: int = 97) -> jnp.ndarray:
    """
    ✅ CONVENIENCE: Create anti-aliasing filter for downsampling
    
    Provides backward compatibility for existing code.
    """
    return design_windowed_sinc_lowpass(0.5, decim_factor, taps)

