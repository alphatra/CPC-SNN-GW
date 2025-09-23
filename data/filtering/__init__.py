"""
Unified filtering module for CPC-SNN-GW system.

This module provides consistent filtering implementations across the entire system,
eliminating redundancy between data/preprocessing/core.py and cli/runners/standard.py
"""

from .unified import (
    design_windowed_sinc_bandpass,
    design_windowed_sinc_lowpass,
    antialias_downsample,
    apply_bandpass_filter
)

__all__ = [
    'design_windowed_sinc_bandpass',
    'design_windowed_sinc_lowpass', 
    'antialias_downsample',
    'apply_bandpass_filter'
]

