"""
Advanced signal analysis module for gravitational wave detection.

This module provides professional signal analysis capabilities including:
- Matched filtering for optimal SNR estimation
- Template-based signal detection
- Advanced spectral analysis
- Quality assessment metrics
"""

from .snr_estimation import (
    ProfessionalSNREstimator,
    estimate_snr_matched_filter,
    estimate_snr_spectral,
    estimate_snr_template_bank
)

from .templates import (
    create_chirp_template,
    create_template_bank,
    TemplateParameters
)

__all__ = [
    'ProfessionalSNREstimator',
    'estimate_snr_matched_filter', 
    'estimate_snr_spectral',
    'estimate_snr_template_bank',
    'create_chirp_template',
    'create_template_bank',
    'TemplateParameters'
]

