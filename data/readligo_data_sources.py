"""
Lightweight stubs for ReadLIGO data source types used across preprocessing.

Provides dataclass definitions for QualityMetrics and ProcessingResult to avoid
import-time failures in modules that reference historical readligo_data_sources.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import jax.numpy as jnp


@dataclass
class QualityMetrics:
    # Align fields with gw_preprocessor.Assessors usage
    snr_estimate: float
    noise_floor: float
    glitch_detected: bool
    quality_score: float
    rms_noise: float
    kurtosis: float


@dataclass
class ProcessingResult:
    strain_data: jnp.ndarray
    psd: Optional[jnp.ndarray]
    quality: QualityMetrics
    processing_time: float
    metadata: Dict[str, Any]


__all__ = [
    "QualityMetrics",
    "ProcessingResult",
]


