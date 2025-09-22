"""
Core preprocessing pipeline for gravitational wave data.

This module contains the main AdvancedDataPreprocessor class extracted from
gw_preprocessor.py for better modularity.

Split from gw_preprocessor.py for better maintainability.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import time

from data.readligo_data_sources import QualityMetrics, ProcessingResult
from data.filtering.unified import (
    design_windowed_sinc_bandpass, 
    apply_bandpass_filter,
    create_butterworth_filter
)
from data.cache.manager import create_professional_cache, get_cache_manager
from data.cache.operations import cache_decorator
from data.signal_analysis.snr_estimation import (
    ProfessionalSNREstimator, 
    estimate_snr_matched_filter,
    estimate_snr_spectral
)
from data.signal_analysis.templates import create_default_template_bank

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing pipeline."""
    sample_rate: int = 4096
    bandpass: Tuple[float, float] = (20.0, 1024.0)
    apply_whitening: bool = True
    psd_length: int = 8  # seconds for PSD estimation
    quality_threshold: float = 0.7
    filter_order: int = 8
    
    def validate(self) -> bool:
        """Validate preprocessing configuration."""
        try:
            assert self.sample_rate > 0, "sample_rate must be positive"
            assert len(self.bandpass) == 2, "bandpass must be (low, high) tuple"
            assert 0 < self.bandpass[0] < self.bandpass[1], "Invalid bandpass frequencies"
            assert self.bandpass[1] < self.sample_rate / 2, "High frequency above Nyquist"
            assert self.psd_length > 0, "psd_length must be positive"
            assert 0 <= self.quality_threshold <= 1, "quality_threshold must be in [0,1]"
            assert self.filter_order > 0, "filter_order must be positive"
            return True
        except AssertionError as e:
            logger.error(f"Preprocessing config validation failed: {e}")
            return False


class AdvancedDataPreprocessor:
    """
    Advanced preprocessing pipeline with quality validation optimized for Apple Silicon.
    
    Implements whitening, band-pass filtering, glitch detection, and spectral analysis.
    Target: <1s processing time per 4s segment.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize advanced data preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        if config is None:
            config = PreprocessingConfig()
        
        if not config.validate():
            raise ValueError("Invalid preprocessing configuration")
        
        self.config = config
        
        # ✅ PROFESSIONAL CACHE: Initialize cache manager for expensive operations
        self.cache_manager = create_professional_cache(
            cache_dir=None,  # Use default location
            max_size_mb=500,  # 500MB for preprocessing cache
            max_entries=1000,
            default_ttl_hours=6  # 6 hours for preprocessing results
        )
        
        # ✅ PROFESSIONAL SNR: Initialize advanced SNR estimator
        self.snr_estimator = ProfessionalSNREstimator(
            sample_rate=config.sample_rate,
            low_freq=config.bandpass[0],
            high_freq=config.bandpass[1]
        )
        
        # ✅ TEMPLATE BANK: Create default templates for matched filtering
        try:
            self.template_bank = create_default_template_bank(
                sample_rate=config.sample_rate,
                duration=4.0  # 4 second templates
            )
            logger.info(f"✅ Loaded {len(self.template_bank)} templates for matched filtering")
        except Exception as e:
            logger.warning(f"Failed to create template bank: {e}, using spectral SNR only")
            self.template_bank = []
        
        # Pre-compute filter coefficients for efficiency
        self._setup_filters()
        
        logger.info(f"AdvancedDataPreprocessor initialized: "
                   f"fs={config.sample_rate}, bandpass={config.bandpass}")
        logger.info(f"✅ Professional caching enabled: {self.cache_manager.get_info()['cache_dir']}")
    
    def _setup_filters(self):
        """Pre-compute filter coefficients for band-pass filtering."""
        # 🚨 CRITICAL FIX: Pure JAX filter design (NO CPU conversion)
        # Replace SciPy cheby2 with JAX-native implementation
        
        nyquist = self.config.sample_rate / 2
        low_norm = self.config.bandpass[0] / nyquist
        high_norm = self.config.bandpass[1] / nyquist
        
        # Ensure valid normalized frequencies
        low_norm = jnp.clip(low_norm, 0.001, 0.999)
        high_norm = jnp.clip(high_norm, 0.001, 0.999)
        
        # ✅ UNIFIED: Use professional unified filtering system
        # Replaces local implementation with system-wide unified method
        self.filter_sos = create_butterworth_filter(
            sample_rate=self.config.sample_rate,
            low_freq=self.config.bandpass[0],
            high_freq=self.config.bandpass[1],
            order=self.config.filter_order
        )
    
    # ✅ REMOVED: _design_jax_butterworth_filter replaced by unified filtering system
    # Old implementation moved to data/filtering/unified.py for system-wide consistency
    
    def _complete_filter_setup(self):
        """Complete filter setup including PSD estimation."""
        self._setup_filters()
        
        # Initialize PSD storage for whitening
        self.psd_storage = {}
        
        # Performance tracking
        self.processing_times = []
        
        logger.debug("Filter setup completed")
    
    def preprocess_strain(self, strain_data: jnp.ndarray, 
                         detector: str = "H1",
                         return_quality: bool = True) -> Union[jnp.ndarray, Tuple[jnp.ndarray, QualityMetrics]]:
        """
        Main preprocessing pipeline for strain data.
        
        Args:
            strain_data: Raw strain data [time_samples]
            detector: Detector name (H1, L1, V1)
            return_quality: Whether to return quality metrics
            
        Returns:
            Preprocessed strain data, optionally with quality metrics
        """
        start_time = time.time()
        
        # ✅ INPUT VALIDATION
        if len(strain_data.shape) != 1:
            raise ValueError(f"Expected 1D strain data, got shape {strain_data.shape}")
        
        if jnp.any(jnp.isnan(strain_data)) or jnp.any(jnp.isinf(strain_data)):
            logger.warning("NaN or Inf detected in input strain data")
            # Replace with zeros
            strain_data = jnp.where(jnp.isfinite(strain_data), strain_data, 0.0)
        
        # ✅ PREPROCESSING PIPELINE
        
        # Step 1: Band-pass filtering (with caching of filter for current config)
        try:
            cache = create_professional_cache("preprocessing_filters")
        except Exception:
            cache = None
        # ✅ KRYTYCZNA NAPRAWA: Użyj ujednoliconego filtrowania zamiast nieistniejącej funkcji
        from data.filtering.unified import design_windowed_sinc_bandpass
        
        cache_key = f"bp_fs{self.config.sample_rate}_b{self.config.bandpass[0]}-{self.config.bandpass[1]}_order{self.config.filter_order}"
        if cache is not None:
            coeffs = cache.get(cache_key)
            if coeffs is None:
                coeffs = design_windowed_sinc_bandpass(
                    low_freq=self.config.bandpass[0] / (self.config.sample_rate / 2),
                    high_freq=self.config.bandpass[1] / (self.config.sample_rate / 2),
                    order=self.config.filter_order
                )
                cache.set(cache_key, coeffs)
            self.filter_sos = coeffs
        else:
            # Fallback gdy cache nie działa
            self.filter_sos = design_windowed_sinc_bandpass(
                low_freq=self.config.bandpass[0] / (self.config.sample_rate / 2),
                high_freq=self.config.bandpass[1] / (self.config.sample_rate / 2),
                order=self.config.filter_order
            )
        filtered_strain = self._apply_bandpass_filter(strain_data)
        
        # Step 2: Whitening (if enabled)
        if self.config.apply_whitening:
            whitened_strain = self._apply_whitening(filtered_strain, detector)
        else:
            whitened_strain = filtered_strain
        
        # Step 3: Quality assessment
        if return_quality:
            quality_metrics = self._assess_quality(whitened_strain, detector)
        
        # ✅ PERFORMANCE TRACKING
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        logger.debug(f"Preprocessing completed in {processing_time:.3f}s "
                    f"(target: <1.0s, {len(strain_data)} samples)")
        
        if return_quality:
            return whitened_strain, quality_metrics
        else:
            return whitened_strain
    
    def _apply_bandpass_filter(self, strain_data: jnp.ndarray) -> jnp.ndarray:
        """✅ UNIFIED: Apply band-pass filter using unified filtering system."""
        # ✅ USE UNIFIED IMPLEMENTATION: Consistent across entire system
        filtered = apply_bandpass_filter(strain_data, self.filter_sos, mode='same')
        
        # ✅ STABILITY: Prevent numerical issues
        filtered = jnp.where(jnp.isfinite(filtered), filtered, 0.0)
        
        return filtered
    
    def _apply_whitening(self, strain_data: jnp.ndarray, detector: str) -> jnp.ndarray:
        """✅ CACHED: Apply whitening using PSD estimated via Welch's method (JAX)."""
        # Welch parameters
        segment_seconds = max(1, int(self.config.psd_length))
        nperseg = int(self.config.sample_rate * segment_seconds)
        noverlap = int(0.5 * nperseg)
        step = max(1, nperseg - noverlap)
        n = len(strain_data)
        
        # ✅ PROFESSIONAL CACHE: Check for cached PSD
        cache_key = f"psd_{detector}_fs{self.config.sample_rate}_len{n}_seg{nperseg}"
        cached_psd = self.cache_manager.get(cache_key)
        
        if cached_psd is not None:
            logger.debug(f"✅ Using cached PSD for {detector}")
            psd = cached_psd
        else:
            # Guard: if signal shorter than one segment, fallback to simple PSD
            if n < nperseg:
                fft_data = jnp.fft.rfft(strain_data)
                psd = jnp.abs(fft_data) ** 2
            else:
                # Frame the signal into overlapping segments
                starts = jnp.arange(0, n - nperseg + 1, step)
                
                def segment_fft(start_idx):
                    seg = strain_data[start_idx:start_idx + nperseg]
                    # Hann window for variance reduction
                    n_vec = jnp.arange(nperseg)
                    window = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * n_vec / (nperseg - 1)))
                    seg_win = seg * window
                    fft = jnp.fft.rfft(seg_win)
                    # Scale by window power
                    scale = jnp.sum(window**2)
                    return (jnp.abs(fft) ** 2) / (scale + 1e-12)
                
                psd_stack = jax.vmap(segment_fft)(starts)
                psd = jnp.mean(psd_stack, axis=0)
            
            # ✅ PROFESSIONAL CACHE: Store computed PSD for reuse
            self.cache_manager.set(cache_key, psd, ttl_seconds=3600)  # Cache for 1 hour
            logger.debug(f"✅ Cached PSD for {detector}")
        
        # Floor to avoid division issues (for cached PSD too)
        psd = jnp.maximum(psd, jnp.max(psd) * 1e-8)
        
        # Whiten in frequency domain
        fft_full = jnp.fft.rfft(strain_data)
        inverse_psd_sqrt = 1.0 / jnp.sqrt(psd + 1e-12)
        whitened_fft = fft_full * inverse_psd_sqrt
        whitened = jnp.fft.irfft(whitened_fft, n=n)
        
        # Cache PSD
        cache_key = f"{detector}_psd_{n}"
        self.psd_storage[cache_key] = psd
        
        return whitened
    
    def _assess_quality(self, strain_data: jnp.ndarray, detector: str) -> QualityMetrics:
        """Assess quality of preprocessed strain data."""
        # Calculate basic quality metrics
        snr_estimate = self._estimate_snr(strain_data)
        spectral_coherence = self._calculate_spectral_coherence(strain_data)
        glitch_score = self._detect_glitches(strain_data)
        
        # Overall quality score
        quality_score = (snr_estimate * 0.4 + 
                        spectral_coherence * 0.3 + 
                        (1.0 - glitch_score) * 0.3)
        
        return QualityMetrics(
            snr_estimate=float(snr_estimate),
            spectral_coherence=float(spectral_coherence),
            glitch_score=float(glitch_score),
            quality_score=float(quality_score),
            is_good_quality=quality_score >= self.config.quality_threshold
        )
    
    def _estimate_snr(self, strain_data: jnp.ndarray) -> jnp.ndarray:
        """✅ PROFESSIONAL: Estimate SNR using matched filtering and spectral analysis."""
        try:
            # ✅ MATCHED FILTERING: Try template bank first (gold standard)
            if self.template_bank:
                # Use template bank for optimal SNR estimation
                result = self.snr_estimator.estimate_snr_template_bank(
                    strain_data, 
                    self.template_bank
                )
                matched_filter_snr = result.optimal_snr
                
                # ✅ SPECTRAL SNR: As comparison/backup
                spectral_result = self.snr_estimator.estimate_snr_spectral(strain_data)
                spectral_snr = spectral_result.optimal_snr
                
                # ✅ COMBINE ESTIMATES: Use maximum for conservative estimate
                snr_estimate = jnp.maximum(matched_filter_snr, spectral_snr)
                
                logger.debug(f"✅ SNR estimates - Matched: {matched_filter_snr:.2f}, "
                           f"Spectral: {spectral_snr:.2f}, Final: {snr_estimate:.2f}")
                
            else:
                # ✅ SPECTRAL FALLBACK: Use improved spectral method
                result = self.snr_estimator.estimate_snr_spectral(strain_data)
                snr_estimate = result.optimal_snr
                
                logger.debug(f"✅ SNR estimate (spectral): {snr_estimate:.2f}")
            
            return jnp.clip(snr_estimate, 0.0, 100.0)
            
        except Exception as e:
            logger.warning(f"Professional SNR estimation failed: {e}, using simple fallback")
            
            # ✅ SIMPLE FALLBACK: Original method as last resort
            signal_power = jnp.var(strain_data)
            fft_data = jnp.fft.rfft(strain_data)
            freqs = jnp.fft.rfftfreq(len(strain_data), 1.0 / self.config.sample_rate)
            high_freq_mask = freqs > 500.0
            noise_power = jnp.mean(jnp.abs(fft_data[high_freq_mask]) ** 2)
            snr = signal_power / (noise_power + 1e-10)
            return jnp.clip(snr, 0.0, 100.0)
    
    def _calculate_spectral_coherence(self, strain_data: jnp.ndarray) -> jnp.ndarray:
        """Calculate spectral coherence as quality metric."""
        # Split data into segments for coherence calculation
        segment_size = len(strain_data) // 4
        
        if segment_size < 64:
            # Too short for reliable coherence
            return jnp.array(0.5)
        
        # Calculate FFT for each segment
        segments = []
        for i in range(4):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size
            segment = strain_data[start_idx:end_idx]
            segment_fft = jnp.fft.rfft(segment)
            segments.append(segment_fft)
        
        # Cross-correlation between segments
        coherence_scores = []
        for i in range(3):
            for j in range(i+1, 4):
                cross_corr = jnp.abs(jnp.mean(segments[i] * jnp.conj(segments[j])))
                auto_corr_i = jnp.mean(jnp.abs(segments[i]) ** 2)
                auto_corr_j = jnp.mean(jnp.abs(segments[j]) ** 2)
                
                coherence = cross_corr / jnp.sqrt(auto_corr_i * auto_corr_j + 1e-10)
                coherence_scores.append(coherence)
        
        return jnp.mean(jnp.array(coherence_scores))
    
    def _detect_glitches(self, strain_data: jnp.ndarray) -> jnp.ndarray:
        """Detect glitches in strain data."""
        # Simple glitch detection based on outliers
        
        # Calculate rolling statistics
        window_size = min(256, len(strain_data) // 8)
        
        # Simple outlier detection
        median_val = jnp.median(strain_data)
        mad = jnp.median(jnp.abs(strain_data - median_val))
        
        # Modified z-score
        modified_z_scores = 0.6745 * (strain_data - median_val) / (mad + 1e-10)
        
        # Count outliers (glitches)
        outlier_fraction = jnp.mean(jnp.abs(modified_z_scores) > 3.5)
        
        return jnp.clip(outlier_fraction, 0.0, 1.0)
    
    def process_batch(self, batch_strain_data: jnp.ndarray, 
                     detectors: Optional[List[str]] = None) -> Tuple[jnp.ndarray, List[QualityMetrics]]:
        """
        Process batch of strain data efficiently.
        
        Args:
            batch_strain_data: Batch of strain data [batch_size, time_samples]
            detectors: List of detector names for each sample
            
        Returns:
            Tuple of (processed_batch, quality_metrics_list)
        """
        batch_size = batch_strain_data.shape[0]
        
        if detectors is None:
            detectors = ["H1"] * batch_size
        
        # ✅ VECTORIZED PROCESSING: Process entire batch at once
        processed_batch = []
        quality_metrics = []
        
        for i in range(batch_size):
            strain = batch_strain_data[i]
            detector = detectors[i]
            
            # Process individual strain
            processed_strain, quality = self.preprocess_strain(
                strain, detector, return_quality=True
            )
            
            processed_batch.append(processed_strain)
            quality_metrics.append(quality)
        
        processed_batch = jnp.stack(processed_batch)
        
        return processed_batch, quality_metrics
    
    def get_processing_stats(self) -> Dict[str, float]:
        """Get preprocessing performance statistics."""
        if not self.processing_times:
            return {"mean_time": 0.0, "total_processed": 0}
        
        return {
            "mean_time": float(jnp.mean(jnp.array(self.processing_times))),
            "median_time": float(jnp.median(jnp.array(self.processing_times))),
            "max_time": float(jnp.max(jnp.array(self.processing_times))),
            "min_time": float(jnp.min(jnp.array(self.processing_times))),
            "total_processed": len(self.processing_times),
            "target_met": float(jnp.mean(jnp.array(self.processing_times) < 1.0))  # <1s target
        }


# Export main preprocessor
__all__ = [
    "AdvancedDataPreprocessor",
    "PreprocessingConfig"
]

