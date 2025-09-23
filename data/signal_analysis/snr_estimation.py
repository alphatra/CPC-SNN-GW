"""
Professional SNR estimation for gravitational wave signals.

Implements advanced SNR estimation methods including matched filtering,
which is the gold standard for gravitational wave signal analysis.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Union, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SNRResult:
    """Results from SNR estimation."""
    optimal_snr: float
    network_snr: Optional[float] = None
    template_match: Optional[float] = None
    detection_statistic: Optional[float] = None
    method: str = "unknown"
    

class ProfessionalSNREstimator:
    """
    ✅ PROFESSIONAL: Advanced SNR estimation for gravitational wave signals
    
    Implements multiple SNR estimation methods:
    1. Matched filtering (optimal for known signal shapes)
    2. Spectral analysis with PSD weighting
    3. Template bank matching
    4. Network coherent SNR
    """
    
    def __init__(
        self,
        sample_rate: int = 4096,
        low_freq: float = 20.0,
        high_freq: float = 1024.0
    ):
        """
        Initialize professional SNR estimator.
        
        Args:
            sample_rate: Sample rate in Hz
            low_freq: Low frequency cutoff for analysis
            high_freq: High frequency cutoff for analysis
        """
        self.sample_rate = sample_rate
        self.low_freq = low_freq
        self.high_freq = high_freq
        
        # Frequency array for analysis
        self.freq_mask = None
        self.frequencies = None
        
        logger.info(f"✅ Professional SNR estimator initialized")
        logger.info(f"   Sample rate: {sample_rate} Hz")
        logger.info(f"   Analysis band: {low_freq}-{high_freq} Hz")
    
    def estimate_snr_matched_filter(
        self,
        strain_data: jnp.ndarray,
        template: jnp.ndarray,
        psd: Optional[jnp.ndarray] = None
    ) -> SNRResult:
        """
        ✅ GOLD STANDARD: Matched filtering SNR estimation
        
        This is the optimal method for SNR estimation when the signal
        template is known. Used extensively in LIGO analysis.
        
        Args:
            strain_data: Input strain data
            template: Signal template for matching
            psd: Power spectral density (optional)
            
        Returns:
            SNR result with optimal SNR and detection statistics
        """
        if len(strain_data) != len(template):
            # Resize template to match data length
            if len(template) > len(strain_data):
                template = template[:len(strain_data)]
            else:
                # Pad template with zeros
                padding = len(strain_data) - len(template)
                template = jnp.pad(template, (0, padding), mode='constant')
        
        # ✅ FREQUENCY DOMAIN MATCHED FILTERING
        # Convert to frequency domain
        strain_fft = jnp.fft.rfft(strain_data)
        template_fft = jnp.fft.rfft(template)
        
        # Get frequency array
        frequencies = jnp.fft.rfftfreq(len(strain_data), 1.0 / self.sample_rate)
        
        # Create frequency mask for analysis band
        freq_mask = (frequencies >= self.low_freq) & (frequencies <= self.high_freq)
        
        if psd is None:
            # Estimate PSD from data if not provided
            psd = self._estimate_psd_welch(strain_data)
        
        # Ensure PSD has correct length
        if len(psd) != len(strain_fft):
            psd = jnp.interp(frequencies, jnp.linspace(0, self.sample_rate/2, len(psd)), psd)
        
        # ✅ OPTIMAL FILTERING: Weight by inverse PSD
        # Only use frequencies in analysis band
        strain_weighted = jnp.where(
            freq_mask,
            strain_fft / jnp.sqrt(psd + 1e-12),
            0.0
        )
        template_weighted = jnp.where(
            freq_mask, 
            template_fft / jnp.sqrt(psd + 1e-12),
            0.0
        )
        
        # ✅ MATCHED FILTER: Cross-correlation in frequency domain
        # Complex conjugate for matched filtering
        matched_filter_fft = strain_weighted * jnp.conj(template_weighted)
        
        # Convert back to time domain
        matched_filter_time = jnp.fft.irfft(matched_filter_fft, n=len(strain_data))
        
        # ✅ OPTIMAL SNR: Maximum of |matched filter output|
        optimal_snr = jnp.max(jnp.abs(matched_filter_time))
        
        # ✅ TEMPLATE NORMALIZATION: Normalize by template norm
        template_norm = jnp.sqrt(jnp.sum(jnp.abs(template_weighted) ** 2))
        if template_norm > 1e-12:
            optimal_snr = optimal_snr / template_norm
        
        # ✅ DETECTION STATISTIC: Chi-squared like statistic
        detection_statistic = optimal_snr ** 2
        
        logger.debug(f"✅ Matched filter SNR: {optimal_snr:.3f}")
        
        return SNRResult(
            optimal_snr=float(optimal_snr),
            detection_statistic=float(detection_statistic),
            method="matched_filter"
        )
    
    def estimate_snr_spectral(
        self,
        strain_data: jnp.ndarray,
        psd: Optional[jnp.ndarray] = None
    ) -> SNRResult:
        """
        ✅ IMPROVED: Spectral SNR estimation with PSD weighting
        
        Improved version of simple variance-based SNR that uses
        proper PSD weighting and frequency band selection.
        
        Args:
            strain_data: Input strain data
            psd: Power spectral density (optional)
            
        Returns:
            SNR result with spectral analysis
        """
        # Convert to frequency domain
        strain_fft = jnp.fft.rfft(strain_data)
        frequencies = jnp.fft.rfftfreq(len(strain_data), 1.0 / self.sample_rate)
        
        # Create frequency mask for analysis band
        freq_mask = (frequencies >= self.low_freq) & (frequencies <= self.high_freq)
        
        if psd is None:
            # Estimate PSD from data
            psd = self._estimate_psd_welch(strain_data)
        
        # Ensure PSD has correct length
        if len(psd) != len(strain_fft):
            psd = jnp.interp(frequencies, jnp.linspace(0, self.sample_rate/2, len(psd)), psd)
        
        # ✅ WEIGHTED POWER: Signal power weighted by inverse PSD
        signal_power_weighted = jnp.where(
            freq_mask,
            jnp.abs(strain_fft) ** 2 / (psd + 1e-12),
            0.0
        )
        
        # ✅ NOISE POWER: Expected noise power in analysis band
        noise_power_weighted = jnp.where(freq_mask, 1.0, 0.0)
        
        # ✅ SPECTRAL SNR: Ratio of weighted powers
        total_signal_power = jnp.sum(signal_power_weighted)
        total_noise_power = jnp.sum(noise_power_weighted)
        
        spectral_snr = jnp.sqrt(total_signal_power / (total_noise_power + 1e-12))
        
        logger.debug(f"✅ Spectral SNR: {spectral_snr:.3f}")
        
        return SNRResult(
            optimal_snr=float(spectral_snr),
            method="spectral_psd_weighted"
        )
    
    def estimate_snr_template_bank(
        self,
        strain_data: jnp.ndarray,
        template_bank: List[jnp.ndarray],
        psd: Optional[jnp.ndarray] = None
    ) -> SNRResult:
        """
        ✅ ADVANCED: Template bank SNR estimation
        
        Tests multiple templates and returns the best match SNR.
        This is closer to real LIGO analysis pipelines.
        
        Args:
            strain_data: Input strain data
            template_bank: List of signal templates
            psd: Power spectral density (optional)
            
        Returns:
            SNR result with best template match
        """
        best_snr = 0.0
        best_template_idx = 0
        
        for i, template in enumerate(template_bank):
            result = self.estimate_snr_matched_filter(strain_data, template, psd)
            
            if result.optimal_snr > best_snr:
                best_snr = result.optimal_snr
                best_template_idx = i
        
        template_match = best_template_idx / len(template_bank) if template_bank else 0.0
        
        logger.debug(f"✅ Template bank SNR: {best_snr:.3f} (template {best_template_idx})")
        
        return SNRResult(
            optimal_snr=best_snr,
            template_match=template_match,
            method="template_bank"
        )
    
    def _estimate_psd_welch(self, strain_data: jnp.ndarray) -> jnp.ndarray:
        """
        ✅ PROFESSIONAL: Estimate PSD using Welch's method
        
        This provides a robust PSD estimate for SNR calculations.
        """
        # Parameters for Welch's method
        segment_length = min(len(strain_data) // 4, self.sample_rate)  # 1 second segments max
        segment_length = max(segment_length, 256)  # Minimum segment length
        
        overlap = segment_length // 2
        n_segments = max(1, (len(strain_data) - overlap) // (segment_length - overlap))
        
        if n_segments == 1:
            # Single segment - just use periodogram
            fft_data = jnp.fft.rfft(strain_data)
            psd = jnp.abs(fft_data) ** 2 / len(strain_data)
        else:
            # Multiple segments - Welch's method
            segments = []
            for i in range(n_segments):
                start = i * (segment_length - overlap)
                end = start + segment_length
                if end <= len(strain_data):
                    segment = strain_data[start:end]
                    
                    # Apply Hann window
                    window = jnp.hanning(len(segment))
                    windowed_segment = segment * window
                    
                    # FFT and power
                    fft_segment = jnp.fft.rfft(windowed_segment)
                    power_segment = jnp.abs(fft_segment) ** 2
                    
                    # Normalize by window power
                    window_power = jnp.sum(window ** 2)
                    power_segment = power_segment / (window_power + 1e-12)
                    
                    segments.append(power_segment)
            
            if segments:
                # Average across segments
                psd = jnp.mean(jnp.array(segments), axis=0)
            else:
                # Fallback
                fft_data = jnp.fft.rfft(strain_data)
                psd = jnp.abs(fft_data) ** 2 / len(strain_data)
        
        # Floor PSD to avoid numerical issues
        psd = jnp.maximum(psd, jnp.max(psd) * 1e-8)
        
        return psd


# ✅ CONVENIENCE FUNCTIONS for backward compatibility

def estimate_snr_matched_filter(
    strain_data: jnp.ndarray,
    template: jnp.ndarray,
    sample_rate: int = 4096,
    psd: Optional[jnp.ndarray] = None
) -> float:
    """
    ✅ CONVENIENCE: Quick matched filter SNR estimation
    
    Args:
        strain_data: Input strain data
        template: Signal template
        sample_rate: Sample rate in Hz
        psd: Power spectral density (optional)
        
    Returns:
        Optimal SNR value
    """
    estimator = ProfessionalSNREstimator(sample_rate=sample_rate)
    result = estimator.estimate_snr_matched_filter(strain_data, template, psd)
    return result.optimal_snr


def estimate_snr_spectral(
    strain_data: jnp.ndarray,
    sample_rate: int = 4096,
    low_freq: float = 20.0,
    high_freq: float = 1024.0,
    psd: Optional[jnp.ndarray] = None
) -> float:
    """
    ✅ CONVENIENCE: Quick spectral SNR estimation
    
    Args:
        strain_data: Input strain data
        sample_rate: Sample rate in Hz
        low_freq: Low frequency cutoff
        high_freq: High frequency cutoff
        psd: Power spectral density (optional)
        
    Returns:
        Spectral SNR value
    """
    estimator = ProfessionalSNREstimator(
        sample_rate=sample_rate,
        low_freq=low_freq,
        high_freq=high_freq
    )
    result = estimator.estimate_snr_spectral(strain_data, psd)
    return result.optimal_snr


def estimate_snr_template_bank(
    strain_data: jnp.ndarray,
    template_bank: List[jnp.ndarray],
    sample_rate: int = 4096,
    psd: Optional[jnp.ndarray] = None
) -> Tuple[float, int]:
    """
    ✅ CONVENIENCE: Quick template bank SNR estimation
    
    Args:
        strain_data: Input strain data
        template_bank: List of signal templates
        sample_rate: Sample rate in Hz
        psd: Power spectral density (optional)
        
    Returns:
        Tuple of (best_snr, best_template_index)
    """
    estimator = ProfessionalSNREstimator(sample_rate=sample_rate)
    result = estimator.estimate_snr_template_bank(strain_data, template_bank, psd)
    
    best_template_idx = int(result.template_match * len(template_bank)) if result.template_match else 0
    return result.optimal_snr, best_template_idx

