"""
GW Data Preprocessor and Segment Sampler

Advanced preprocessing pipeline with quality validation optimized for Apple Silicon.
Implements whitening, band-pass filtering, glitch detection, and spectral analysis.

Features:
- Intelligent segment sampling (event/noise/mixed modes)
- Advanced preprocessing with quality assessment
- JAX-optimized filtering and spectral analysis
- Batch processing capabilities
- Apple Silicon compatibility
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from typing import List, Tuple, Optional, Dict, Any
import logging
import time
import numpy as np

from .gw_data_sources import QualityMetrics, ProcessingResult

logger = logging.getLogger(__name__)


class SegmentSampler:
    """
    Intelligent segment sampler for gravitational wave data.
    
    Implements mixed sampling strategy combining noise periods and known GW events.
    """
    
    def __init__(self, mode: str = "mixed", seed: Optional[int] = None):
        """
        Initialize segment sampler.
        
        Args:
            mode: Sampling mode ("noise", "event", "mixed")
            seed: Random seed for reproducible experiments
        """
        if mode not in ["noise", "event", "mixed"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'noise', 'event', or 'mixed'.")
        
        self.mode = mode
        self.seed = seed
        
        # Known GW events for training
        self.known_events = [
            # GPS time, detector, description
            (1126259446, ['H1', 'L1'], 'GW150914'),  # First detection
            (1128678900, ['H1', 'L1'], 'GW151012'),  # Second detection
            (1135136350, ['H1', 'L1'], 'GW151226'),  # Third detection
            (1167559936, ['H1', 'L1'], 'GW170104'),  # Fourth detection
            (1180922494, ['H1', 'L1'], 'GW170608'),  # Fifth detection
            (1185389807, ['H1', 'L1'], 'GW170814'),  # Triple detection
            (1187008882, ['H1', 'L1'], 'GW170823'),  # Sixth detection
        ]
        
        # Noise periods (GPS times with good data quality but no known events)
        self.noise_periods = [
            (1126258000, ['H1', 'L1']),  # Before GW150914
            (1126261000, ['H1', 'L1']),  # After GW150914
            (1128677000, ['H1', 'L1']),  # Before GW151012
            (1128680000, ['H1', 'L1']),  # After GW151012
            (1135135000, ['H1', 'L1']),  # Before GW151226
            (1135138000, ['H1', 'L1']),  # After GW151226
            (1167558000, ['H1', 'L1']),  # Before GW170104
            (1167561000, ['H1', 'L1']),  # After GW170104
            (1180921000, ['H1', 'L1']),  # Before GW170608
            (1180924000, ['H1', 'L1']),  # After GW170608
        ]
    
    def sample_segments(self, num_segments: int, duration: float = 4.0) -> List[Tuple[str, int, float]]:
        """
        Sample segments according to the specified mode.
        
        Args:
            num_segments: Number of segments to sample
            duration: Duration of each segment in seconds
            
        Returns:
            List of (detector, start_time, duration) tuples
        """
        segments = []
        
        if self.mode == "noise":
            segments = self._sample_noise_segments(num_segments, duration)
        elif self.mode == "event":
            segments = self._sample_event_segments(num_segments, duration)
        elif self.mode == "mixed":
            # Mixed strategy: 50% events, 50% noise
            num_events = num_segments // 2
            num_noise = num_segments - num_events
            
            # Use proper JAX random key splitting
            if self.seed is not None:
                key = jax.random.PRNGKey(self.seed)
                event_key, noise_key = jax.random.split(key)
            else:
                event_key = jax.random.PRNGKey(42)
                noise_key = jax.random.PRNGKey(43)
            
            segments.extend(self._sample_event_segments(num_events, duration, key=event_key))
            segments.extend(self._sample_noise_segments(num_noise, duration, key=noise_key))
        
        return segments
    
    def _sample_event_segments(self, num_segments: int, duration: float, 
                             key: jax.random.PRNGKey = None) -> List[Tuple[str, int, float]]:
        """Sample segments around known GW events with JAX random."""
        segments = []
        
        # Use provided key or create new one
        if key is None:
            if self.seed is not None:
                key = jax.random.PRNGKey(self.seed)
            else:
                key = jax.random.PRNGKey(42)  # Default seed
        
        for i in range(num_segments):
            # Split key for this iteration
            key, subkey = jax.random.split(key)
            
            # Cycle through known events
            event_idx = i % len(self.known_events)
            event_time, detectors, event_name = self.known_events[event_idx]
            
            # Random detector from available detectors
            detector_idx = jax.random.randint(subkey, (), 0, len(detectors))
            detector = detectors[int(detector_idx)]
            
            # Random offset around event time (-2 to +2 seconds)
            key, offset_key = jax.random.split(key)
            offset = jax.random.uniform(offset_key, (), minval=-2.0, maxval=2.0)
            start_time = int(event_time + offset)
            
            segments.append((detector, start_time, duration))
            logger.debug(f"Event segment: {detector} around {event_name} at {start_time}")
        
        return segments
    
    def _sample_noise_segments(self, num_segments: int, duration: float,
                             key: jax.random.PRNGKey = None) -> List[Tuple[str, int, float]]:
        """Sample segments from noise periods with JAX random."""
        segments = []
        
        # Use provided key or create new one
        if key is None:
            if self.seed is not None:
                key = jax.random.PRNGKey(self.seed + 1)  # Different seed for noise
            else:
                key = jax.random.PRNGKey(43)  # Default seed
        
        for i in range(num_segments):
            # Split key for this iteration
            key, subkey = jax.random.split(key)
            
            # Cycle through noise periods
            period_idx = i % len(self.noise_periods)
            base_time, detectors = self.noise_periods[period_idx]
            
            # Random detector from available detectors
            detector_idx = jax.random.randint(subkey, (), 0, len(detectors))
            detector = detectors[int(detector_idx)]
            
            # Random offset within noise period (0 to 1000 seconds)
            key, offset_key = jax.random.split(key)
            offset = jax.random.uniform(offset_key, (), minval=0, maxval=1000)
            start_time = int(base_time + offset)
            
            segments.append((detector, start_time, duration))
            logger.debug(f"Noise segment: {detector} at {start_time}")
        
        return segments


class AdvancedDataPreprocessor:
    """
    Advanced preprocessing pipeline with quality validation optimized for Apple Silicon.
    
    Implements whitening, band-pass filtering, glitch detection, and spectral analysis.
    Target: <1s processing time per 4s segment.
    """
    
    def __init__(self, 
                 sample_rate: int = 4096,
                 bandpass: Tuple[float, float] = (20.0, 1024.0),
                 apply_whitening: bool = True,
                 psd_length: int = 8,  # seconds for PSD estimation
                 quality_threshold: float = 0.7):
        self.sample_rate = sample_rate
        self.bandpass = bandpass
        self.apply_whitening = apply_whitening
        self.psd_length = psd_length
        self.quality_threshold = quality_threshold
        
        # Pre-compute filter coefficients for efficiency
        self._setup_filters()
        
    def _setup_filters(self):
        """Pre-compute filter coefficients for band-pass filtering."""
        from scipy.signal import cheby2
        
        nyquist = self.sample_rate / 2
        low_norm = self.bandpass[0] / nyquist
        high_norm = self.bandpass[1] / nyquist
        
        # Ensure valid normalized frequencies
        low_norm = jnp.clip(low_norm, 0.001, 0.999)
        high_norm = jnp.clip(high_norm, 0.001, 0.999)
        
        # Design 8th-order Chebyshev Type II filter
        self.filter_sos = cheby2(
            N=8,  # 8th order filter
            rs=40,  # 40dB stopband attenuation
            Wn=[low_norm, high_norm],
            btype='band',
            output='sos'
        )
        
        # Convert to JAX array for JIT compilation
        self.filter_sos_jax = jnp.array(self.filter_sos)
        
        logger.info(f"Initialized Chebyshev Type II band-pass filter: {self.bandpass[0]}-{self.bandpass[1]} Hz")
        
    def _bandpass_filter_jax(self, data: jnp.ndarray) -> jnp.ndarray:
        """
        JAX-optimized band-pass filtering using CPU fallback.
        
        Uses Chebyshev Type II filter with CPU SciPy for compatibility.
        JAX sosfilt is not implemented, so we use CPU processing.
        
        Note: Cannot use @jax.jit due to CPU fallback.
        """
        # CPU fallback since jax.scipy.signal.sosfilt doesn't exist
        from scipy.signal import sosfilt
        
        # Convert to CPU, filter, and convert back
        cpu_data = np.array(data)
        filtered_cpu = sosfilt(self.filter_sos, cpu_data)
        
        # Convert back to JAX array
        return jnp.asarray(filtered_cpu)
    
    def _bandpass_filter_batch(self, data_batch: jnp.ndarray) -> jnp.ndarray:
        """
        Vectorized batch processing of band-pass filtering using vmap.
        
        Args:
            data_batch: Batch of data [batch, time_samples]
            
        Returns:
            Filtered batch [batch, time_samples]
        """
        # Vectorize filtering function over batch dimension
        vectorized_filter = jax.vmap(self._bandpass_filter_jax, in_axes=0, out_axes=0)
        
        return vectorized_filter(data_batch)
        
    def estimate_psd(self, strain_data: jnp.ndarray) -> jnp.ndarray:
        """
        Estimate power spectral density using CPU Welch method.
        
        Args:
            strain_data: Input strain timeseries
            
        Returns:
            Power spectral density
        """
        # CPU fallback since jax.scipy.signal.welch doesn't exist
        from scipy.signal import welch
        
        # Fix for short segments: ensure psd_length is at least 4 seconds
        effective_psd_length = max(4, self.psd_length)
        nperseg = effective_psd_length * self.sample_rate
        
        # Ensure nperseg doesn't exceed data length
        nperseg = min(nperseg, len(strain_data) // 4)
        
        # Convert to CPU for processing
        cpu_data = np.array(strain_data)
        
        # Use SciPy Welch's method
        freqs, psd = welch(
            cpu_data,
            fs=self.sample_rate,
            nperseg=nperseg,
            noverlap=nperseg // 2,  # 50% overlap
            window='hann'
        )
        
        # Convert back to JAX array
        return jnp.asarray(psd)
    
    def estimate_psd_batch(self, strain_batch: jnp.ndarray) -> jnp.ndarray:
        """
        Vectorized batch PSD estimation using vmap.
        
        Note: Cannot use @jax.jit due to CPU fallback in estimate_psd.
        """
        # Use vmap for vectorization even with CPU processing
        vectorized_psd = jax.vmap(self.estimate_psd)
        return vectorized_psd(strain_batch)
        
    def _whiten_data(self, strain_data: jnp.ndarray, psd: jnp.ndarray) -> jnp.ndarray:
        """
        Whiten strain data using estimated PSD.
        
        Args:
            strain_data: Input strain timeseries
            psd: Power spectral density
            
        Returns:
            Whitened strain data
        """
        # FFT of strain data
        strain_fft = jnp.fft.fft(strain_data)
        freqs = jnp.fft.fftfreq(len(strain_data), 1/self.sample_rate)
        
        # Interpolate PSD to match FFT frequencies
        psd_interp = jnp.interp(jnp.abs(freqs), 
                               jnp.linspace(0, self.sample_rate/2, len(psd)), 
                               psd)
        
        # Avoid division by zero
        psd_interp = jnp.where(psd_interp < 1e-40, 1e-40, psd_interp)
        
        # Whiten in frequency domain
        whitened_fft = strain_fft / jnp.sqrt(psd_interp)
        
        # Return to time domain
        return jnp.real(jnp.fft.ifft(whitened_fft))
    
    def assess_quality(self, strain_data: jnp.ndarray, psd: Optional[jnp.ndarray] = None) -> QualityMetrics:
        """
        Assess data quality using multiple metrics.
        
        Args:
            strain_data: Input strain data
            psd: Optional PSD for SNR estimation
            
        Returns:
            Quality metrics object
        """
        # Basic statistics
        mean_val = float(jnp.mean(strain_data))
        std_val = float(jnp.std(strain_data))
        rms_val = float(jnp.sqrt(jnp.mean(strain_data**2)))
        
        # Estimate SNR (simplified)
        if psd is not None:
            # Simple SNR estimate based on signal power vs noise floor
            signal_power = jnp.mean(strain_data**2)
            noise_floor = jnp.median(psd)
            snr_estimate = float(10 * jnp.log10(signal_power / noise_floor))
        else:
            # Fallback SNR estimate
            snr_estimate = float(20 * jnp.log10(rms_val / (jnp.abs(mean_val) + 1e-10)))
        
        # Kurtosis for non-Gaussianity detection
        kurtosis = float(_compute_kurtosis(strain_data))
        
        # Quality score (0-1 scale)
        quality_score = self._compute_quality_score(std_val, snr_estimate, kurtosis)
        
        # Glitch detection (simplified)
        glitch_detected = abs(kurtosis) > 5.0 or quality_score < self.quality_threshold
        
        return QualityMetrics(
            snr_estimate=snr_estimate,
            noise_floor=float(jnp.median(jnp.abs(strain_data))),
            glitch_detected=glitch_detected,
            quality_score=quality_score,
            rms_noise=rms_val,
            kurtosis=kurtosis
        )
    
    def _compute_quality_score(self, std_val: float, snr_estimate: float, kurtosis: float) -> float:
        """Compute composite quality score."""
        # Normalize components to 0-1 scale
        std_score = min(1.0, max(0.0, 1.0 - abs(np.log10(std_val + 1e-10))))
        snr_score = min(1.0, max(0.0, (snr_estimate + 50) / 100))  # Assume SNR range -50 to 50
        kurtosis_score = min(1.0, max(0.0, 1.0 - abs(kurtosis) / 10))  # Penalize high kurtosis
        
        # Weighted average
        return 0.4 * std_score + 0.4 * snr_score + 0.2 * kurtosis_score
    
    def process(self, strain_data: jnp.ndarray) -> ProcessingResult:
        """
        Complete processing pipeline for strain data.
        
        Args:
            strain_data: Raw strain data array
            
        Returns:
            Processing result with processed data and quality metrics
        """
        start_time = time.perf_counter()
        
        # 1. Band-pass filtering
        filtered_data = self._bandpass_filter_jax(strain_data)
        
        # 2. PSD estimation
        psd = self.estimate_psd(filtered_data)
        
        # 3. Whitening (optional)
        if self.apply_whitening:
            processed_data = self._whiten_data(filtered_data, psd)
        else:
            processed_data = filtered_data
        
        # 4. Quality assessment
        quality = self.assess_quality(processed_data, psd)
        
        processing_time = time.perf_counter() - start_time
        
        # 5. Metadata collection
        metadata = {
            'original_length': len(strain_data),
            'processed_length': len(processed_data),
            'sample_rate': self.sample_rate,
            'bandpass': self.bandpass,
            'whitening_applied': self.apply_whitening,
            'processing_time_ms': processing_time * 1000
        }
        
        logger.debug(f"Processing completed in {processing_time*1000:.1f}ms "
                    f"(quality: {quality.snr_estimate:.1f} SNR)")
        
        return ProcessingResult(
            strain_data=processed_data,
            psd=psd,
            quality=quality,
            processing_time=processing_time,
            metadata=metadata
        )
        
    def process_batch(self, strain_segments: List[jnp.ndarray]) -> List[ProcessingResult]:
        """
        Process multiple strain segments in batch for efficiency.
        
        Args:
            strain_segments: List of strain data arrays
            
        Returns:
            List of processing results
        """
        logger.info(f"Batch processing {len(strain_segments)} segments")
        
        results = []
        total_start = time.perf_counter()
        
        for i, segment in enumerate(strain_segments):
            if segment is not None:
                result = self.process(segment)
                results.append(result)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    elapsed = time.perf_counter() - total_start
                    avg_time = elapsed / (i + 1) * 1000
                    logger.info(f"Processed {i+1}/{len(strain_segments)} segments "
                               f"(avg: {avg_time:.1f}ms per segment)")
            else:
                results.append(None)
                
        total_time = time.perf_counter() - total_start
        valid_results = [r for r in results if r is not None]
        
        logger.info(f"Batch processing completed in {total_time:.2f}s "
                   f"({len(valid_results)}/{len(strain_segments)} successful)")
        
        return results
    
    def assess_quality_batch(self, strain_batch: jnp.ndarray) -> List[QualityMetrics]:
        """
        Vectorized quality assessment for batch processing.
        
        Args:
            strain_batch: Batch of strain data arrays [batch_size, time_samples]
            
        Returns:
            List of quality metrics for each sample
        """
        # JAX vmap doesn't work well with custom dataclasses, so we'll do it manually
        # but use vectorized computations where possible
        batch_results = []
        
        for i in range(strain_batch.shape[0]):
            quality = self.assess_quality(strain_batch[i])
            batch_results.append(quality)
            
        return batch_results


def _compute_kurtosis(data: jnp.ndarray) -> float:
    """Compute kurtosis manually since jax.scipy.stats.kurtosis doesn't exist."""
    mean = jnp.mean(data)
    std = jnp.std(data)
    standardized = (data - mean) / std
    kurt = jnp.mean(standardized**4) - 3  # Excess kurtosis
    return float(kurt)


# Legacy alias for backward compatibility
DataPreprocessor = AdvancedDataPreprocessor 