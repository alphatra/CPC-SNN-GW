"""
GW-Specific Preprocessing for Ultra-Weak Signals

This module implements preprocessing specifically designed for gravitational wave detection
where signals are naturally ultra-weak (SNR ~0.05-20) buried in noise.

Based on successful GW detection methods:
- Matched filtering (PyCBC standard)
- Spectral whitening (essential for GW)
- Template bank correlation
- SNR enhancement techniques

References:
- LIGO detection methods
- AResGW successful approach
- PyCBC matched filtering
"""

import logging
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional
from scipy import signal
from scipy.signal import welch

logger = logging.getLogger(__name__)


class GWSpecificPreprocessor:
    """
    Preprocessor specifically designed for ultra-weak GW signals.
    
    Key features:
    - Matched filtering for SNR enhancement
    - Spectral whitening (essential for GW)
    - Template correlation features
    - Chirp-specific filtering
    - SNR-based feature extraction
    """
    
    def __init__(self, sample_rate: int = 4096, segment_length: float = 8.0):
        """
        Initialize GW-specific preprocessor.
        
        Args:
            sample_rate: Sample rate in Hz
            segment_length: Segment length in seconds
        """
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(segment_length * sample_rate)
        
        # GW frequency band
        self.f_low = 20  # Hz
        self.f_high = 1000  # Hz
        
        # Create simple template bank
        self.templates = self._create_simple_templates()
        
        logger.info(f"🔧 GW-Specific Preprocessor initialized:")
        logger.info(f"   📊 Sample rate: {self.sample_rate} Hz")
        logger.info(f"   ⏱️ Segment length: {self.segment_length}s")
        logger.info(f"   🎵 Frequency band: {self.f_low}-{self.f_high} Hz")
        logger.info(f"   📚 Templates: {len(self.templates)}")
    
    def _create_simple_templates(self) -> list:
        """Create simple chirp templates for matched filtering."""
        templates = []
        
        # Different chirp masses for template diversity
        chirp_masses = [15, 25, 35]  # Solar masses
        
        for mchirp in chirp_masses:
            t = np.linspace(0, self.segment_length, self.segment_samples)
            
            # Simple chirp approximation: f(t) = f0 * (1 - t/tau)^(-3/8)
            f0 = 50  # Starting frequency
            tau = self.segment_length * 0.8  # Coalescence time
            
            # Frequency evolution
            freq = f0 * (1 - t/tau)**(-3/8)
            freq = np.clip(freq, self.f_low, self.f_high)
            
            # Amplitude evolution (increases towards coalescence)
            amp = 0.1 * (1 - t/tau)**(-1/4)
            amp = np.clip(amp, 0, 1)
            
            # Template waveform
            template = amp * np.sin(2 * np.pi * np.cumsum(freq) / self.sample_rate)
            
            # Normalize template
            template = template / np.sqrt(np.sum(template**2))
            
            templates.append(template)
        
        return templates
    
    def spectral_whitening(self, strain_data: np.ndarray) -> np.ndarray:
        """
        Apply spectral whitening (essential for GW detection).
        
        Args:
            strain_data: Raw strain data
            
        Returns:
            Whitened strain data
        """
        # Calculate PSD using Welch method
        freqs, psd = welch(
            strain_data, 
            fs=self.sample_rate,
            nperseg=self.segment_samples // 4,
            noverlap=self.segment_samples // 8,
            window='hann'
        )
        
        # Avoid division by zero
        psd = np.maximum(psd, np.max(psd) * 1e-10)
        
        # Apply whitening in frequency domain
        strain_fft = np.fft.rfft(strain_data)
        
        # Interpolate PSD to match FFT frequencies
        fft_freqs = np.fft.rfftfreq(len(strain_data), 1/self.sample_rate)
        psd_interp = np.interp(fft_freqs, freqs, psd)
        
        # Whiten
        whitened_fft = strain_fft / np.sqrt(psd_interp)
        
        # Back to time domain
        whitened_strain = np.fft.irfft(whitened_fft, n=len(strain_data))
        
        return whitened_strain
    
    def matched_filter_features(self, strain_data: np.ndarray) -> np.ndarray:
        """
        Extract matched filter features (SNR timeseries).
        
        Args:
            strain_data: Preprocessed strain data
            
        Returns:
            Feature vector from matched filtering
        """
        features = []
        
        for template in self.templates:
            # Matched filter: correlation with template
            correlation = signal.correlate(strain_data, template, mode='same')
            
            # Normalize by template energy
            template_energy = np.sum(template**2)
            if template_energy > 0:
                correlation = correlation / np.sqrt(template_energy)
            
            # Extract features from correlation
            max_corr = np.max(np.abs(correlation))
            mean_corr = np.mean(np.abs(correlation))
            std_corr = np.std(correlation)
            
            features.extend([max_corr, mean_corr, std_corr])
        
        return np.array(features)
    
    def extract_gw_features(self, strain_data: np.ndarray) -> np.ndarray:
        """
        Extract GW-specific features for ML classification.
        
        Args:
            strain_data: Raw strain data
            
        Returns:
            Feature vector optimized for GW detection
        """
        # Step 1: Bandpass filter to GW band
        sos = signal.butter(4, [self.f_low, self.f_high], btype='band', fs=self.sample_rate, output='sos')
        filtered_strain = signal.sosfilt(sos, strain_data)
        
        # Step 2: Spectral whitening (essential!)
        whitened_strain = self.spectral_whitening(filtered_strain)
        
        # Step 3: Matched filter features
        mf_features = self.matched_filter_features(whitened_strain)
        
        # Step 4: Additional GW-specific features
        
        # Frequency domain features
        strain_fft = np.fft.rfft(whitened_strain)
        power_spectrum = np.abs(strain_fft)**2
        
        # GW frequency band power
        freqs = np.fft.rfftfreq(len(whitened_strain), 1/self.sample_rate)
        gw_band_mask = (freqs >= self.f_low) & (freqs <= self.f_high)
        gw_band_power = np.sum(power_spectrum[gw_band_mask])
        total_power = np.sum(power_spectrum)
        gw_power_fraction = gw_band_power / total_power if total_power > 0 else 0
        
        # Time-frequency features
        f, t, Sxx = signal.spectrogram(
            whitened_strain, 
            fs=self.sample_rate,
            nperseg=512,
            noverlap=256
        )
        
        # Chirp-like pattern detection
        chirp_score = self._detect_chirp_pattern(f, t, Sxx)
        
        # Combine all features
        additional_features = [
            gw_power_fraction,
            chirp_score,
            np.max(np.abs(whitened_strain)),  # Peak amplitude
            np.std(whitened_strain),          # RMS
        ]
        
        # Combine matched filter + additional features
        all_features = np.concatenate([mf_features, additional_features])
        
        return all_features
    
    def _detect_chirp_pattern(self, f: np.ndarray, t: np.ndarray, Sxx: np.ndarray) -> float:
        """Detect chirp-like patterns in spectrogram."""
        # Look for increasing frequency over time (chirp signature)
        # This is a simplified version - real implementation would be more sophisticated
        
        # Find frequency centroid over time
        freq_centroids = []
        for i in range(Sxx.shape[1]):
            power_col = Sxx[:, i]
            if np.sum(power_col) > 0:
                centroid = np.sum(f * power_col) / np.sum(power_col)
                freq_centroids.append(centroid)
            else:
                freq_centroids.append(0)
        
        freq_centroids = np.array(freq_centroids)
        
        # Check if frequency increases over time (chirp signature)
        if len(freq_centroids) > 1:
            freq_slope = np.polyfit(range(len(freq_centroids)), freq_centroids, 1)[0]
            chirp_score = max(0, freq_slope / 100)  # Normalize
        else:
            chirp_score = 0
        
        return chirp_score
    
    def process_background_foreground(self, background_strain: np.ndarray, 
                                    foreground_strain: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process background and foreground data for ML training.
        
        Args:
            background_strain: Pure noise data
            foreground_strain: Noise + injections data
            
        Returns:
            Tuple of (background_features, foreground_features)
        """
        logger.info("🔧 Processing background and foreground with GW-specific methods...")
        
        # Create segments
        bg_segments = self._create_segments(background_strain)
        fg_segments = self._create_segments(foreground_strain)
        
        logger.info(f"📊 Created segments: bg={len(bg_segments)}, fg={len(fg_segments)}")
        
        # Extract features for each segment
        bg_features = []
        fg_features = []
        
        # Process background
        for i, segment in enumerate(bg_segments):
            if i % 100 == 0:
                logger.info(f"   Processing background segment {i}/{len(bg_segments)}")
            features = self.extract_gw_features(segment)
            bg_features.append(features)
        
        # Process foreground
        for i, segment in enumerate(fg_segments):
            if i % 100 == 0:
                logger.info(f"   Processing foreground segment {i}/{len(fg_segments)}")
            features = self.extract_gw_features(segment)
            fg_features.append(features)
        
        bg_features = np.array(bg_features)
        fg_features = np.array(fg_features)
        
        logger.info(f"📊 Feature extraction complete:")
        logger.info(f"   Background features: {bg_features.shape}")
        logger.info(f"   Foreground features: {fg_features.shape}")
        
        return bg_features, fg_features
    
    def _create_segments(self, strain_data: np.ndarray) -> list:
        """Create segments from strain data."""
        segments = []
        step_size = self.segment_samples // 2  # 50% overlap
        
        for start in range(0, len(strain_data) - self.segment_samples + 1, step_size):
            segment = strain_data[start:start + self.segment_samples]
            segments.append(segment)
        
        return segments


def test_gw_specific_preprocessing():
    """Test GW-specific preprocessing on MLGWSC-1 data."""
    logger.info("🧪 Testing GW-Specific Preprocessing")
    logger.info("=" * 80)
    
    try:
        from data.mlgwsc1_data_loader import MLGWSC1DataLoader
        
        # Load MLGWSC-1 data
        loader = MLGWSC1DataLoader('/teamspace/studios/this_studio/data/dataset-4/strong_test')
        
        bg_file = loader.data_dir / 'test_background_strong.hdf'
        fg_file = loader.data_dir / 'test_foreground_strong.hdf'
        
        bg_strain = loader.load_hdf_file(bg_file, 'H1')
        fg_strain = loader.load_hdf_file(fg_file, 'H1')
        
        # Take smaller sample for quick test
        bg_sample = bg_strain[:65536]  # 16 seconds
        fg_sample = fg_strain[:65536]
        
        logger.info(f"📊 Testing with {len(bg_sample)} samples ({len(bg_sample)/4096:.1f}s)")
        
        # Create GW-specific preprocessor
        gw_preprocessor = GWSpecificPreprocessor(sample_rate=4096, segment_length=8.0)
        
        # Process data
        bg_features, fg_features = gw_preprocessor.process_background_foreground(bg_sample, fg_sample)
        
        # Analyze feature separability
        bg_mean = np.mean(bg_features, axis=0)
        fg_mean = np.mean(fg_features, axis=0)
        
        # Calculate separability for each feature
        feature_separabilities = []
        for i in range(len(bg_mean)):
            bg_std = np.std(bg_features[:, i])
            fg_std = np.std(fg_features[:, i])
            mean_sep = abs(fg_mean[i] - bg_mean[i])
            combined_std = (bg_std + fg_std) / 2
            sep = mean_sep / combined_std if combined_std > 0 else 0
            feature_separabilities.append(sep)
        
        feature_separabilities = np.array(feature_separabilities)
        best_separability = np.max(feature_separabilities)
        best_feature_idx = np.argmax(feature_separabilities)
        
        logger.info(f"📊 GW-Specific Feature Analysis:")
        logger.info(f"   📊 Background features: {bg_features.shape}")
        logger.info(f"   📊 Foreground features: {fg_features.shape}")
        logger.info(f"   🎯 Best separability: {best_separability:.6f} (feature {best_feature_idx})")
        logger.info(f"   📈 Features with sep >0.01: {np.sum(feature_separabilities > 0.01)}")
        
        if best_separability > 0.01:
            logger.info("✅ GW-SPECIFIC PREPROCESSING BREAKTHROUGH!")
            improvement = best_separability / 0.0051
            logger.info(f"   📈 Improvement: {improvement:.1f}x vs raw synthetic")
            return True, bg_features, fg_features
        else:
            logger.warning(f"⚠️ Still weak separability: {best_separability:.6f}")
            return False, bg_features, fg_features
            
    except Exception as e:
        logger.error(f"❌ GW preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success, bg_feat, fg_feat = test_gw_specific_preprocessing()
    
    if success:
        logger.info("🎉 GW-Specific Preprocessing SUCCESS!")
    else:
        logger.error("❌ GW-Specific Preprocessing FAILED")
