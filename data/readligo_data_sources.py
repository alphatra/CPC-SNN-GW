"""
ReadLIGO Data Sources - Production LIGO Data Integration

⚠️ DEPRECATED: Use MLGWSCDataLoader instead for MLGWSC-1 dataset.

Replaces problematic GWOSC API with reliable ReadLIGO library.
Based on working solution from real_ligo_test.py

NOTE: This module downloads individual events from GWOSC. 
For production training, use data.mlgwsc_data_loader.MLGWSCDataLoader
with the comprehensive MLGWSC-1 dataset.
"""

import jax
import jax.numpy as jnp
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import os

logger = logging.getLogger(__name__)

# Try to import ReadLIGO - our proven solution
try:
    import readligo as rl
    HAS_READLIGO = True
    logger.info("✅ ReadLIGO available - using real LIGO data")
except ImportError:
    HAS_READLIGO = False
    logger.error("❌ ReadLIGO not available - install with: pip install readligo")

@dataclass
class LIGOEventData:
    """Container for ReadLIGO event data."""
    event_name: str
    gps_time: float
    detector: str
    strain: jnp.ndarray
    sample_rate: float
    duration: float
    snr: float
    metadata: Dict[str, Any]

@dataclass
class ProcessingResult:
    """Result from data processing operations."""
    data: jnp.ndarray
    quality_score: float
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class QualityMetrics:
    """Quality metrics for data assessment."""
    snr: float
    psd_score: float
    glitch_probability: float
    data_quality_flag: int
    timestamp: float

@dataclass
class LIGODataQuality:
    """Data quality metrics for LIGO strain data."""
    snr: float
    whitened_snr: float
    sigma_squared: float
    kurtosis: float
    rms: float
    peak_amplitude: float
    frequency_content: Dict[str, float]
    quality_flag: bool

class ReadLIGODataFetcher:
    """
    ✅ PRODUCTION ReadLIGO Data Fetcher - WORKING SOLUTION
    Uses proven ReadLIGO library instead of problematic GWOSC API.
    Based on successful implementation from real_ligo_test.py
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        if not HAS_READLIGO:
            raise ImportError("ReadLIGO not available. Install with: pip install readligo")
        
        self.cache_dir = cache_dir or Path("./data/readligo_cache")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # ✅ VERIFIED REAL LIGO FILES - Known working HDF5 files
        self.available_files = {
            'GW150914': {
                'H1': 'H-H1_LOSC_4_V2-1126259446-32.hdf5',
                'L1': 'L-L1_LOSC_4_V2-1126259446-32.hdf5',
                'gps': 1126259462.4,
                'duration': 32,
                'recommended_snr': 23.7,
                'masses': [35.5, 30.5],
                'distance': 410
            },
            'GW151226': {
                'H1': 'H-H1_LOSC_4_V2-1135136334-32.hdf5',
                'L1': 'L-L1_LOSC_4_V2-1135136334-32.hdf5', 
                'gps': 1135136350.6,
                'duration': 32,
                'recommended_snr': 13.0,
                'masses': [14.2, 7.5],
                'distance': 440
            }
        }
        
        logger.info(f"✅ ReadLIGO Data Fetcher initialized")
        logger.info(f"   Available events: {list(self.available_files.keys())}")
        logger.info(f"   Cache directory: {self.cache_dir}")
    
    def download_file_if_needed(self, filename: str) -> bool:
        """Download HDF5 file if it doesn't exist locally."""
        if os.path.exists(filename):
            logger.info(f"✅ File exists: {filename}")
            return True
            
        # Define download URLs for each file
        download_urls = {
            'H-H1_LOSC_4_V2-1126259446-32.hdf5': 'https://www.gw-openscience.org/GW150914data/H-H1_LOSC_4_V2-1126259446-32.hdf5',
            'L-L1_LOSC_4_V2-1126259446-32.hdf5': 'https://www.gw-openscience.org/GW150914data/L-L1_LOSC_4_V2-1126259446-32.hdf5',
            'H-H1_LOSC_4_V2-1135136334-32.hdf5': 'https://www.gw-openscience.org/GW151226data/H-H1_LOSC_4_V2-1135136334-32.hdf5',
            'L-L1_LOSC_4_V2-1135136334-32.hdf5': 'https://www.gw-openscience.org/GW151226data/L-L1_LOSC_4_V2-1135136334-32.hdf5'
        }
        
        url = download_urls.get(filename)
        if not url:
            logger.error(f"❌ No download URL for {filename}")
            return False
            
        try:
            import requests
            logger.info(f"📥 Downloading {filename}...")
            
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                f.write(response.content)
                
            logger.info(f"✅ Downloaded {filename} ({len(response.content)} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download {filename}: {e}")
            return False
    
    def fetch_strain_data(self, event_name: str, detector: str = 'H1',
                         duration: int = 32, sample_rate: int = 4096) -> Optional[LIGOEventData]:
        """
        ✅ WORKING: Fetch real strain data using ReadLIGO library.
        This is the proven method that works in real_ligo_test.py
        """
        if event_name not in self.available_files:
            logger.error(f"❌ Event {event_name} not available. Available: {list(self.available_files.keys())}")
            return None
            
        event_info = self.available_files[event_name]
        
        if detector not in event_info:
            logger.error(f"❌ Detector {detector} not available for {event_name}")
            return None
            
        filename = event_info[detector]
        
        # Download file if needed
        if not self.download_file_if_needed(filename):
            logger.error(f"❌ Cannot obtain file {filename}")
            return None
            
        try:
            logger.info(f"✅ Loading real LIGO data: {filename}")
            
            # ✅ PROVEN READLIGO METHOD - Same as real_ligo_test.py
            strain, time, chan_dict = rl.loaddata(filename, detector)
            
            logger.info(f"✅ Successfully loaded {detector} data:")
            logger.info(f"   Samples: {len(strain)}")
            logger.info(f"   Time span: {time[0]:.3f} to {time[-1]:.3f} GPS")
            logger.info(f"   Sample rate: {1.0/(time[1]-time[0]):.0f} Hz")
            
            # Extract portion around event time
            event_gps_time = event_info['gps']
            event_idx = np.argmin(np.abs(time - event_gps_time))
            
            # Extract samples around event (±half duration)
            half_samples = (duration * sample_rate) // 2
            start_idx = max(0, event_idx - half_samples)
            end_idx = min(len(strain), start_idx + duration * sample_rate)
            
            strain_subset = strain[start_idx:end_idx]
            
            # Pad if needed
            target_length = duration * sample_rate
            if len(strain_subset) < target_length:
                strain_padded = np.zeros(target_length, dtype=np.float32)
                strain_padded[:len(strain_subset)] = strain_subset
                strain_subset = strain_padded
            
            # Convert to JAX array
            strain_jax = jnp.array(strain_subset, dtype=jnp.float32)
            
            # Calculate basic SNR estimate
            signal_power = float(jnp.mean(strain_jax**2))
            noise_estimate = float(jnp.std(jnp.diff(strain_jax))) * np.sqrt(sample_rate/2)
            snr = np.sqrt(signal_power) / (noise_estimate + 1e-20)
            
            # Create event data object
            event_data = LIGOEventData(
                event_name=event_name,
                gps_time=event_gps_time,
                detector=detector,
                strain=strain_jax,
                sample_rate=float(sample_rate),
                duration=float(duration),
                snr=snr,
                metadata={
                    'source': 'readligo_real',
                    'filename': filename,
                    'masses': event_info['masses'],
                    'distance': event_info['distance'],
                    'verified_event': True,
                    'sample_rate_actual': 1.0/(time[1]-time[0])
                }
            )
            
            logger.info(f"✅ Created LIGOEventData: {len(strain_jax)} samples, SNR={snr:.2f}")
            return event_data
            
        except Exception as e:
            logger.error(f"❌ ReadLIGO loading failed for {event_name}/{detector}: {e}")
            return None
    
    def get_available_events(self) -> List[str]:
        """Get list of available events."""
        return list(self.available_files.keys())
    
    def get_detectors_for_event(self, event_name: str) -> List[str]:
        """Get available detectors for specific event."""
        if event_name in self.available_files:
            return [det for det in ['H1', 'L1'] if det in self.available_files[event_name]]
        return []

class LIGODataValidator:
    """
    Validates LIGO strain data quality for training use.
    Same quality criteria as original but working with ReadLIGO data.
    """
    
    def __init__(self):
        # Quality thresholds
        self.min_snr = 8.0
        self.max_kurtosis = 3.0
        self.min_quality_score = 0.8
        
    def compute_data_quality(self, strain: jnp.ndarray, 
                           sample_rate: float) -> LIGODataQuality:
        """Compute comprehensive data quality metrics."""
        # Basic statistics
        rms = float(jnp.sqrt(jnp.mean(strain**2)))
        peak_amplitude = float(jnp.max(jnp.abs(strain)))
        
        # SNR estimation
        signal_power = jnp.mean(strain**2)
        noise_estimate = jnp.std(jnp.diff(strain)) * jnp.sqrt(sample_rate/2)
        snr = float(jnp.sqrt(signal_power) / (noise_estimate + 1e-20))
        
        # Kurtosis (measure of non-Gaussianity)
        normalized_strain = (strain - jnp.mean(strain)) / (jnp.std(strain) + 1e-20)
        kurtosis = float(jnp.mean(normalized_strain**4) - 3.0)
        
        # Frequency content analysis
        fft_strain = jnp.fft.fft(strain)
        freqs = jnp.fft.fftfreq(len(strain), 1/sample_rate)
        power_spectrum = jnp.abs(fft_strain)**2
        
        # Frequency band powers
        gw_band_mask = (freqs >= 20) & (freqs <= 1024)
        total_power = jnp.sum(power_spectrum)
        gw_power = jnp.sum(power_spectrum[gw_band_mask])
        
        freq_content = {
            'gw_band_fraction': float(gw_power / (total_power + 1e-20)),
            'peak_frequency': float(freqs[jnp.argmax(power_spectrum[:len(power_spectrum)//2])]),
            'bandwidth': float(jnp.sum(gw_band_mask))
        }
        
        # Quality flag
        quality_flag = (
            snr >= self.min_snr and
            abs(kurtosis) <= self.max_kurtosis and
            freq_content['gw_band_fraction'] >= 0.1
        )
        
        return LIGODataQuality(
            snr=snr,
            whitened_snr=snr * 1.2,  # Approximate whitened SNR
            sigma_squared=float(signal_power),
            kurtosis=kurtosis,
            rms=rms,
            peak_amplitude=peak_amplitude,
            frequency_content=freq_content,
            quality_flag=quality_flag
        )
    
    def validate_for_training(self, strain: jnp.ndarray, 
                            sample_rate: float, 
                            event_name: str) -> Tuple[bool, LIGODataQuality]:
        """Validate strain data for training use."""
        quality = self.compute_data_quality(strain, sample_rate)
        
        # Additional validation for training
        is_valid = (
            quality.quality_flag and
            not jnp.any(jnp.isnan(strain)) and
            not jnp.any(jnp.isinf(strain)) and
            len(strain) > 0
        )
        
        if is_valid:
            logger.info(f"✅ Data validation passed for {event_name}: SNR={quality.snr:.2f}")
        else:
            logger.warning(f"❌ Data validation failed for {event_name}")
            
        return is_valid, quality

# Factory functions for easy access
def create_readligo_fetcher(cache_dir: Optional[Path] = None) -> ReadLIGODataFetcher:
    """Create ReadLIGO data fetcher."""
    return ReadLIGODataFetcher(cache_dir)

def create_ligo_validator() -> LIGODataValidator:
    """Create LIGO data validator."""
    return LIGODataValidator() 