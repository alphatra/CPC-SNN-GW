"""
GW Data Sources: Abstractions and Source Interfaces
Extracted from gw_download.py for modular architecture.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality assessment metrics for strain data."""
    is_valid: bool
    snr_estimate: float
    glitch_probability: float
    spectral_line_contamination: float
    data_completeness: float
    outlier_fraction: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'is_valid': self.is_valid,
            'snr_estimate': self.snr_estimate,
            'glitch_probability': self.glitch_probability,
            'spectral_line_contamination': self.spectral_line_contamination,
            'data_completeness': self.data_completeness,
            'outlier_fraction': self.outlier_fraction
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QualityMetrics':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ProcessingResult:
    """Result of data processing with quality metrics."""
    strain_data: jnp.ndarray
    psd: Optional[jnp.ndarray]
    quality: QualityMetrics
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'strain_data_shape': self.strain_data.shape,
            'strain_data_dtype': str(self.strain_data.dtype),
            'psd_shape': self.psd.shape if self.psd is not None else None,
            'quality': self.quality.to_dict(),
            'processing_time': self.processing_time,
            'metadata': self.metadata
        }


class DataSource(ABC):
    """Abstract base class for gravitational wave data sources."""
    
    @abstractmethod
    def fetch(self, detector: str, start_time: int, duration: float) -> jnp.ndarray:
        """
        Fetch strain data from source.
        
        Args:
            detector: Detector identifier (e.g., 'H1', 'L1', 'V1')
            start_time: GPS start time
            duration: Duration in seconds
            
        Returns:
            Strain data array
        """
        pass
    
    @abstractmethod
    def get_available_detectors(self) -> List[str]:
        """Get list of available detectors."""
        pass
    
    @abstractmethod
    def get_available_timerange(self, detector: str) -> Tuple[int, int]:
        """Get available time range for detector as (start_gps, end_gps)."""
        pass
    
    def is_detector_available(self, detector: str) -> bool:
        """Check if detector is available."""
        return detector in self.get_available_detectors()
    
    def validate_request(self, detector: str, start_time: int, duration: float) -> bool:
        """
        Validate data request parameters.
        
        Args:
            detector: Detector identifier
            start_time: GPS start time
            duration: Duration in seconds
            
        Returns:
            True if request is valid
        """
        # Check detector availability
        if not self.is_detector_available(detector):
            logger.error(f"Detector {detector} not available")
            return False
        
        # Check time range
        try:
            start_available, end_available = self.get_available_timerange(detector)
            end_time = start_time + duration
            
            if start_time < start_available or end_time > end_available:
                logger.error(f"Requested time range [{start_time}, {end_time}] "
                           f"outside available range [{start_available}, {end_available}]")
                return False
        except Exception as e:
            logger.warning(f"Could not validate time range: {e}")
        
        # Check duration
        if duration <= 0:
            logger.error(f"Duration must be positive, got {duration}")
            return False
        
        return True


class SegmentSampler:
    """
    Smart segment sampler for GWOSC data with event-aware sampling.
    
    Features:
    - Mixed sampling strategy (around known events + pure noise)
    - Configurable sampling modes
    - Avoids known problematic periods
    """
    
    def __init__(self, mode: str = "mixed", seed: Optional[int] = None):
        """
        Initialize segment sampler.
        
        Args:
            mode: Sampling mode ('mixed', 'events', 'noise')
            seed: Random seed for reproducibility
        """
        self.mode = mode
        self.key = jax.random.PRNGKey(seed or 42)
        
        # Known GW events for event-aware sampling
        self.known_events = {
            'GW150914': 1126259462,  # First detection
            'GW151012': 1128678900,  
            'GW151226': 1135136350,
            'GW170104': 1167559936,
            'GW170608': 1180922494,
            'GW170729': 1185389807,
            'GW170809': 1186302519,
            'GW170814': 1186741861,
            'GW170817': 1187008882,  # Binary neutron star
            'GW170823': 1187529256
        }
        
        # Typical LIGO operational periods
        self.operational_periods = [
            (1126051217, 1137254417),  # O1: Sep 2015 - Jan 2016
            (1164556817, 1187733618),  # O2: Nov 2016 - Aug 2017
            (1238166018, 1253977218),  # O3a: Apr 2019 - Oct 2019
            (1256655618, 1269363618),  # O3b: Nov 2019 - Mar 2020
        ]
        
        logger.info(f"Initialized SegmentSampler with mode: {mode}")
        logger.info(f"  Known events: {len(self.known_events)}")
        logger.info(f"  Operational periods: {len(self.operational_periods)}")


    def sample_segments(self, num_segments: int, duration: float = 4.0) -> List[Tuple[str, int, float]]:
        """
        Sample data segments using configured strategy.
        
        Args:
            num_segments: Number of segments to sample
            duration: Duration of each segment
            
        Returns:
            List of (detector, start_time, duration) tuples
        """
        if self.mode == "mixed":
            # 50% around events, 50% pure noise
            num_events = num_segments // 2
            num_noise = num_segments - num_events
            
            self.key, subkey1, subkey2 = jax.random.split(self.key, 3)
            
            event_segments = self._sample_event_segments(num_events, duration, subkey1)
            noise_segments = self._sample_noise_segments(num_noise, duration, subkey2)
            
            return event_segments + noise_segments
            
        elif self.mode == "events":
            self.key, subkey = jax.random.split(self.key)
            return self._sample_event_segments(num_segments, duration, subkey)
            
        elif self.mode == "noise":
            self.key, subkey = jax.random.split(self.key)
            return self._sample_noise_segments(num_segments, duration, subkey)
            
        else:
            raise ValueError(f"Unknown sampling mode: {self.mode}")


    def _sample_event_segments(self, num_segments: int, duration: float, 
                             key: jax.random.PRNGKey = None) -> List[Tuple[str, int, float]]:
        """Sample segments around known GW events."""
        if key is None:
            key = self.key
            
        segments = []
        event_times = list(self.known_events.values())
        detectors = ['H1', 'L1']
        
        for i in range(num_segments):
            # Random event and detector
            key, subkey1, subkey2 = jax.random.split(key, 3)
            
            event_idx = jax.random.randint(subkey1, (), 0, len(event_times))
            detector_idx = jax.random.randint(subkey2, (), 0, len(detectors))
            
            event_time = event_times[event_idx]
            detector = detectors[detector_idx]
            
            # Random offset around event (±60 seconds)
            key, subkey3 = jax.random.split(key)
            offset = jax.random.uniform(subkey3, (), minval=-60.0, maxval=60.0)
            start_time = int(event_time + offset)
            
            segments.append((detector, start_time, duration))
        
        return segments


    def _sample_noise_segments(self, num_segments: int, duration: float,
                             key: jax.random.PRNGKey = None) -> List[Tuple[str, int, float]]:
        """Sample pure noise segments from operational periods."""
        if key is None:
            key = self.key
            
        segments = []
        detectors = ['H1', 'L1']
        
        for i in range(num_segments):
            # Random operational period and detector
            key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
            
            period_idx = jax.random.randint(subkey1, (), 0, len(self.operational_periods))
            detector_idx = jax.random.randint(subkey2, (), 0, len(detectors))
            
            period_start, period_end = self.operational_periods[period_idx]
            detector = detectors[detector_idx]
            
            # Random time within period (avoiding event neighborhoods)
            safe_start = period_start + 3600  # 1 hour buffer
            safe_end = period_end - 3600 - int(duration)
            
            if safe_end <= safe_start:
                # Fallback to simple random in period
                safe_start = period_start
                safe_end = period_end - int(duration)
            
            start_time = jax.random.randint(subkey3, (), safe_start, safe_end)
            start_time = int(start_time)
            
            segments.append((detector, start_time, duration))
        
        return segments


    def get_event_info(self, event_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a known GW event."""
        if event_name not in self.known_events:
            return None
        
        gps_time = self.known_events[event_name]
        
        # Additional info for well-known events
        event_info = {
            'name': event_name,
            'gps_time': gps_time,
            'type': 'binary_merger',
            'confirmed': True
        }
        
        # Special cases
        if event_name == 'GW170817':
            event_info['type'] = 'binary_neutron_star'
            event_info['multi_messenger'] = True
        
        return event_info


    def list_available_events(self) -> List[str]:
        """Get list of available event names."""
        return list(self.known_events.keys())


def create_quality_metrics(is_valid: bool = True, 
                          snr_estimate: float = 10.0,
                          **kwargs) -> QualityMetrics:
    """Create QualityMetrics with defaults."""
    defaults = {
        'glitch_probability': 0.1,
        'spectral_line_contamination': 0.05,
        'data_completeness': 1.0,
        'outlier_fraction': 0.02
    }
    defaults.update(kwargs)
    
    return QualityMetrics(
        is_valid=is_valid,
        snr_estimate=snr_estimate,
        **defaults
    )


def validate_strain_data(strain_data: jnp.ndarray, 
                        expected_duration: float = 4.0,
                        expected_sample_rate: int = 4096) -> bool:
    """
    Validate strain data array.
    
    Args:
        strain_data: Strain data array
        expected_duration: Expected duration in seconds
        expected_sample_rate: Expected sample rate in Hz
        
    Returns:
        True if data is valid
    """
    try:
        # Check basic properties
        if strain_data is None or strain_data.size == 0:
            logger.error("Empty strain data")
            return False
        
        # Check for NaN or infinite values
        if not jnp.all(jnp.isfinite(strain_data)):
            logger.error("Strain data contains NaN or infinite values")
            return False
        
        # Check expected length
        expected_length = int(expected_duration * expected_sample_rate)
        if abs(len(strain_data) - expected_length) > expected_sample_rate:  # 1 second tolerance
            logger.warning(f"Strain data length {len(strain_data)} differs from "
                         f"expected {expected_length}")
        
        # Check amplitude range (typical LIGO strain)
        strain_rms = jnp.sqrt(jnp.mean(strain_data**2))
        if strain_rms < 1e-25 or strain_rms > 1e-18:
            logger.warning(f"Unusual strain RMS: {strain_rms:.2e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Strain data validation failed: {e}")
        return False 