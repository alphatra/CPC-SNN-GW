"""
GW Signal Parameters: Dataclasses and Configuration
Extracted from continuous_gw_generator.py for modular architecture.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import jax.numpy as jnp

logger = logging.getLogger(__name__)

# Check PyFstat availability
try:
    import pyfstat
    PYFSTAT_AVAILABLE = True
    logger.info("PyFstat available - enhanced F-statistic features enabled")
except ImportError:
    PYFSTAT_AVAILABLE = False
    # Downgrade to INFO to avoid noisy warnings in normal runs; still visible at INFO once.
    logger.info("PyFstat not available - using fallback synthetic generation")


@dataclass
class ContinuousGWParams:
    """Enhanced parameters for continuous gravitational wave signals."""
    frequency: float = 50.0  # Hz
    frequency_dot: float = -1e-10  # Hz/s (frequency derivative)
    alpha: float = 0.0  # right ascension (rad)
    delta: float = 0.0  # declination (rad)
    amplitude_h0: float = 1e-23  # strain amplitude
    cosi: float = 0.0  # cosine of inclination angle
    psi: float = 0.0  # polarization angle (rad)
    phi0: float = 0.0  # initial phase (rad)
    include_doppler: bool = True  # include Doppler modulation
    
    # Detector-specific parameters (default to LIGO Hanford)
    detector_latitude: float = 0.6370  # rad (Hanford latitude)
    detector_longitude: float = -2.0847  # rad (Hanford longitude)
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.frequency <= 0:
            raise ValueError(f"Frequency must be positive, got {self.frequency}")
        if not -1.0 <= self.cosi <= 1.0:
            raise ValueError(f"cos(i) must be in [-1,1], got {self.cosi}")
        if not 0.0 <= self.psi <= jnp.pi:
            raise ValueError(f"Polarization angle must be in [0,π], got {self.psi}")
        if not 0.0 <= self.phi0 <= 2*jnp.pi:
            raise ValueError(f"Initial phase must be in [0,2π], got {self.phi0}")
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'frequency': self.frequency,
            'frequency_dot': self.frequency_dot,
            'alpha': self.alpha,
            'delta': self.delta,
            'amplitude_h0': self.amplitude_h0,
            'cosi': self.cosi,
            'psi': self.psi,
            'phi0': self.phi0,
            'include_doppler': self.include_doppler,
            'detector_latitude': self.detector_latitude,
            'detector_longitude': self.detector_longitude
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContinuousGWParams':
        """Create from dictionary."""
        return cls(**data)


@dataclass 
class SignalConfiguration:
    """Enhanced configuration for signal generation."""
    base_frequency: float = 50.0  # Hz
    freq_range: Tuple[float, float] = (20.0, 200.0)  # Hz
    duration: float = 1000.0  # seconds
    sampling_rate: int = 4096  # Hz
    include_doppler: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.base_frequency <= 0:
            raise ValueError(f"Base frequency must be positive, got {self.base_frequency}")
        if self.duration <= 0:
            raise ValueError(f"Duration must be positive, got {self.duration}")
        if self.freq_range[0] <= 0 or self.freq_range[1] <= self.freq_range[0]:
            raise ValueError(f"Invalid frequency range, got {self.freq_range}")
        if self.sampling_rate <= 0:
            raise ValueError(f"Sampling rate must be positive, got {self.sampling_rate}")


@dataclass
class GeneratorSettings:
    """Settings for continuous GW signal generator."""
    num_signals: int = 100
    signal_duration: float = 4.0  # seconds
    noise_level: float = 1e-23
    include_noise_only: bool = True
    train_split: float = 0.7
    validation_split: float = 0.2
    test_split: float = 0.1
    
    def __post_init__(self):
        """Validate generator settings."""
        if self.num_signals <= 0:
            raise ValueError(f"Number of signals must be positive, got {self.num_signals}")
        if self.signal_duration <= 0:
            raise ValueError(f"Signal duration must be positive, got {self.signal_duration}")
        if self.noise_level < 0:
            raise ValueError(f"Noise level cannot be negative, got {self.noise_level}")
            
        # Validate splits sum to 1.0
        total_split = self.train_split + self.validation_split + self.test_split
        if not 0.99 <= total_split <= 1.01:  # Allow small floating point errors
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")


@dataclass
class PhysicsConstants:
    """Physical constants for GW signal generation."""
    LIGHT_SPEED: float = 299792458.0  # m/s
    EARTH_RADIUS: float = 6.371e6  # m
    ORBITAL_RADIUS: float = 1.496e11  # m (1 AU)
    ORBITAL_PERIOD: float = 365.25 * 24 * 3600  # seconds
    SIDEREAL_RATE: float = 7.292115e-5  # rad/s
    
    @property
    def orbital_velocity(self) -> float:
        """Earth's orbital velocity around the Sun."""
        return 2 * jnp.pi * self.ORBITAL_RADIUS / self.ORBITAL_PERIOD
    
    @property
    def rotation_velocity_factor(self) -> float:
        """Factor for Earth rotation velocity calculation."""
        return 2 * jnp.pi * self.EARTH_RADIUS / (24 * 3600)


# Global physics constants instance
PHYSICS = PhysicsConstants()


def create_default_params() -> ContinuousGWParams:
    """Create default continuous GW parameters."""
    return ContinuousGWParams()


def create_hanford_params(**kwargs) -> ContinuousGWParams:
    """Create parameters for LIGO Hanford detector."""
    params = {
        'detector_latitude': 0.6370,  # rad
        'detector_longitude': -2.0847,  # rad
        **kwargs
    }
    return ContinuousGWParams(**params)


def create_livingston_params(**kwargs) -> ContinuousGWParams:
    """Create parameters for LIGO Livingston detector."""
    params = {
        'detector_latitude': 0.5335,  # rad  
        'detector_longitude': -1.5843,  # rad
        **kwargs
    }
    return ContinuousGWParams(**params)


def create_virgo_params(**kwargs) -> ContinuousGWParams:
    """Create parameters for Virgo detector."""
    params = {
        'detector_latitude': 0.7648,  # rad
        'detector_longitude': 0.1833,  # rad
        **kwargs
    }
    return ContinuousGWParams(**params)


def validate_signal_params(params: ContinuousGWParams) -> bool:
    """Validate signal parameters for physical consistency."""
    try:
        # Basic parameter validation is done in __post_init__
        
        # Additional physics validation
        if params.amplitude_h0 <= 0:
            logger.error(f"Amplitude must be positive, got {params.amplitude_h0}")
            return False
            
        if params.amplitude_h0 > 1e-20:
            logger.warning(f"Very large amplitude: {params.amplitude_h0}. "
                         "This may not be physically realistic.")
        
        if abs(params.frequency_dot) > 1e-6:
            logger.warning(f"Large frequency derivative: {params.frequency_dot}. "
                         "This may indicate rapid evolution.")
        
        # Frequency bounds check
        if not 10.0 <= params.frequency <= 2000.0:
            logger.warning(f"Frequency {params.frequency} Hz outside typical "
                         "LIGO sensitivity range [10, 2000] Hz")
        
        return True
        
    except Exception as e:
        logger.error(f"Parameter validation failed: {e}")
        return False


def get_detector_params(detector_name: str) -> Dict[str, float]:
    """Get detector-specific parameters."""
    detectors = {
        'H1': {'latitude': 0.6370, 'longitude': -2.0847},  # Hanford
        'L1': {'latitude': 0.5335, 'longitude': -1.5843},  # Livingston  
        'V1': {'latitude': 0.7648, 'longitude': 0.1833},   # Virgo
    }
    
    if detector_name not in detectors:
        raise ValueError(f"Unknown detector: {detector_name}. "
                        f"Available: {list(detectors.keys())}")
    
    return detectors[detector_name] 