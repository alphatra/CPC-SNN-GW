"""
ReadLIGO Downloader - Production LIGO Data Downloader
Replaces problematic GWOSC/gwpy with reliable ReadLIGO library.
Based on working solution from real_ligo_test.py and readligo_data_sources.py
"""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Optional, Any, Dict
import logging
import time
import hashlib
from pathlib import Path
import numpy as np
import os

from .cache_manager import ProfessionalCacheManager
from .readligo_data_sources import ProcessingResult, QualityMetrics, ReadLIGODataFetcher

logger = logging.getLogger(__name__)

# Try to import ReadLIGO - our proven solution
try:
    import readligo as rl
    HAS_READLIGO = True
    logger.info("✅ ReadLIGO available - using real LIGO data")
except ImportError:
    HAS_READLIGO = False
    logger.error("❌ ReadLIGO not available - install with: pip install readligo")

def _safe_jax_cpu_context():
    """Safe CPU context optimized for Apple Silicon and older JAX versions."""
    import os
    
    # Apple Silicon optimization
    if hasattr(jax, 'default_device'):
        # Set XLA flags for Apple Silicon
        os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=1')
    
    try:
        # JAX >= 0.4.25
        return jax.default_device(jax.devices('cpu')[0])
    except (AttributeError, IndexError):
        # Fallback for older JAX versions or no CPU device
        class _DummyContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return _DummyContext()

def _safe_array_to_jax(array: np.ndarray) -> jnp.ndarray:
    """Safely convert numpy array to JAX array with Apple Silicon optimization."""
    try:
        with _safe_jax_cpu_context():
            return jnp.array(array, dtype=jnp.float32)
    except Exception as e:
        logger.warning(f"JAX conversion failed, using numpy: {e}")
        return jnp.array(np.array(array, dtype=np.float32))

def _compute_kurtosis(data: jnp.ndarray) -> float:
    """Compute kurtosis with numerical stability."""
    try:
        normalized = (data - jnp.mean(data)) / (jnp.std(data) + 1e-10)
        return float(jnp.mean(normalized**4) - 3.0)
    except Exception:
        return 0.0

def _generate_cache_key(detector: str, start_time: int, duration: float) -> str:
    """Generate unique cache key for data segment."""
    key_string = f"{detector}_{start_time}_{duration}"
    return hashlib.sha256(key_string.encode()).hexdigest()[:16]

class ReadLIGODownloader:
    """
    ✅ PRODUCTION ReadLIGO Downloader - WORKING SOLUTION
    Uses proven ReadLIGO library instead of problematic GWOSC/gwpy.
    Based on successful implementation from real_ligo_test.py
    """
    
    def __init__(self, 
                 sample_rate: int = 4096,
                 cache_manager: Optional[ProfessionalCacheManager] = None,
                 max_retries: int = 3,
                 base_wait: float = 1.0,
                 timeout: int = 30):
        """
        Initialize ReadLIGO downloader.
        
        Args:
            sample_rate: Target sample rate in Hz
            cache_manager: Professional cache manager instance
            max_retries: Maximum number of retry attempts
            base_wait: Base wait time for exponential backoff
            timeout: Request timeout in seconds
        """
        if not HAS_READLIGO:
            raise ImportError("ReadLIGO not available. Install with: pip install readligo")
        
        self.sample_rate = sample_rate
        self.cache_manager = cache_manager or ProfessionalCacheManager()
        self.max_retries = max_retries
        self.base_wait = base_wait
        self.timeout = timeout
        
        # Initialize ReadLIGO fetcher
        self.readligo_fetcher = ReadLIGODataFetcher()
        
        logger.info(f"✅ ReadLIGO downloader initialized: {sample_rate} Hz, {max_retries} retries")
        logger.info(f"   Available events: {self.readligo_fetcher.get_available_events()}")
    
    def fetch(self, detector: str, start_time: int, duration: float, 
              event_name: Optional[str] = None) -> jnp.ndarray:
        """
        ✅ WORKING: Fetch strain data using ReadLIGO with professional caching.
        
        Args:
            detector: Detector name ('H1', 'L1')
            start_time: GPS start time (or event context)
            duration: Duration in seconds
            event_name: Specific event name (e.g., 'GW150914')
            
        Returns:
            Strain data as JAX array
            
        Raises:
            RuntimeError: If download fails after all retries
        """
        # Generate cache key
        cache_key = _generate_cache_key(detector, start_time, duration)
        
        # Try cache first
        try:
            cached_data = self.cache_manager.get(cache_key)
            if cached_data is not None:
                logger.debug(f"✅ Cache hit for {detector} {start_time}")
                return cached_data
        except Exception as e:
            logger.debug(f"Cache access failed: {e}")
        
        # Determine event name if not provided
        if event_name is None:
            # Try to match GPS time to known events
            for event in self.readligo_fetcher.get_available_events():
                event_info = self.readligo_fetcher.available_files[event]
                if abs(event_info['gps'] - start_time) < duration:
                    event_name = event
                    logger.info(f"✅ Matched GPS time {start_time} to event {event_name}")
                    break
            
            if event_name is None:
                logger.error(f"❌ No event found for GPS time {start_time}")
                raise RuntimeError(f"No event data available for GPS time {start_time}")
        
        # Download with retry logic
        for attempt in range(self.max_retries):
            try:
                logger.info(f"📥 Fetching {detector} data: {event_name}, duration {duration}s")
                
                # ✅ PROVEN METHOD: Use ReadLIGO fetcher
                event_data = self.readligo_fetcher.fetch_strain_data(
                    event_name=event_name,
                    detector=detector,
                    duration=int(duration),
                    sample_rate=self.sample_rate
                )
                
                if event_data is None:
                    raise RuntimeError(f"ReadLIGO fetch failed for {event_name}/{detector}")
                
                strain_array = event_data.strain
                
                # Cache the result
                try:
                    self.cache_manager.put(cache_key, strain_array, {
                        'detector': detector,
                        'start_time': start_time,
                        'duration': duration,
                        'sample_rate': self.sample_rate,
                        'event_name': event_name,
                        'download_time': time.time(),
                        'source': 'readligo',
                        'snr': event_data.snr
                    })
                except Exception as e:
                    logger.warning(f"Failed to cache data: {e}")
                
                logger.info(f"✅ Successfully fetched {len(strain_array)} samples, SNR={event_data.snr:.2f}")
                return strain_array
                
            except Exception as e:
                wait_time = self.base_wait * (2 ** attempt)
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to fetch data after {self.max_retries} attempts: {e}")
    
    def fetch_event_data(self, event_name: str, detector: str = 'H1', 
                        duration: int = 32) -> jnp.ndarray:
        """
        ✅ SIMPLIFIED: Fetch data for a specific known event.
        
        Args:
            event_name: Event name (e.g., 'GW150914')
            detector: Detector name ('H1', 'L1')
            duration: Duration in seconds
            
        Returns:
            Strain data as JAX array
        """
        if event_name not in self.readligo_fetcher.get_available_events():
            raise ValueError(f"Event {event_name} not available. Available: {self.readligo_fetcher.get_available_events()}")
        
        event_info = self.readligo_fetcher.available_files[event_name]
        gps_time = int(event_info['gps'])
        
        return self.fetch(detector, gps_time, duration, event_name)
    
    def fetch_batch(self, 
                   segments: List[Tuple[str, int, float]], 
                   max_workers: int = 2,
                   event_names: Optional[List[str]] = None) -> List[Optional[jnp.ndarray]]:
        """
        ✅ WORKING: Fetch multiple data segments in batch.
        
        Args:
            segments: List of (detector, start_time, duration) tuples
            max_workers: Maximum parallel download workers (reduced for ReadLIGO)
            event_names: Optional list of event names for each segment
            
        Returns:
            List of strain data arrays (None for failed fetches)
        """
        import concurrent.futures
        
        logger.info(f"📥 Batch fetching {len(segments)} segments with {max_workers} workers")
        
        if event_names is None:
            event_names = [None] * len(segments)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.fetch, detector, start_time, duration, event_name)
                for (detector, start_time, duration), event_name in zip(segments, event_names)
            ]
            
            results = []
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    data = future.result()
                    results.append(data)
                    logger.info(f"✅ Completed segment {i+1}/{len(segments)}")
                except Exception as e:
                    logger.error(f"❌ Failed to fetch segment {i+1}: {e}")
                    results.append(None)
                    
        return results
    
    def get_available_events(self) -> List[str]:
        """Get list of available events."""
        return self.readligo_fetcher.get_available_events()
    
    def get_detectors_for_event(self, event_name: str) -> List[str]:
        """Get available detectors for specific event."""
        return self.readligo_fetcher.get_detectors_for_event(event_name)
    
    def validate_data(self, strain: jnp.ndarray, detector: str) -> Dict[str, Any]:
        """
        Validate fetched strain data quality.
        
        Args:
            strain: Strain data array
            detector: Detector name
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Basic validation
            if len(strain) == 0:
                return {'valid': False, 'reason': 'Empty data'}
            
            if jnp.any(jnp.isnan(strain)) or jnp.any(jnp.isinf(strain)):
                return {'valid': False, 'reason': 'Contains NaN/Inf values'}
            
            # Quality metrics
            rms = float(jnp.sqrt(jnp.mean(strain**2)))
            peak = float(jnp.max(jnp.abs(strain)))
            kurtosis = _compute_kurtosis(strain)
            
            # SNR estimation
            signal_power = jnp.mean(strain**2)
            noise_estimate = jnp.std(jnp.diff(strain)) * jnp.sqrt(self.sample_rate/2)
            snr = float(jnp.sqrt(signal_power) / (noise_estimate + 1e-20))
            
            validation_result = {
                'valid': True,
                'detector': detector,
                'samples': len(strain),
                'rms': rms,
                'peak_amplitude': peak,
                'kurtosis': kurtosis,
                'snr_estimate': snr,
                'quality_score': min(1.0, snr / 10.0)  # Normalize to 0-1
            }
            
            logger.info(f"✅ Data validation passed: {detector}, SNR={snr:.2f}")
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Data validation failed: {e}")
            return {'valid': False, 'reason': f'Validation error: {e}'}
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics for monitoring."""
        return {
            'cache_hits': getattr(self.cache_manager, '_hits', 0),
            'cache_misses': getattr(self.cache_manager, '_misses', 0),
            'cache_size': len(getattr(self.cache_manager, '_cache', {})),
            'available_events': len(self.get_available_events()),
            'readligo_available': HAS_READLIGO
        }

# Legacy aliases for backward compatibility
ProductionReadLIGODownloader = ReadLIGODownloader

# Factory function
def create_readligo_downloader(sample_rate: int = 4096, 
                              cache_manager: Optional[ProfessionalCacheManager] = None) -> ReadLIGODownloader:
    """Create ReadLIGO downloader instance."""
    return ReadLIGODownloader(sample_rate=sample_rate, cache_manager=cache_manager) 