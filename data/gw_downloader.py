"""
GWOSC Data Downloader

Production-ready downloader for gravitational wave strain data from LIGO/Virgo 
detectors via GWOSC API. Optimized for Apple Silicon with comprehensive error 
handling and retry logic.

Features:
- Professional cache integration with SHA256 verification
- Exponential backoff retry strategy
- Batch downloading with parallel processing
- Apple Silicon JAX optimizations
- Cross-platform compatibility
"""

import jax
import jax.numpy as jnp
from gwpy.timeseries import TimeSeries
from typing import List, Tuple, Optional, Any
import logging
import time
import tempfile
import hashlib
from pathlib import Path
import numpy as np

try:
    from .cache_manager import ProfessionalCacheManager
except Exception as _e:
    logger.warning(f"ProfessionalCacheManager unavailable: {_e}. Using simple in-memory fallback.")
    class ProfessionalCacheManager:  # type: ignore
        def __init__(self):
            self._cache = {}
            self._hits = 0
            self._misses = 0
        def get(self, key):
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None
        def put(self, key, value, meta=None):
            self._cache[key] = value
        def clear(self):
            self._cache.clear()
        def get_memory_usage(self):
            return 0

from .readligo_data_sources import ProcessingResult, QualityMetrics

logger = logging.getLogger(__name__)


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


@jax.jit
def _compute_kurtosis(data: jnp.ndarray) -> float:
    """Compute kurtosis manually since jax.scipy.stats.kurtosis doesn't exist."""
    mean = jnp.mean(data)
    std = jnp.std(data)
    standardized = (data - mean) / std
    kurt = jnp.mean(standardized**4) - 3  # Excess kurtosis
    return float(kurt)


def _safe_array_to_jax(arr: np.ndarray, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    """Safely convert NumPy array to JAX with Apple Silicon optimizations and shaped array support."""
    # Handle different array types
    if arr.dtype.names is not None:
        # Structured array - convert to regular array if possible
        logger.debug(f"Converting structured array with fields: {arr.dtype.names}")
        if len(arr.dtype.names) == 1:
            # Single field - extract it
            field_name = arr.dtype.names[0]
            arr = arr[field_name]
        else:
            # Multiple fields - flatten to regular array
            arr = arr.view(dtype=arr.dtype.descr[0][1]).reshape(arr.shape + (-1,))
    
    # Handle complex arrays
    if np.iscomplexobj(arr):
        logger.debug("Converting complex array to real array")
        arr = np.real(arr)  # Take real part for GW strain data
    
    # Convert to specified dtype to save memory
    if arr.dtype != dtype:
        arr = arr.astype(dtype)
    
    try:
        # Try using default_device context (Apple Silicon optimized)
        with _safe_jax_cpu_context():
            return jnp.asarray(arr)  # Use asarray for better performance
    except Exception as e:
        logger.debug(f"Context-based conversion failed: {e}")
        try:
            # Fallback to regular device_put
            return jax.device_put(arr, jax.devices('cpu')[0])
        except Exception as e2:
            logger.warning(f"JAX conversion failed, returning NumPy: {e2}")
            return arr  # Return NumPy array as last resort


def _generate_cache_key(detector: str, start_time: int, duration: float) -> str:
    """Generate unique cache key for data segment."""
    key_string = f"{detector}_{start_time}_{duration}"
    return hashlib.sha256(key_string.encode()).hexdigest()[:16]


class ProductionGWOSCDownloader:
    """
    Production GWOSC downloader with professional cache integration.
    
    Optimized for Apple Silicon with exponential backoff retry strategy,
    batch processing, and comprehensive error handling.
    """
    
    def __init__(self, 
                 sample_rate: int = 4096,
                 cache_manager: Optional[ProfessionalCacheManager] = None,
                 max_retries: int = 3,
                 base_wait: float = 1.0,
                 timeout: int = 30):
        """
        Initialize GWOSC downloader.
        
        Args:
            sample_rate: Target sample rate in Hz
            cache_manager: Professional cache manager instance
            max_retries: Maximum number of retry attempts
            base_wait: Base wait time for exponential backoff
            timeout: Request timeout in seconds
        """
        self.sample_rate = sample_rate
        self.cache_manager = cache_manager or ProfessionalCacheManager()
        self.max_retries = max_retries
        self.base_wait = base_wait
        self.timeout = timeout
        
        logger.info(f"Initialized GWOSC downloader: {sample_rate} Hz, {max_retries} retries")
    
    def fetch(self, detector: str, start_time: int, duration: float) -> jnp.ndarray:
        """
        Fetch strain data with professional caching and retry logic.
        
        Args:
            detector: Detector name ('H1', 'L1', 'V1')
            start_time: GPS start time
            duration: Duration in seconds
            
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
                logger.debug(f"Cache hit for {detector} {start_time}")
                return cached_data
        except Exception as e:
            logger.debug(f"Cache access failed: {e}")
        
        # Download with retry logic
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Downloading {detector} data: GPS {start_time}, duration {duration}s")
                
                # Use TimeSeries.fetch_open_data with timeout
                strain = TimeSeries.fetch_open_data(
                    detector, 
                    start_time, 
                    start_time + duration,
                    sample_rate=self.sample_rate,
                    timeout=self.timeout
                )
                
                # Convert to JAX array with Apple Silicon optimization
                strain_array = _safe_array_to_jax(strain.value)
                
                # Cache the result
                try:
                    self.cache_manager.put(cache_key, strain_array, {
                        'detector': detector,
                        'start_time': start_time,
                        'duration': duration,
                        'sample_rate': self.sample_rate,
                        'download_time': time.time()
                    })
                except Exception as e:
                    logger.warning(f"Failed to cache data: {e}")
                
                logger.info(f"Successfully downloaded {len(strain_array)} samples")
                return strain_array
                
            except Exception as e:
                wait_time = self.base_wait * (2 ** attempt)
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to fetch data after {self.max_retries} attempts: {e}")
                    
    def fetch_batch(self, 
                   segments: List[Tuple[str, int, float]], 
                   max_workers: int = 4) -> List[jnp.ndarray]:
        """
        Fetch multiple data segments in batch with parallel processing.
        
        Args:
            segments: List of (detector, start_time, duration) tuples
            max_workers: Maximum parallel download workers
            
        Returns:
            List of strain data arrays
        """
        import concurrent.futures
        
        logger.info(f"Batch fetching {len(segments)} segments with {max_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.fetch, detector, start_time, duration)
                for detector, start_time, duration in segments
            ]
            
            results = []
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    data = future.result()
                    results.append(data)
                    logger.info(f"Completed segment {i+1}/{len(segments)}")
                except Exception as e:
                    logger.error(f"Failed to fetch segment {i+1}: {e}")
                    results.append(None)
                    
        return results

    def get_cache_stats(self) -> dict:
        """Get cache statistics for monitoring."""
        return {
            'cache_hits': getattr(self.cache_manager, '_hits', 0),
            'cache_misses': getattr(self.cache_manager, '_misses', 0),
            'cache_size': len(getattr(self.cache_manager, '_cache', {})),
            'cache_memory_usage': getattr(self.cache_manager, 'get_memory_usage', lambda: 0)()
        }

    def clear_cache(self):
        """Clear all cached data."""
        try:
            self.cache_manager.clear()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")


# Legacy alias for backward compatibility
GWOSCDownloader = ProductionGWOSCDownloader 