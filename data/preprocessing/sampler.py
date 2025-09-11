"""
Intelligent segment sampler for gravitational wave data.

This module contains the SegmentSampler class extracted from
gw_preprocessor.py for better modularity.

Split from gw_preprocessor.py for better maintainability.
"""

import jax
import jax.numpy as jnp
from typing import List, Optional, Tuple
import logging

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


# Export sampler class
__all__ = [
    "SegmentSampler"
]

