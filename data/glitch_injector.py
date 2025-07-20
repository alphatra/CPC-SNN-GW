"""
Glitch Injection System for LIGO CPC+SNN
Implements real and synthetic disturbance augmentation as identified in Executive Summary.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod

@dataclass
class GlitchParameters:
    """Configuration for different glitch types based on Gravity Spy catalog"""
    # Blip glitches - short duration transients
    blip_amplitude_range: Tuple[float, float] = (1e-23, 5e-22)
    blip_duration_range: Tuple[float, float] = (0.01, 0.1)  # 10ms to 100ms
    blip_central_freq_range: Tuple[float, float] = (50.0, 500.0)
    
    # Whistle glitches - frequency sweeps  
    whistle_amplitude_range: Tuple[float, float] = (5e-23, 2e-22)
    whistle_duration_range: Tuple[float, float] = (0.1, 1.0)  # 100ms to 1s
    whistle_freq_start_range: Tuple[float, float] = (100.0, 300.0)
    whistle_freq_end_range: Tuple[float, float] = (200.0, 800.0)
    
    # Scattered light glitches - narrow band lines
    scattered_amplitude_range: Tuple[float, float] = (1e-22, 8e-22)
    scattered_duration_range: Tuple[float, float] = (0.5, 2.0)
    scattered_freq_range: Tuple[float, float] = (60.0, 1000.0)
    scattered_q_factor_range: Tuple[float, float] = (10.0, 100.0)
    
    # Power line harmonics
    power_line_freqs: List[float] = None
    power_line_amplitude_range: Tuple[float, float] = (1e-22, 3e-22)
    
    def __post_init__(self):
        if self.power_line_freqs is None:
            # 60Hz harmonics common in US detectors
            self.power_line_freqs = [60.0, 120.0, 180.0, 240.0, 300.0, 360.0]

class GlitchGenerator(ABC):
    """Abstract base class for glitch generation"""
    
    @abstractmethod
    def generate(self, 
                key: jnp.ndarray,
                duration: float,
                sample_rate: int,
                **kwargs) -> jnp.ndarray:
        """Generate glitch time series"""
        pass

class BlipGlitchGenerator(GlitchGenerator):
    """Generates short-duration transient glitches similar to Gravity Spy blips"""
    
    def __init__(self, params: GlitchParameters):
        self.params = params
    
    def generate(self, 
                key: jnp.ndarray,
                duration: float, 
                sample_rate: int,
                **kwargs) -> jnp.ndarray:
        """Generate blip glitch - short Gaussian-modulated sinusoid"""
        key1, key2, key3, key4 = random.split(key, 4)
        
        # Random parameters
        amplitude = random.uniform(
            key1, (), *self.params.blip_amplitude_range
        )
        glitch_duration = random.uniform(
            key2, (), *self.params.blip_duration_range  
        )
        central_freq = random.uniform(
            key3, (), *self.params.blip_central_freq_range
        )
        
        # Random placement in segment
        start_time = random.uniform(key4, (), 0.0, duration - glitch_duration)
        
        # Generate time vector
        t = jnp.linspace(0, duration, int(duration * sample_rate))
        
        # Gaussian envelope
        t_center = start_time + glitch_duration / 2
        sigma = glitch_duration / 6  # 6-sigma duration
        envelope = jnp.exp(-0.5 * ((t - t_center) / sigma) ** 2)
        
        # Modulated sinusoid
        phase = 2 * jnp.pi * central_freq * (t - start_time)
        carrier = jnp.sin(phase)
        
        # Apply envelope only in glitch duration
        mask = (t >= start_time) & (t <= start_time + glitch_duration)
        glitch = amplitude * envelope * carrier * mask
        
        return glitch

class WhistleGlitchGenerator(GlitchGenerator):
    """Generates frequency-sweeping whistles"""
    
    def __init__(self, params: GlitchParameters):
        self.params = params
    
    def generate(self,
                key: jnp.ndarray,
                duration: float,
                sample_rate: int,
                **kwargs) -> jnp.ndarray:
        """Generate whistle glitch - frequency sweep"""
        key1, key2, key3, key4, key5 = random.split(key, 5)
        
        amplitude = random.uniform(
            key1, (), *self.params.whistle_amplitude_range
        )
        glitch_duration = random.uniform(
            key2, (), *self.params.whistle_duration_range
        )
        freq_start = random.uniform(
            key3, (), *self.params.whistle_freq_start_range
        )
        freq_end = random.uniform(
            key4, (), *self.params.whistle_freq_end_range
        )
        start_time = random.uniform(key5, (), 0.0, duration - glitch_duration)
        
        t = jnp.linspace(0, duration, int(duration * sample_rate))
        
        # Linear frequency sweep
        t_normalized = (t - start_time) / glitch_duration
        t_clipped = jnp.clip(t_normalized, 0.0, 1.0)
        freq_instantaneous = freq_start + (freq_end - freq_start) * t_clipped
        
        # Integrated phase for frequency sweep
        phase = 2 * jnp.pi * jnp.cumsum(freq_instantaneous) / sample_rate
        
        # Smooth envelope
        envelope = jnp.where(
            (t >= start_time) & (t <= start_time + glitch_duration),
            jnp.sin(jnp.pi * t_clipped) ** 2,  # Smooth rise/fall
            0.0
        )
        
        glitch = amplitude * envelope * jnp.sin(phase)
        return glitch

class ScatteredLightGlitchGenerator(GlitchGenerator):
    """Generates narrow-band scattered light glitches"""
    
    def __init__(self, params: GlitchParameters):
        self.params = params
    
    def generate(self,
                key: jnp.ndarray, 
                duration: float,
                sample_rate: int,
                **kwargs) -> jnp.ndarray:
        """Generate scattered light glitch - narrow band line"""
        key1, key2, key3, key4, key5 = random.split(key, 5)
        
        amplitude = random.uniform(
            key1, (), *self.params.scattered_amplitude_range
        )
        glitch_duration = random.uniform(
            key2, (), *self.params.scattered_duration_range
        )
        central_freq = random.uniform(
            key3, (), *self.params.scattered_freq_range
        )
        q_factor = random.uniform(
            key4, (), *self.params.scattered_q_factor_range
        )
        start_time = random.uniform(key5, (), 0.0, duration - glitch_duration)
        
        t = jnp.linspace(0, duration, int(duration * sample_rate))
        
        # Exponential decay envelope (characteristic of scattered light)
        decay_time = glitch_duration / 3
        envelope = jnp.where(
            (t >= start_time) & (t <= start_time + glitch_duration),
            jnp.exp(-(t - start_time) / decay_time),
            0.0
        )
        
        # Narrow-band sinusoid with phase noise
        phase_noise = random.normal(key4, t.shape) * 0.1
        phase = 2 * jnp.pi * central_freq * t + phase_noise
        
        glitch = amplitude * envelope * jnp.sin(phase)
        return glitch

class PowerLineGlitchGenerator(GlitchGenerator):
    """Generates power line harmonics"""
    
    def __init__(self, params: GlitchParameters):
        self.params = params
    
    def generate(self,
                key: jnp.ndarray,
                duration: float, 
                sample_rate: int,
                **kwargs) -> jnp.ndarray:
        """Generate power line glitch - 60Hz harmonics"""
        key1, key2 = random.split(key, 2)
        
        amplitude = random.uniform(
            key1, (), *self.params.power_line_amplitude_range
        )
        
        # Select random harmonic
        freq_idx = random.randint(key2, (), 0, len(self.params.power_line_freqs))
        frequency = self.params.power_line_freqs[freq_idx]
        
        t = jnp.linspace(0, duration, int(duration * sample_rate))
        
        # Continuous sinusoid with slow amplitude modulation
        mod_freq = 0.1  # 0.1 Hz modulation
        envelope = 1.0 + 0.3 * jnp.sin(2 * jnp.pi * mod_freq * t)
        
        phase = 2 * jnp.pi * frequency * t
        glitch = amplitude * envelope * jnp.sin(phase)
        
        return glitch

class GlitchInjector:
    """
    Main glitch injection system for LIGO data augmentation.
    
    As recommended in Executive Summary:
    - Adds real or synthetic disturbances to training samples
    - Teaches model to distinguish detector artifacts from astrophysical signals
    - Critical for achieving >80% accuracy target
    """
    
    def __init__(self, 
                 glitch_params: Optional[GlitchParameters] = None,
                 injection_probability: float = 0.3):
        """
        Initialize glitch injector
        
        Args:
            glitch_params: Configuration for glitch generation
            injection_probability: Probability of injecting glitch per sample
        """
        self.params = glitch_params or GlitchParameters()
        self.injection_probability = injection_probability
        
        # Initialize generators
        self.generators = {
            'blip': BlipGlitchGenerator(self.params),
            'whistle': WhistleGlitchGenerator(self.params), 
            'scattered': ScatteredLightGlitchGenerator(self.params),
            'powerline': PowerLineGlitchGenerator(self.params)
        }
        
        self.glitch_types = list(self.generators.keys())
    
    def inject_glitch(self,
                     strain_data: jnp.ndarray,
                     key: jnp.ndarray,
                     duration: float = 4.0,
                     sample_rate: int = 4096,
                     target_snr: Optional[float] = None) -> Tuple[jnp.ndarray, Dict]:
        """
        Inject random glitch into strain data
        
        Args:
            strain_data: Input strain time series [N_samples]
            key: JAX random key
            duration: Segment duration in seconds
            sample_rate: Sampling rate in Hz
            target_snr: Optional target SNR for glitch injection
            
        Returns:
            Tuple of (augmented_strain, glitch_metadata)
        """
        key1, key2, key3 = random.split(key, 3)
        
        # Decision: inject glitch or not
        inject = random.uniform(key1) < self.injection_probability
        
        if not inject:
            return strain_data, {'glitch_injected': False}
        
        # Select random glitch type
        glitch_idx = random.randint(key2, (), 0, len(self.glitch_types))
        glitch_type = self.glitch_types[glitch_idx]
        generator = self.generators[glitch_type]
        
        # Generate glitch
        glitch = generator.generate(
            key3, duration, sample_rate
        )
        
        # SNR-based scaling if requested
        if target_snr is not None:
            noise_power = jnp.var(strain_data)
            glitch_power = jnp.var(glitch)
            scale_factor = jnp.sqrt(target_snr * noise_power / glitch_power)
            glitch = glitch * scale_factor
        
        # Inject glitch
        augmented_strain = strain_data + glitch
        
        metadata = {
            'glitch_injected': True,
            'glitch_type': glitch_type,
            'glitch_snr': float(jnp.sqrt(jnp.var(glitch) / jnp.var(strain_data))),
            'injection_time': None  # Could be extracted from generator
        }
        
        return augmented_strain, metadata
    
    def batch_inject(self,
                    strain_batch: jnp.ndarray,
                    key: jnp.ndarray,
                    **kwargs) -> Tuple[jnp.ndarray, List[Dict]]:
        """
        Batch inject glitches for efficient training
        
        Args:
            strain_batch: Batch of strain data [batch_size, N_samples]
            key: JAX random key
            
        Returns:
            Tuple of (augmented_batch, metadata_list)
        """
        batch_size = strain_batch.shape[0]
        keys = random.split(key, batch_size)
        
        # Vectorized injection
        def inject_single(strain, key):
            return self.inject_glitch(strain, key, **kwargs)
        
        # JAX vmap for efficient batch processing
        batch_inject_fn = jax.vmap(inject_single, in_axes=(0, 0))
        augmented_batch, metadata_batch = batch_inject_fn(strain_batch, keys)
        
        # Convert metadata to list of dicts
        metadata_list = []
        for i in range(batch_size):
            metadata = {k: v[i] if hasattr(v, '__getitem__') else v 
                       for k, v in metadata_batch.items()}
            metadata_list.append(metadata)
        
        return augmented_batch, metadata_list
    
    def get_statistics(self) -> Dict:
        """Get statistics about configured glitch types"""
        return {
            'available_types': self.glitch_types,
            'injection_probability': self.injection_probability,
            'blip_freq_range': self.params.blip_central_freq_range,
            'whistle_freq_range': (
                self.params.whistle_freq_start_range,
                self.params.whistle_freq_end_range
            ),
            'scattered_freq_range': self.params.scattered_freq_range,
            'powerline_freqs': self.params.power_line_freqs
        }

# Factory function for easy instantiation
def create_ligo_glitch_injector(injection_probability: float = 0.3) -> GlitchInjector:
    """
    Factory function to create LIGO-specific glitch injector
    
    Args:
        injection_probability: Fraction of samples to augment with glitches
        
    Returns:
        Configured GlitchInjector instance
    """
    # LIGO-specific parameters based on Gravity Spy characterization
    ligo_params = GlitchParameters(
        blip_amplitude_range=(1e-23, 5e-22),
        blip_duration_range=(0.01, 0.1),
        blip_central_freq_range=(50.0, 500.0),
        
        whistle_amplitude_range=(5e-23, 2e-22), 
        whistle_duration_range=(0.1, 1.0),
        whistle_freq_start_range=(100.0, 300.0),
        whistle_freq_end_range=(200.0, 800.0),
        
        scattered_amplitude_range=(1e-22, 8e-22),
        scattered_duration_range=(0.5, 2.0),
        scattered_freq_range=(60.0, 1000.0),
        scattered_q_factor_range=(10.0, 100.0),
        
        power_line_freqs=[60.0, 120.0, 180.0, 240.0, 300.0, 360.0],
        power_line_amplitude_range=(1e-22, 3e-22)
    )
    
    return GlitchInjector(ligo_params, injection_probability) 