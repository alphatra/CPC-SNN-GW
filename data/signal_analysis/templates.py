"""
Template generation for gravitational wave signal analysis.

Provides functions to create signal templates for matched filtering
and template bank construction.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TemplateParameters:
    """Parameters for gravitational wave template generation."""
    mass1: float  # Primary mass in solar masses
    mass2: float  # Secondary mass in solar masses
    spin1z: float = 0.0  # Primary spin (z-component)
    spin2z: float = 0.0  # Secondary spin (z-component)
    distance: float = 100.0  # Distance in Mpc
    inclination: float = 0.0  # Inclination angle in radians
    phase: float = 0.0  # Orbital phase in radians
    

def create_chirp_template(
    params: TemplateParameters,
    sample_rate: int = 4096,
    duration: float = 4.0,
    low_freq: float = 20.0
) -> jnp.ndarray:
    """
    ✅ PROFESSIONAL: Create gravitational wave chirp template
    
    Generates a simplified but realistic chirp template using
    post-Newtonian approximation for the frequency evolution.
    
    Args:
        params: Template parameters
        sample_rate: Sample rate in Hz
        duration: Template duration in seconds
        low_freq: Starting frequency in Hz
        
    Returns:
        Chirp template as JAX array
    """
    # ✅ PHYSICAL PARAMETERS
    total_mass = params.mass1 + params.mass2
    chirp_mass = ((params.mass1 * params.mass2) ** (3/5)) / (total_mass ** (1/5))
    symmetric_mass_ratio = (params.mass1 * params.mass2) / (total_mass ** 2)
    
    # Convert to SI units
    solar_mass = 1.989e30  # kg
    c = 2.998e8  # m/s
    G = 6.674e-11  # m^3 kg^-1 s^-2
    
    chirp_mass_si = chirp_mass * solar_mass
    
    # ✅ TIME ARRAY
    n_samples = int(sample_rate * duration)
    time_array = jnp.linspace(0, duration, n_samples)
    
    # ✅ POST-NEWTONIAN FREQUENCY EVOLUTION
    # Simplified 2.5PN frequency evolution
    # f(t) = f0 * (1 - t/t_coalescence)^(-3/8)
    
    # Coalescence time estimate
    t_coal_factor = (5 * c**5) / (256 * G**(5/3) * (jnp.pi * low_freq)**(8/3) * chirp_mass_si**(5/3))
    
    # Frequency evolution (simplified)
    f_factor = (96 * jnp.pi / 5) * (G * chirp_mass_si / c**3)**(5/3)
    tau = t_coal_factor - time_array
    tau = jnp.maximum(tau, 1e-6)  # Avoid division by zero
    
    frequency = low_freq * (tau / t_coal_factor)**(-3/8)
    frequency = jnp.clip(frequency, low_freq, sample_rate / 2)
    
    # ✅ PHASE EVOLUTION
    # Integrate frequency to get phase
    dt = duration / n_samples
    phase_evolution = jnp.cumsum(2 * jnp.pi * frequency * dt) + params.phase
    
    # ✅ AMPLITUDE EVOLUTION
    # Simplified amplitude that increases towards coalescence
    amplitude_factor = (tau / t_coal_factor)**(-1/4)
    amplitude_factor = jnp.clip(amplitude_factor, 1.0, 10.0)  # Reasonable range
    
    # Distance scaling
    distance_factor = 100.0 / params.distance  # Normalize to 100 Mpc
    
    # ✅ WAVEFORM GENERATION
    # Plus polarization (simplified)
    h_plus = distance_factor * amplitude_factor * (1 + jnp.cos(params.inclination)**2) / 2
    
    # Cross polarization
    h_cross = distance_factor * amplitude_factor * jnp.cos(params.inclination)
    
    # Combine polarizations (simplified detector response)
    waveform = h_plus * jnp.cos(phase_evolution) + h_cross * jnp.sin(phase_evolution)
    
    # ✅ WINDOWING: Apply smooth window to avoid edge effects
    # Tukey window (10% taper)
    taper_fraction = 0.1
    taper_samples = int(taper_fraction * n_samples)
    
    window = jnp.ones(n_samples)
    
    # Left taper
    left_taper = 0.5 * (1 + jnp.cos(jnp.pi * (jnp.arange(taper_samples) / taper_samples - 1)))
    window = window.at[:taper_samples].set(left_taper)
    
    # Right taper  
    right_taper = 0.5 * (1 + jnp.cos(jnp.pi * jnp.arange(taper_samples) / taper_samples))
    window = window.at[-taper_samples:].set(right_taper)
    
    # Apply window
    waveform = waveform * window
    
    # ✅ NORMALIZATION: Unit norm for matched filtering
    norm = jnp.sqrt(jnp.sum(waveform**2))
    if norm > 1e-12:
        waveform = waveform / norm
    
    logger.debug(f"✅ Created chirp template: M={total_mass:.1f}M☉, f0={low_freq}Hz")
    
    return waveform


def create_template_bank(
    mass_range: Tuple[float, float] = (10.0, 50.0),
    num_templates: int = 10,
    sample_rate: int = 4096,
    duration: float = 4.0,
    low_freq: float = 20.0,
    random_seed: int = 42
) -> List[jnp.ndarray]:
    """
    ✅ PROFESSIONAL: Create bank of gravitational wave templates
    
    Generates a diverse set of templates covering the specified
    mass range and other parameter variations.
    
    Args:
        mass_range: Range of component masses in solar masses
        num_templates: Number of templates to generate
        sample_rate: Sample rate in Hz
        duration: Template duration in seconds
        low_freq: Starting frequency in Hz
        random_seed: Random seed for reproducibility
        
    Returns:
        List of template waveforms
    """
    rng = np.random.RandomState(random_seed)
    templates = []
    
    for i in range(num_templates):
        # ✅ RANDOM PARAMETER SAMPLING
        # Component masses
        mass1 = rng.uniform(mass_range[0], mass_range[1])
        mass2 = rng.uniform(mass_range[0], mass1)  # m2 <= m1
        
        # Spins (aligned, moderate values)
        spin1z = rng.uniform(-0.5, 0.5)
        spin2z = rng.uniform(-0.5, 0.5)
        
        # Distance (log-uniform distribution)
        distance = rng.lognormal(np.log(100), 0.5)  # Around 100 Mpc
        distance = np.clip(distance, 50, 1000)
        
        # Inclination (isotropic distribution)
        inclination = np.arccos(rng.uniform(-1, 1))
        
        # Random phase
        phase = rng.uniform(0, 2*np.pi)
        
        # ✅ CREATE TEMPLATE
        params = TemplateParameters(
            mass1=mass1,
            mass2=mass2,
            spin1z=spin1z,
            spin2z=spin2z,
            distance=distance,
            inclination=inclination,
            phase=phase
        )
        
        template = create_chirp_template(
            params=params,
            sample_rate=sample_rate,
            duration=duration,
            low_freq=low_freq
        )
        
        templates.append(template)
    
    logger.info(f"✅ Created template bank with {num_templates} templates")
    logger.info(f"   Mass range: {mass_range[0]}-{mass_range[1]} M☉")
    logger.info(f"   Duration: {duration}s, Sample rate: {sample_rate}Hz")
    
    return templates


def create_noise_template(
    sample_rate: int = 4096,
    duration: float = 4.0,
    random_seed: int = 42
) -> jnp.ndarray:
    """
    ✅ UTILITY: Create noise template for testing
    
    Generates colored noise template that mimics LIGO noise characteristics.
    
    Args:
        sample_rate: Sample rate in Hz
        duration: Template duration in seconds
        random_seed: Random seed for reproducibility
        
    Returns:
        Noise template
    """
    rng = np.random.RandomState(random_seed)
    n_samples = int(sample_rate * duration)
    
    # ✅ WHITE NOISE BASE
    white_noise = rng.normal(0, 1, n_samples)
    
    # ✅ FREQUENCY DOMAIN COLORING
    noise_fft = jnp.fft.rfft(white_noise)
    frequencies = jnp.fft.rfftfreq(n_samples, 1.0 / sample_rate)
    
    # Simple LIGO-like noise curve (1/f^2 at low frequencies)
    noise_curve = jnp.where(
        frequencies > 20,
        1.0 / jnp.sqrt(frequencies + 1e-12),
        1.0 / jnp.sqrt(20.0)
    )
    
    # Apply coloring
    colored_fft = noise_fft * noise_curve
    colored_noise = jnp.fft.irfft(colored_fft, n=n_samples)
    
    # ✅ NORMALIZATION
    norm = jnp.sqrt(jnp.sum(colored_noise**2))
    if norm > 1e-12:
        colored_noise = colored_noise / norm
    
    return colored_noise


# ✅ CONVENIENCE FUNCTIONS

def create_simple_chirp(
    total_mass: float = 30.0,
    sample_rate: int = 4096,
    duration: float = 4.0,
    low_freq: float = 20.0
) -> jnp.ndarray:
    """
    ✅ CONVENIENCE: Create simple chirp template
    
    Args:
        total_mass: Total mass in solar masses
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        low_freq: Starting frequency in Hz
        
    Returns:
        Simple chirp template
    """
    # Assume equal mass binary
    mass1 = mass2 = total_mass / 2
    
    params = TemplateParameters(mass1=mass1, mass2=mass2)
    
    return create_chirp_template(
        params=params,
        sample_rate=sample_rate,
        duration=duration,
        low_freq=low_freq
    )


def create_default_template_bank(
    sample_rate: int = 4096,
    duration: float = 4.0
) -> List[jnp.ndarray]:
    """
    ✅ CONVENIENCE: Create default template bank for testing
    
    Returns a small but diverse template bank suitable for
    initial testing and development.
    """
    return create_template_bank(
        mass_range=(15.0, 40.0),
        num_templates=5,
        sample_rate=sample_rate,
        duration=duration
    )

