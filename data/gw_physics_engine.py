"""
GW Physics Engine: Signal Physics and Doppler Modulation
Extracted from continuous_gw_generator.py for modular architecture.
"""

import logging
from typing import Dict, Tuple
import jax
import jax.numpy as jnp
from .gw_signal_params import ContinuousGWParams, PHYSICS

logger = logging.getLogger(__name__)


@jax.jit
def compute_doppler_factor(t: jnp.ndarray, 
                          alpha: float, 
                          delta: float,
                          detector_latitude: float = 0.6370,
                          detector_longitude: float = -2.0847,
                          t_ref: float = 0.0) -> jnp.ndarray:
    """
    Compute Doppler factor from Earth rotation and orbital motion with full orbital model.
    
    Args:
        t: Time array (seconds)
        alpha: Right ascension (rad)
        delta: Declination (rad)
        detector_latitude: Detector latitude (rad)
        detector_longitude: Detector longitude (rad)
        t_ref: Reference time for orbital phase (GPS seconds, default: 0.0)
        
    Returns:
        Doppler factor array
    """
    # Convert time to days for orbital calculations
    t_days = t / (24 * 3600)
    
    # Earth rotation component (sidereal rate)
    earth_rotation_phase = PHYSICS.SIDEREAL_RATE * t + detector_longitude
    
    # Earth orbital motion around the Sun
    orbital_velocity = PHYSICS.orbital_velocity
    
    # Orbital phase (t_ref sets the starting phase)
    orbital_phase = 2 * jnp.pi * (t + t_ref) / PHYSICS.ORBITAL_PERIOD
    
    # Position of Earth in orbital plane (simplified circular orbit)
    earth_orbital_x = PHYSICS.ORBITAL_RADIUS * jnp.cos(orbital_phase)
    earth_orbital_y = PHYSICS.ORBITAL_RADIUS * jnp.sin(orbital_phase)
    
    # Earth velocity components in orbital plane
    v_earth_x = -orbital_velocity * jnp.sin(orbital_phase)
    v_earth_y = orbital_velocity * jnp.cos(orbital_phase)
    
    # Unit vector pointing to the GW source
    source_x = jnp.cos(alpha) * jnp.cos(delta)
    source_y = jnp.sin(alpha) * jnp.cos(delta) 
    source_z = jnp.sin(delta)
    
    # Earth rotation velocity at detector
    rotation_velocity = PHYSICS.rotation_velocity_factor * jnp.cos(detector_latitude)
    
    # Velocity components from Earth rotation
    v_rotation_x = -rotation_velocity * jnp.sin(earth_rotation_phase)
    v_rotation_y = rotation_velocity * jnp.cos(earth_rotation_phase)
    v_rotation_z = 0.0
    
    # Total velocity of detector
    v_detector_x = v_earth_x + v_rotation_x
    v_detector_y = v_earth_y + v_rotation_y
    v_detector_z = v_rotation_z
    
    # Doppler factor: 1 + v·n/c where n is unit vector to source
    # Dot product of velocity with source direction
    v_dot_n = (v_detector_x * source_x + 
               v_detector_y * source_y + 
               v_detector_z * source_z)
    
    # Basic Doppler factor
    doppler_factor = 1.0 + v_dot_n / PHYSICS.LIGHT_SPEED
    
    return doppler_factor


@jax.jit
def enhanced_doppler_factor(t: jnp.ndarray, 
                           alpha: float, 
                           delta: float,
                           detector_latitude: float = 0.6370,
                           detector_longitude: float = -2.0847,
                           include_relativity: bool = True,
                           t_ref: float = 0.0) -> Tuple[jnp.ndarray, Dict]:
    """
    Enhanced Doppler factor computation with detailed component breakdown.
    
    Returns:
        total_doppler: Combined Doppler factor
        components: Dictionary with individual contributions
    """
    # Reuse basic computation
    basic_doppler = compute_doppler_factor(
        t, alpha, delta, detector_latitude, detector_longitude, t_ref
    )
    
    # Additional relativistic corrections if requested
    if include_relativity:
        # Time dilation from Earth's orbital motion (~10^-8 effect)
        orbital_velocity = PHYSICS.orbital_velocity
        gamma_factor = 1.0 / jnp.sqrt(1.0 - (orbital_velocity / PHYSICS.LIGHT_SPEED)**2)
        time_dilation = 1.0 / gamma_factor
        
        # Gravitational redshift from Sun (~10^-8 effect)
        # This is a very small correction for most applications
        gravitational_redshift = 1.0 + 2.12e-8  # Approximate
        
        total_doppler = basic_doppler * time_dilation * gravitational_redshift
    else:
        time_dilation = 1.0
        gravitational_redshift = 1.0
        total_doppler = basic_doppler
    
    # Component breakdown for analysis
    components = {
        'basic_doppler': basic_doppler,
        'time_dilation': time_dilation,
        'gravitational_redshift': gravitational_redshift,
        'total': total_doppler
    }
    
    return total_doppler, components


@jax.jit
def compute_gw_polarizations(phase: jnp.ndarray, 
                           amplitude_h0: float,
                           cosi: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute plus and cross polarizations of GW signal.
    
    Args:
        phase: Signal phase array
        amplitude_h0: Strain amplitude
        cosi: Cosine of inclination angle
        
    Returns:
        h_plus: Plus polarization
        h_cross: Cross polarization
    """
    # Plus polarization
    h_plus = (
        amplitude_h0 * 
        (1 + cosi**2) / 2 * 
        jnp.cos(phase)
    )
    
    # Cross polarization  
    h_cross = (
        amplitude_h0 * 
        cosi * 
        jnp.sin(phase)
    )
    
    return h_plus, h_cross


@jax.jit
def compute_detector_response(h_plus: jnp.ndarray, 
                            h_cross: jnp.ndarray,
                            psi: float,
                            detector_name: str = 'H1') -> jnp.ndarray:
    """
    Compute detector response to GW polarizations.
    
    Args:
        h_plus: Plus polarization
        h_cross: Cross polarization
        psi: Polarization angle (rad)
        detector_name: Detector identifier
        
    Returns:
        Detector strain output
    """
    # Simplified antenna pattern (full implementation would need detector tensor)
    # This is a placeholder for proper antenna pattern calculation
    antenna_pattern_plus = jnp.cos(2 * psi)
    antenna_pattern_cross = jnp.sin(2 * psi)
    
    # Detector response
    signal = (antenna_pattern_plus * h_plus + 
             antenna_pattern_cross * h_cross)
    
    return signal


@jax.jit
def integrate_phase(instantaneous_frequency: jnp.ndarray, 
                   dt: float,
                   phi0: float = 0.0) -> jnp.ndarray:
    """
    Integrate instantaneous frequency to get signal phase.
    
    Args:
        instantaneous_frequency: Frequency array (rad/s)
        dt: Time step (seconds)
        phi0: Initial phase (rad)
        
    Returns:
        Integrated phase array
    """
    # Use cumulative trapezoidal integration for accuracy
    integrated_phase = jnp.cumsum(instantaneous_frequency) * dt
    phase = integrated_phase + phi0
    
    # Apply modulo 2π to prevent phase overflow for long durations
    phase = phase % (2 * jnp.pi)
    
    return phase


def test_signal_physics(params: ContinuousGWParams,
                       duration: float = 4.0) -> Dict[str, float]:
    """
    Test signal physics for consistency and realistic behavior.
    
    Args:
        params: Signal parameters
        duration: Test duration (seconds)
        
    Returns:
        Dictionary with physics test results
    """
    try:
        sampling_rate = 4096
        t = jnp.arange(0, duration, 1/sampling_rate)
        
        # Test Doppler modulation
        if params.include_doppler:
            doppler_factor = compute_doppler_factor(
                t, params.alpha, params.delta,
                params.detector_latitude, params.detector_longitude
            )
            doppler_variation = jnp.max(doppler_factor) - jnp.min(doppler_factor)
        else:
            doppler_variation = 0.0
        
        # Test frequency evolution
        omega = 2 * jnp.pi * params.frequency
        omega_dot = 2 * jnp.pi * params.frequency_dot
        
        instantaneous_frequency = omega + omega_dot * t
        if params.include_doppler:
            instantaneous_frequency *= compute_doppler_factor(
                t, params.alpha, params.delta,
                params.detector_latitude, params.detector_longitude
            )
        
        frequency_drift = (instantaneous_frequency[-1] - instantaneous_frequency[0]) / (2 * jnp.pi)
        
        # Test phase integration
        phase = integrate_phase(instantaneous_frequency, 1/sampling_rate, params.phi0)
        phase_wraps = jnp.sum(jnp.abs(jnp.diff(phase)) > jnp.pi)
        
        # Test polarizations
        h_plus, h_cross = compute_gw_polarizations(phase, params.amplitude_h0, params.cosi)
        
        # Test detector response
        strain = compute_detector_response(h_plus, h_cross, params.psi)
        
        # Compute test metrics
        results = {
            'doppler_variation': float(doppler_variation),
            'frequency_drift_hz': float(frequency_drift),
            'phase_wraps': int(phase_wraps),
            'strain_rms': float(jnp.sqrt(jnp.mean(strain**2))),
            'strain_peak': float(jnp.max(jnp.abs(strain))),
            'h_plus_rms': float(jnp.sqrt(jnp.mean(h_plus**2))),
            'h_cross_rms': float(jnp.sqrt(jnp.mean(h_cross**2))),
            'duration_tested': duration,
            'sample_rate': sampling_rate
        }
        
        # Physics consistency checks
        expected_doppler = 1e-4  # ~v/c for Earth motion
        if results['doppler_variation'] > 10 * expected_doppler:
            logger.warning(f"Large Doppler variation: {results['doppler_variation']:.2e}")
        
        if abs(results['frequency_drift_hz']) > 0.1:
            logger.warning(f"Large frequency drift: {results['frequency_drift_hz']:.2e} Hz")
        
        if results['strain_peak'] / params.amplitude_h0 > 2.0:
            logger.warning(f"Signal amplification > 2x: {results['strain_peak']/params.amplitude_h0:.2f}")
        
        logger.info("Signal physics test completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Signal physics test failed: {e}")
        return {'error': str(e)}


def validate_physics_consistency(params: ContinuousGWParams) -> bool:
    """
    Validate physics consistency of signal parameters.
    
    Args:
        params: Signal parameters to validate
        
    Returns:
        True if physics is consistent, False otherwise
    """
    try:
        # Test short duration signal
        test_results = test_signal_physics(params, duration=1.0)
        
        if 'error' in test_results:
            logger.error(f"Physics validation failed: {test_results['error']}")
            return False
        
        # Check for reasonable values
        if test_results['strain_rms'] <= 0:
            logger.error("Zero or negative strain RMS")
            return False
        
        if test_results['strain_peak'] > 100 * params.amplitude_h0:
            logger.error(f"Unrealistic signal amplification: "
                        f"{test_results['strain_peak']/params.amplitude_h0:.1f}x")
            return False
        
        logger.info("Physics consistency validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Physics validation error: {e}")
        return False 