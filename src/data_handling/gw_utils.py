# src/data_handling/gw_utils.py
import numpy as np
import logging
import warnings
from typing import Tuple, Optional

# PyCBC imports
from pycbc.types import TimeSeries as PyCBCTimeSeries, FrequencySeries
from pycbc.filter import resample_to_delta_t, highpass
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from scipy.signal import welch

def sanitize_strain(strain, min_duration=32.0):
    """
    Removes NaN fragments from data and returns the longest continuous segment.
    """
    data = strain.numpy()
    # Mask valid data (not NaN and not Inf)
    mask = np.isfinite(data)

    if np.all(mask):
        return strain

    # Find continuous segments
    # Change in mask value (edges)
    change_indices = np.where(np.diff(np.concatenate(([False], mask, [False]))))[0]

    # Pairs (start, stop)
    segments = change_indices.reshape(-1, 2)

    # Find the longest
    if len(segments) == 0:
        raise ValueError("Data contains only NaN/Inf values.")

    lengths = segments[:, 1] - segments[:, 0]
    max_idx = np.argmax(lengths)
    best_start, best_end = segments[max_idx]

    # Check if the longest segment has a reasonable length
    duration = (best_end - best_start) * strain.delta_t
    if duration < min_duration:
        raise ValueError(f"Longest continuous data segment is only {duration:.2f}s (required {min_duration}s).")

    # Cut in PyCBC (time slicing)
    # Calculate start/end times based on indices
    t0 = strain.start_time + best_start * strain.delta_t
    t1 = strain.start_time + best_end * strain.delta_t

    print(f"      [Dr. Gravitas] Found gaps (NaN). Trimming to longest segment: {duration:.1f}s")
    return strain.time_slice(t0, t1)


def get_whitened_strain(
        strain_data,
        sample_rate: float,
        low_freq_cutoff: float = 20.0
) -> Tuple[PyCBCTimeSeries, FrequencySeries]:
    """
    Rigorous data whitening.
    Automatically converts GWpy objects to PyCBC and cleans NaNs.
    """

    # --- TYPE ADAPTER (GWpy -> PyCBC) ---
    if hasattr(strain_data, 'to_pycbc'):
        strain = strain_data.to_pycbc()
    else:
        strain = strain_data

    # Convert to float64 (required for numerical stability of filters)
    if strain.dtype != np.float64:
        strain = strain.astype(np.float64)

    # --- SANITIZATION (New) ---
    # Remove NaNs before doing anything
    try:
        strain = sanitize_strain(strain)
    except ValueError as e:
        # Pass error up so main script knows this segment is useless
        raise RuntimeError(f"Sanitization failed: {e}")

    # 1. Resampling
    # Resampling can introduce artifacts at edges, so do it on clean segment
    if strain.sample_rate != sample_rate:
        strain = resample_to_delta_t(strain, 1.0 / sample_rate)

    # 2. Highpass filter (remove seismic drift)
    # Note: filtering introduces "ringing" at start and end, we will crop this later
    strain = highpass(strain, frequency=15.0)

    # 3. PSD Estimation (Welch's method)
    # Use 4-second segments.
    seg_len_sec = 4
    seg_len = int(seg_len_sec * sample_rate)
    # seg_stride = int(seg_len / 2) # Unused variable

    # Safety: do we have enough data for PSD?
    if strain.duration < (seg_len_sec * 2):
        raise RuntimeError(f"Data after sanitization is too short ({strain.duration}s) to calculate PSD.")

    # Use scipy.signal.welch for robust PSD estimation
    f_val, p_val = welch(
        strain.numpy(), 
        fs=strain.sample_rate, 
        nperseg=seg_len,
        average='mean'
    )
    
    # Convert to PyCBC FrequencySeries (welch returns one-sided PSD)
    delta_f = f_val[1]
    psd = FrequencySeries(p_val, delta_f=delta_f)



    # 4. PSD Interpolation
    psd = interpolate(psd, strain.delta_f)
    psd = inverse_spectrum_truncation(psd, int(4 * strain.sample_rate),
                                      low_frequency_cutoff=low_freq_cutoff,
                                      trunc_method='hann')

    # 5. Whitening in frequency domain
    # Divide signal spectrum by square root of PSD
    whitened = (strain.to_frequencyseries() / psd ** 0.5).to_timeseries()

    # 6. Crop filter edge artifacts
    # Remove 4 seconds from start and end, as filters go crazy there
    if whitened.duration > 8.0:
        whitened = whitened.crop(4, 4)
    else:
        # If little data remains, crop less aggressively, but warn
        warnings.warn("Short data segment, cropping only 1s margin.")
        whitened = whitened.crop(1, 1)

    # 7. Normalization (Critical for SNN stability)
    # Normalize to mean 0 and standard deviation 1
    data = whitened.numpy()
    data = (data - np.mean(data)) / np.std(data)
    whitened = PyCBCTimeSeries(data, delta_t=whitened.delta_t, epoch=whitened.start_time)

    return whitened, psd

def whiten_with_psd(strain: PyCBCTimeSeries, psd: FrequencySeries) -> PyCBCTimeSeries:
    """
    Whitens a strain using a pre-computed PSD.
    """
    # 1. Convert to FrequencySeries
    strain_f = strain.to_frequencyseries()
    
    # 2. Interpolate PSD to match strain's delta_f if needed
    if psd.delta_f != strain.delta_f:
        psd = interpolate(psd, strain.delta_f)
        
    # 3. Whiten
    # Ensure lengths match (pad PSD if needed)
    if len(psd) < len(strain_f):
        psd.resize(len(strain_f))
        
    whitened = (strain_f / psd ** 0.5).to_timeseries()
    
    return whitened


def generate_waveform(
    mass_range: tuple = (10, 50),
    sample_rate: float = 2048.0,
    f_lower: float = 20.0,
    approximant: str = "IMRPhenomD",
    preferential_prob: float = 0.5
) -> Tuple[PyCBCTimeSeries, PyCBCTimeSeries]:
    """
    Generates a synthetic gravitational wave (BBH) waveform (plus and cross).
    
    Args:
        preferential_prob (float): Probability (0.0-1.0) of applying Preferential Accretion physics 
                                   (Comerford & Simon 2025), which favors equal mass ratios q->1.
    """
    m1 = np.random.uniform(*mass_range)
    
    if np.random.random() < preferential_prob:
        # --- SCIENTIFIC PATH (Comerford & Simon 2025) ---
        # Preferential Accretion favors q values close to 1.0 (equal masses).
        # Beta(5, 1) provides a distribution heavily skewed towards 1.
        q = np.random.beta(a=5, b=1)
        m2 = m1 * q
    else:
        # --- CLASSICAL PATH (Hard cases coverage) ---
        # Naive uniform sampling, allows for highly asymmetric binaries (q << 1)
        m2 = np.random.uniform(*mass_range)
        if m2 > m1: m1, m2 = m2, m1

    try:
        hp, hc = get_td_waveform(
            approximant=approximant,
            mass1=m1, mass2=m2,
            delta_t=1.0 / sample_rate,
            f_lower=f_lower
        )
    except Exception as e:
        print(f"Waveform generation error: {e}. Retrying with standard masses.")
        hp, hc = get_td_waveform(
            approximant=approximant, 
            mass1=30, mass2=30, 
            delta_t=1.0 / sample_rate, 
            f_lower=f_lower
        )
    
    return hp, hc


def project_waveform_to_ifo(
    hp: PyCBCTimeSeries,
    hc: PyCBCTimeSeries,
    ifo_name: str,
    ra: float,
    dec: float,
    psi: float,
    t_gps: float
) -> PyCBCTimeSeries:
    """
    Projects the waveform onto a specific interferometer (antenna pattern + time delay).
    """
    det = Detector(ifo_name)
    
    # Calculate antenna patterns
    fp, fc = det.antenna_pattern(ra, dec, psi, t_gps)
    
    # Calculate time delay relative to Earth center
    dt = det.time_delay_from_earth_center(ra, dec, t_gps)
    
    # Apply time delay and antenna pattern
    signal = fp * hp + fc * hc
    
    # Shift signal in time
    # PyCBC TimeSeries has a start_time. We can shift it or cyclically shift the data.
    # Ideally, we want the merger to happen at t_gps + dt.
    # The generated waveform usually has merger near the end or 0.
    # Let's assume we just want to shift the content.
    
    # Note: cyclic_time_shift is good for circular buffers, but here we might want exact placement.
    # However, for short injections in noise, cyclic shift is often used if the buffer is large enough.
    
    # Shift so that merger is at t_gps
    # The waveform starts at 0. We want it to "end" (merger) at t_gps.
    # But wait, PyCBC waveforms usually have merger at the end?
    # Actually, get_td_waveform returns a TimeSeries with a specific start_time.
    # We just need to align it.
    
    # Simpler approach: Just return the projected signal. The caller handles placement.
    # But wait, the antenna pattern depends on t_gps.
    
    return signal


def generate_glitch(
    sample_rate: float = 2048.0,
    duration: float = 4.0,
    f_range: tuple = (30, 250),
    q_range: tuple = (5, 20),
    hrss_range: tuple = (1e-22, 1e-20)
) -> PyCBCTimeSeries:
    """
    Generates a Sine-Gaussian glitch (burst).
    """
    # Random parameters
    freq = np.random.uniform(*f_range)
    q = np.random.uniform(*q_range)
    hrss = np.random.uniform(*hrss_range) # Amplitude
    
    # Tau (duration of the burst)
    tau = q / (2 * np.pi * freq)
    
    # Time vector centered at duration/2
    dt = 1.0 / sample_rate
    t = np.arange(0, duration, dt)
    t0 = duration / 2.0
    
    # Sine-Gaussian formula
    # h(t) = A * exp(-(t-t0)^2 / tau^2) * sin(2*pi*f*(t-t0))
    # We approximate A from hrss (root sum squared)
    
    # Envelope
    env = np.exp(-(t - t0)**2 / tau**2)
    
    # Waveform
    # Random phase
    phase = np.random.uniform(0, 2*np.pi)
    strain = env * np.sin(2 * np.pi * freq * (t - t0) + phase)
    
    # Normalize to desired hrss
    # hrss^2 = sum(h(t)^2 * dt)
    current_hrss = np.sqrt(np.sum(strain**2) * dt)
    strain *= (hrss / current_hrss)
    
    # Convert to PyCBC TimeSeries
    ts = PyCBCTimeSeries(strain, delta_t=dt)
    
    return ts



def generate_injection(
        noise_bg: np.ndarray,
        sample_rate: float,
        mass_range: tuple = (10, 50),
        snr_target: float = None
) -> np.ndarray:
    """
    Generates a synthetic gravitational wave (BBH) and injects it into background noise.
    (Single-channel version for backward compatibility)
    """
    hp, hc = generate_waveform(mass_range, sample_rate)

    phase = np.random.uniform(0, 2 * np.pi)
    signal = hp * np.cos(phase) + hc * np.sin(phase)

    # Length matching
    if len(signal) > len(noise_bg):
        mid = len(signal) // 2
        half_win = len(noise_bg) // 2
        signal = signal[mid - half_win: mid + half_win]
        signal.resize(len(noise_bg))
    else:
        signal.resize(len(noise_bg))
        shift_samples = np.random.randint(int(0.2 * len(noise_bg)), int(0.8 * len(noise_bg)))
        signal = signal.cyclic_time_shift(shift_samples * signal.delta_t)

    if snr_target:
        sig_energy = np.sqrt(np.sum(signal.numpy() ** 2))
        if sig_energy > 0:
            factor = snr_target / sig_energy
            signal *= factor

    data_with_signal = noise_bg + signal.numpy()
    return data_with_signal.astype(np.float32)