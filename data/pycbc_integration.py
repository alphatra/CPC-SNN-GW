"""
PyCBC integration for realistic GW waveform simulation and dataset creation.

Features (if PyCBC is available):
- TD waveforms (IMRPhenomD) with masses, spins, distance, inclination
- Detector response projection (H1/L1), time delays, F+/Fx
- PSD-colored noise and whitening
- Windowing with random injection position and proper labeling

Fallback: returns None if PyCBC is not installed or an error occurs.
"""

from typing import Optional, Tuple
import logging
import numpy as np
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def create_pycbc_enhanced_dataset(
    num_samples: int = 2000,
    window_size: int = 256,
    sample_rate: int = 4096,
    snr_range: Tuple[float, float] = (8.0, 20.0),
    mass_range: Tuple[float, float] = (10.0, 50.0),
    positive_ratio: float = 0.35,
    random_seed: int = 42,
    psd_name: str = "aLIGOZeroDetHighPower",
    whiten: bool = True,
    multi_channel: bool = False,
    sample_rate_high: int = 8192,
    resample_to: int = 256,
) -> Optional[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Create a realistic mixed dataset using PyCBC (if available).

    Returns:
        (signals, labels) as JAX arrays, or None if PyCBC not available.
    """
    try:
        from pycbc.waveform import get_td_waveform
        from pycbc.detector import Detector
        from pycbc import psd as _psd
        from pycbc.noise import noise_from_psd
        from pycbc.filter import highpass
        from pycbc.filter import whiten as ts_whiten
        from pycbc.types import TimeSeries
    except Exception as e:
        logger.warning(f"PyCBC not available: {e}")
        return None

    rng = np.random.default_rng(random_seed)
    h1 = Detector('H1')
    l1 = Detector('L1')

    # Prepare arrays
    signals: list[np.ndarray] = []
    labels: list[int] = []

    # PSD for noise (length in seconds must cover longest time)
    psd_len = 8  # seconds
    flen = int(sample_rate_high * psd_len // 2 + 1)
    try:
        psd_fn = getattr(_psd, psd_name)
        psd = psd_fn(flen, 1.0 / sample_rate_high, 20)
    except Exception:
        psd = _psd.aLIGOZeroDetHighPower(flen, 1.0 / sample_rate_high, 20)

    def _resample_np(x: np.ndarray, to_len: int) -> np.ndarray:
        try:
            from scipy.signal import resample
            return resample(x, to_len).astype(np.float32)
        except Exception:
            # naive decimate/pad
            factor = max(1, int(round(len(x) / to_len)))
            y = x[::factor]
            if len(y) < to_len:
                y = np.pad(y, (0, to_len - len(y)))
            return y[:to_len].astype(np.float32)

    def synthesize_one(positive: bool) -> np.ndarray:
        # Generate noise at high sample rate
        n_len_high = int(sample_rate_high * (window_size / sample_rate))
        noise_ts = noise_from_psd(n_len_high, 1.0 / sample_rate_high, psd)
        noise_ts = highpass(noise_ts, 20.0)

        if not positive:
            return _resample_np(np.array(noise_ts.numpy(), dtype=np.float32), window_size)

        # Random source parameters
        m1 = rng.uniform(*mass_range)
        m2 = rng.uniform(*mass_range)
        spin1z = rng.uniform(-0.8, 0.8)
        spin2z = rng.uniform(-0.8, 0.8)
        distance = rng.uniform(200, 1000)  # Mpc
        iota = rng.uniform(0, np.pi)
        ra = rng.uniform(0, 2 * np.pi)
        dec = rng.uniform(-np.pi/2, np.pi/2)
        psi = rng.uniform(0, np.pi)

        # Waveform
        try:
            hp, hc = get_td_waveform(
                approximant='IMRPhenomD',
                mass1=m1, mass2=m2,
                spin1z=spin1z, spin2z=spin2z,
                distance=distance,
                inclination=iota,
                delta_t=1.0 / sample_rate_high,
                f_lower=20.0,
            )
        except Exception:
            return _resample_np(np.array(noise_ts.numpy(), dtype=np.float32), window_size)

        # Project to H1 and L1 then average (simple baseline)
        hplus = hp.copy()
        hcross = hc.copy()
        # Trim/pad to high-rate window
        hp_arr = hplus.numpy()
        hc_arr = hcross.numpy()
        target_len_high = n_len_high
        if len(hp_arr) < target_len_high:
            pad = np.zeros(target_len_high - len(hp_arr), dtype=np.float32)
            hp_arr = np.concatenate([hp_arr, pad])
            hc_arr = np.concatenate([hc_arr, pad])
        else:
            hp_arr = hp_arr[-target_len_high:]
            hc_arr = hc_arr[-target_len_high:]

        # Antenna pattern and time delays
        gps = 0
        Fp_h1, Fx_h1 = h1.antenna_response(ra, dec, psi, gps)
        Fp_l1, Fx_l1 = l1.antenna_response(ra, dec, psi, gps)
        delay_h1 = h1.time_delay_from_earth_center(ra, dec, gps)
        delay_l1 = l1.time_delay_from_earth_center(ra, dec, gps)
        s_h1 = Fp_h1 * hp_arr + Fx_h1 * hc_arr
        s_l1 = Fp_l1 * hp_arr + Fx_l1 * hc_arr
        # Apply integer-sample delays
        shift_h1 = int(np.round(delay_h1 * sample_rate_high))
        shift_l1 = int(np.round(delay_l1 * sample_rate_high))
        proj_h1 = np.roll(s_h1, shift_h1)
        proj_l1 = np.roll(s_l1, shift_l1)

        # Random injection offset inside window
        shift = rng.integers(0, max(1, target_len_high // 4))
        proj_h1 = np.roll(proj_h1, shift)
        proj_l1 = np.roll(proj_l1, shift)

        # Mix with noise
        n_arr = np.array(noise_ts.numpy(), dtype=np.float32)
        x_h1 = proj_h1 + n_arr
        x_l1 = proj_l1 + n_arr

        # Whitening
        if whiten:
            try:
                ts_h1 = TimeSeries(x_h1, delta_t=1.0 / sample_rate_high)
                ts_l1 = TimeSeries(x_l1, delta_t=1.0 / sample_rate_high)
                ts_h1 = ts_whiten(ts_h1, psd, 20.0)
                ts_l1 = ts_whiten(ts_l1, psd, 20.0)
                x_h1 = ts_h1.numpy()
                x_l1 = ts_l1.numpy()
            except Exception:
                pass

        # Resample to target window length
        x_h1 = _resample_np(x_h1.astype(np.float32), window_size)
        x_l1 = _resample_np(x_l1.astype(np.float32), window_size)

        # Scale approximate SNR into target range by RMS ratio
        target_snr = rng.uniform(*snr_range)
        sig_rms = np.sqrt(np.mean((proj_h1**2 + proj_l1**2) / 2.0) + 1e-12)
        noise_rms = np.sqrt(np.mean(n_arr**2) + 1e-12)
        scale = (target_snr * noise_rms) / (sig_rms + 1e-12)
        x_h1 *= scale
        x_l1 *= scale

        if multi_channel:
            x = np.stack([x_h1, x_l1], axis=-1)
        else:
            x = 0.5 * (x_h1 + x_l1)

        # Normalize
        x = x - np.mean(x)
        x = x / (np.std(x) + 1e-6)
        return x.astype(np.float32)

    num_pos = int(num_samples * positive_ratio)
    num_neg = num_samples - num_pos

    for _ in range(num_pos):
        signals.append(synthesize_one(True))
        labels.append(1)
    for _ in range(num_neg):
        signals.append(synthesize_one(False))
        labels.append(0)

    # Shuffle
    idx = rng.permutation(len(signals))
    signals = [signals[i] for i in idx]
    labels = [labels[i] for i in idx]

    return jnp.array(signals), jnp.array(labels)


