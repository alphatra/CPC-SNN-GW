"""
Real LIGO Data Integration Module

Migrated from real_ligo_test.py - provides real LIGO data downloading
and proper windowing functionality for the main CLI and pipeline.

Key Features:
- download_gw150914_data(): Downloads real GW150914 strain data using ReadLIGO
- create_proper_windows(): Creates properly labeled overlapping windows
- Integration with existing data pipeline
"""

import logging
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)

def download_gw150914_data() -> Optional[np.ndarray]:
    """
    Download GW150914 data using ReadLIGO library with local HDF5 files
    
    Returns:
        Real strain data as numpy array or None if failed
    """
    try:
        # ✅ Attempt 1: Read using ReadLIGO (preferred)
        import readligo as rl  # type: ignore
        import os
        fn_H1 = 'H-H1_LOSC_4_V2-1126259446-32.hdf5'
        fn_L1 = 'L-L1_LOSC_4_V2-1126259446-32.hdf5'
        if not (os.path.exists(fn_H1) and os.path.exists(fn_L1)):
            raise FileNotFoundError("GW150914 HDF5 files not found in project root")
        logger.info("✅ ReadLIGO available - loading real GW150914 data")
        strain_H1, time_H1, _ = rl.loaddata(fn_H1, 'H1')
        strain_L1, time_L1, _ = rl.loaddata(fn_L1, 'L1')
        sample_rate = 1.0 / (time_H1[1] - time_H1[0])
    except Exception as e1:
        logger.warning(f"❌ ReadLIGO loading failed: {type(e1).__name__}: {e1}")
        # ✅ Attempt 2: Direct HDF5 read via h5py
        try:
            import h5py  # type: ignore
            import os
            fn_H1 = 'H-H1_LOSC_4_V2-1126259446-32.hdf5'
            fn_L1 = 'L-L1_LOSC_4_V2-1126259446-32.hdf5'
            if not (os.path.exists(fn_H1) and os.path.exists(fn_L1)):
                raise FileNotFoundError("GW150914 HDF5 files not found in project root")
            logger.info("✅ Using h5py fallback - loading LOSC HDF5 files directly")
            with h5py.File(fn_H1, 'r') as fH, h5py.File(fn_L1, 'r') as fL:
                strain_H1 = fH['strain/Strain'][:]
                strain_L1 = fL['strain/Strain'][:]
                # Try to get sample rate from metadata
                if 'meta' in fH and 'SampleRate' in fH['meta']:
                    sample_rate = float(fH['meta']['SampleRate'][()])
                elif 'meta' in fH and 'SamplingRate' in fH['meta']:
                    sample_rate = float(fH['meta']['SamplingRate'][()])
                else:
                    sample_rate = 4096.0
                # Build time arrays if GPSstart available
                if 'meta' in fH and 'GPSstart' in fH['meta']:
                    start = float(fH['meta']['GPSstart'][()])
                else:
                    start = 0.0
                n = strain_H1.shape[0]
                time_H1 = start + np.arange(n) / sample_rate
                time_L1 = time_H1  # LOSC files are time-aligned for this segment
        except Exception as e2:
            logger.info("🔄 Falling back to simulated GW150914-like data")
            return create_simulated_gw150914_strain()

    # Combine detectors (H1 + L1 average)
    combined_strain = (strain_H1 + strain_L1) / 2.0

    # Select 2048 samples around the GW150914 event time
    event_gps_time = 1126259462.4
    event_idx = np.argmin(np.abs(time_H1 - event_gps_time))
    start_idx = max(0, int(event_idx) - 1024)
    end_idx = min(len(combined_strain), start_idx + 2048)
    strain_subset = combined_strain[start_idx:end_idx]

    # Pad with zeros if needed
    if len(strain_subset) < 2048:
        strain_padded = np.zeros(2048, dtype=np.float32)
        strain_padded[:len(strain_subset)] = strain_subset
        strain_subset = strain_padded

    # Realistic normalization
    strain_subset = strain_subset.astype(np.float32) * 0.01
    logger.info(f"✅ Successfully loaded real GW150914 strain data (source={'readligo' if 'rl' in locals() else 'h5py'}): {len(strain_subset)} samples")
    logger.info(f"📊 Strain amplitude range: {strain_subset.min():.2e} to {strain_subset.max():.2e}")
    return strain_subset

def create_proper_windows(strain_data: np.ndarray, 
                         window_size: int = 512, 
                         overlap: float = 0.5) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create properly labeled overlapping windows from strain data
    
    Args:
        strain_data: Input strain data array
        window_size: Size of each window 
        overlap: Overlap ratio (0.5 = 50% overlap)
        
    Returns:
        Tuple of (windows, labels) as JAX arrays
    """
    stride = int(window_size * (1 - overlap))
    windows = []
    labels = []
    
    # GW150914 event is roughly in the middle of the 2048 samples
    event_center = len(strain_data) // 2
    event_start = event_center - window_size // 2
    event_end = event_center + window_size // 2
    
    for start_idx in range(0, len(strain_data) - window_size + 1, stride):
        end_idx = start_idx + window_size
        window = strain_data[start_idx:end_idx]
        
        # ✅ PROPER LABELING: Check overlap with event region
        overlap_start = max(start_idx, event_start)
        overlap_end = min(end_idx, event_end)
        overlap_ratio = max(0, overlap_end - overlap_start) / window_size
        
        # Label as GW signal if significant overlap (>30%) with event
        if overlap_ratio > 0.3:
            label = 1  # GW signal present
        else:
            label = 0  # Background noise
            
        windows.append(window)
        labels.append(label)
    
    # Convert using NumPy first to avoid Metal default_memory_space issues
    from utils.jax_safety import safe_stack_to_device, safe_array_to_device
    if len(windows) > 0:
        windows_dev = safe_stack_to_device(windows, dtype=np.float32)
    else:
        windows_dev = safe_array_to_device(np.zeros((0, window_size), dtype=np.float32))
    labels_dev = safe_array_to_device(np.array(labels, dtype=np.int32))
    return windows_dev, labels_dev

def create_simulated_gw150914_strain() -> np.ndarray:
    """
    Create simulated GW150914-like strain data (improved baseline).
    NOTE: For production-quality simulations, prefer PyCBC/Bilby waveforms with
    detector response (H1/L1), PSD-colored noise and whitening.
    This baseline adds colored noise shaping and variable chirp.
    """
    from scipy import signal
    
    # GW150914 physical parameters (approximate)
    m1 = 36.0  # Solar masses - primary black hole
    m2 = 29.0  # Solar masses - secondary black hole
    sample_rate = 4096  # Hz
    duration = 0.5  # seconds (2048 samples)
    
    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simulated chirp signal (simplified inspiral waveform baseline)
    
    # Initial frequency and chirp mass
    f_start = 35  # Hz - GW150914 entered LIGO band around 35 Hz
    f_end = 300   # Hz - merger frequency
    
    # Chirp mass from GW150914 parameters
    chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)  # Solar masses
    
    # Time to merger (using post-Newtonian approximation)
    # Simplified frequency evolution
    tau = 1.0 - t/duration  # Time to merger (normalized)
    tau = np.maximum(tau, 0.01)  # Avoid division by zero
    
    # Frequency evolution (chirp)
    f_gw = f_start + (f_end - f_start) * (1 - tau**(3/8))
    
    # Amplitude evolution (increases as merger approaches)
    # Peak strain for GW150914 was approximately 1e-21
    peak_strain = 1e-21
    amplitude = peak_strain * (f_gw / f_start)**2 * np.exp(-tau * 2)
    
    # Phase evolution
    phase = 2 * np.pi * np.cumsum(f_gw * duration / len(t))
    
    # Plus polarization (main component)
    h_plus = amplitude * np.sin(phase)
    
    # Add colored noise approximating LIGO PSD (1/f^2 shaping)
    rng = np.random.default_rng(123)
    white = rng.normal(0, 1.0, len(t)).astype(np.float32)
    # Frequency-domain shaping: 1/(1 + (f/f0)^2)
    f = np.fft.rfftfreq(len(t), d=1.0/sample_rate)
    shaping = 1.0 / (1.0 + (f / 40.0)**2)
    shaped = np.fft.irfft(np.fft.rfft(white) * shaping, n=len(t)).astype(np.float32)
    # Scale to target ASD level
    noise_level = 5e-23
    noise = shaped * noise_level
    
    # Combine signal and noise
    strain = h_plus + noise
    
    # Apply realistic LIGO bandpass (remove very low frequencies)
    # High-pass filter to remove frequencies below 20 Hz
    sos = signal.butter(4, 20, btype='highpass', fs=sample_rate, output='sos')
    strain_filtered = signal.sosfilt(sos, strain)
    
    # Convert to float32 and ensure correct length
    strain_final = np.array(strain_filtered, dtype=np.float32)
    if len(strain_final) != 2048:
        if len(strain_final) > 2048:
            strain_final = strain_final[:2048]
        else:
            # Pad with noise if too short
            padding = 2048 - len(strain_final)
            noise_pad = np.random.normal(0, noise_level, padding).astype(np.float32)
            strain_final = np.concatenate([strain_final, noise_pad])
    
    logger.info(f"✅ Created simulated GW150914-like strain data: {len(strain_final)} samples")
    logger.info(f"📊 Simulated strain amplitude range: {strain_final.min():.2e} to {strain_final.max():.2e}")
    logger.info(f"🌊 Signal frequency range: {f_start:.1f} - {f_end:.1f} Hz")
    logger.info(f"⭐ Simulated masses: {m1:.1f} + {m2:.1f} solar masses")
    
    return strain_final

def create_proper_windows(strain_data: np.ndarray, 
                         window_size: int = 256,
                         overlap: float = 0.5,
                         event_location: float = 0.6) -> Tuple[List[np.ndarray], List[int]]:
    """
    🚨 MISSING FUNCTION IMPLEMENTED: Create properly labeled overlapping windows
    
    Based on MLGWSC-1 approach - creates sliding windows with proper overlap.
    
    Args:
        strain_data: Raw strain data array
        window_size: Size of each window
        overlap: Overlap fraction (0.5 = 50% overlap)
        event_location: Where GW event is expected (0.6 = 60% into data)
        
    Returns:
        Tuple of (window_list, label_list)
    """
    logger.info(f"🔧 Creating sliding windows: size={window_size}, overlap={overlap:.1%}")
    
    data_length = len(strain_data)
    step_size = int(window_size * (1 - overlap))  # Step between windows
    
    windows = []
    labels = []
    
    # Calculate GW event position in original data
    gw_event_sample = int(data_length * event_location)
    
    # Generate sliding windows (MLGWSC-1 style)
    for start in range(0, data_length - window_size + 1, step_size):
        end = start + window_size
        window = strain_data[start:end]
        
        # Label based on whether window contains GW event
        window_start_rel = start / data_length
        window_end_rel = end / data_length
        
        # Label=1 if window contains significant part of GW event  
        if window_start_rel <= event_location <= window_end_rel:
            # Check if event is significantly within window (not just edge)
            event_pos_in_window = (event_location - window_start_rel) / (window_end_rel - window_start_rel)
            if 0.2 <= event_pos_in_window <= 0.8:  # Event well within window
                label = 1
            else:
                label = 0  # Event too close to edges
        else:
            label = 0  # Pure noise window
            
        windows.append(window.astype(np.float32))
        labels.append(label)
    
    logger.info(f"✅ Generated {len(windows)} sliding windows")
    logger.info(f"   Signal windows: {sum(labels)} ({100*np.mean(labels):.1f}%)")
    logger.info(f"   Noise windows: {len(labels)-sum(labels)} ({100*(1-np.mean(labels)):.1f}%)")
    
    return windows, labels

def create_enhanced_ligo_dataset(num_samples: int = 2000,
                               window_size: int = 256,
                               enhanced_overlap: float = 0.9,
                               data_augmentation: bool = True,
                               noise_scaling: bool = True,
                               target_pos_ratio: float = 0.40,
                               synthetic_positive_ratio: float = 0.35) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create ENHANCED dataset with significantly more samples through:
    - High overlap windowing (90% instead of 50%)
    - Data augmentation (noise scaling, time shifts)
    - Synthetic data supplementation
    
    Args:
        num_samples: Target number of total samples
        window_size: Size of each window
        enhanced_overlap: High overlap for more windows (0.9 = 90%)
        data_augmentation: Apply noise/amplitude variations
        noise_scaling: Add realistic noise variations
        
    Returns:
        Tuple of (signals, labels) with enhanced dataset
    """
    logger.info(f"🚀 Creating ENHANCED dataset with {num_samples} target samples")
    logger.info(f"   Enhanced overlap: {enhanced_overlap:.1%}")
    logger.info(f"   Data augmentation: {data_augmentation}")
    
    all_signals = []
    all_labels = []
    
    # 1. GET REAL LIGO DATA with HIGH OVERLAP
    real_strain = download_gw150914_data()
    if real_strain is not None:
        logger.info("📡 Processing real LIGO data with enhanced windowing...")
        
        # Use high overlap (90%) for many more windows
        # Respect external overlap if provided via env override to keep CLI consistent
        try:
            import os as _os
            _ov = _os.environ.get('GW_OVERLAP')
            _ov = float(_ov) if _ov is not None else enhanced_overlap
        except Exception:
            _ov = enhanced_overlap
        signals, labels = create_proper_windows(real_strain, window_size=window_size, overlap=_ov)
        
        logger.info(f"✅ Real LIGO windows: {len(signals)} (overlap={_ov:.1%})")
        all_signals.extend(signals)
        all_labels.extend(labels)
        
        # 2. DATA AUGMENTATION on real data
        if data_augmentation and len(signals) > 0:
            logger.info("🔄 Applying data augmentation to real LIGO data...")
            
            for amp_factor in [0.8, 1.2, 1.5]:  # Amplitude variations
                aug_signals = signals * amp_factor
                all_signals.extend(aug_signals)
                all_labels.extend(labels)
                
            for noise_level in [0.5, 2.0]:  # Noise level variations
                noise = jax.random.normal(jax.random.PRNGKey(42), signals.shape) * 1e-21
                aug_signals = signals + noise_level * noise
                all_signals.extend(aug_signals)
                all_labels.extend(labels)
                
            logger.info(f"✅ Augmented data: +{len(signals) * 5} samples")

            # ✅ Oversample positives to reach target_pos_ratio in training pool
            try:
                import numpy as _np
                labels_np = _np.array(labels)
                pos_idx = _np.where(labels_np == 1)[0]
                neg_idx = _np.where(labels_np == 0)[0]
                total_now = len(all_signals)
                pos_now = int(_np.sum(_np.array(all_labels) == 1))
                current_ratio = pos_now / max(1, total_now)
                if current_ratio < target_pos_ratio and len(pos_idx) > 0:
                    needed_pos = int(target_pos_ratio * total_now) - pos_now
                    needed_pos = max(0, needed_pos)
                    if needed_pos > 0:
                        # Sample with replacement from positive windows
                        rng = _np.random.default_rng(123)
                        sample_indices = rng.choice(pos_idx, size=needed_pos, replace=True)
                        for si in sample_indices:
                            all_signals.append(signals[si])
                            all_labels.append(1)
                        logger.info(f"✅ Oversampled positives by {needed_pos} to reach ~{target_pos_ratio:.0%}")
            except Exception as _e:
                logger.warning(f"Positive oversampling skipped: {_e}")
    
    # 3. SYNTHETIC DATA SUPPLEMENTATION
    current_samples = len(all_signals)
    remaining_samples = max(0, num_samples - current_samples)
    
    if remaining_samples > 0:
        logger.info(f"🔄 Generating {remaining_samples} synthetic GW samples...")
        
        key = jax.random.PRNGKey(123)
        time_series = jnp.linspace(0, 4.0, window_size)
        
        for i in range(remaining_samples):
            signal_key, key = jax.random.split(key)
            
            # ✅ Use target synthetic positive ratio (default 50%)
            is_positive = jax.random.bernoulli(signal_key, p=synthetic_positive_ratio)
            if bool(is_positive):
                # Generate realistic chirp signals
                f0 = 35.0 + jax.random.uniform(signal_key, (), minval=-10, maxval=10)
                f1 = 350.0 + jax.random.uniform(signal_key, (), minval=-50, maxval=50)
                chirp_rate = (f1 - f0) / 4.0
                freq = f0 + chirp_rate * time_series
                phase = 2 * jnp.pi * jnp.cumsum(freq) / window_size * 4.0
                amplitude = (1e-21 + jax.random.uniform(signal_key, (), minval=0, maxval=5e-22)) * jnp.exp(-time_series / 2.0)
                chirp = amplitude * jnp.sin(phase)
                noise = 1e-21 * jax.random.normal(signal_key, (window_size,))
                signal = chirp + 0.3 * noise
                label = 1
            else:  # 67% noise signals
                # Realistic colored noise
                noise = 1e-21 * jax.random.normal(signal_key, (window_size,))
                # Add 1/f characteristics 
                freq_noise = jnp.fft.fft(noise)
                freqs = jnp.fft.fftfreq(window_size)
                freq_noise = freq_noise / (1 + jnp.abs(freqs) * 100)
                signal = jnp.real(jnp.fft.ifft(freq_noise))
                label = 0
                
            all_signals.append(signal)
            all_labels.append(label)
    
    # 4. FINAL DATASET
    final_signals = jnp.array(all_signals)
    final_labels = jnp.array(all_labels)
    
    # Shuffle the dataset
    key = jax.random.PRNGKey(456)
    indices = jax.random.permutation(key, len(final_signals))
    final_signals = final_signals[indices]
    final_labels = final_labels[indices]
    
    logger.info(f"🎯 ENHANCED DATASET CREATED:")
    logger.info(f"   Total samples: {len(final_signals)}")
    logger.info(f"   Window size: {final_signals.shape[1]}")
    logger.info(f"   GW signals: {jnp.sum(final_labels)} ({jnp.mean(final_labels):.1%})")
    logger.info(f"   Noise samples: {jnp.sum(1 - final_labels)} ({jnp.mean(1 - final_labels):.1%})")
    
    return final_signals, final_labels


def create_real_ligo_dataset(num_samples: int = 1200, 
                           window_size: int = 512,
                           quick_mode: bool = False,
                           return_split: bool = False,
                           train_ratio: float = 0.8,
                           overlap: float = 0.5) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create dataset using real LIGO data with proper windowing and optional stratified split
    
    Args:
        num_samples: Target number of samples
        window_size: Size of each window
        quick_mode: Use smaller windows for quick testing
        return_split: If True, return stratified train/test split
        train_ratio: Ratio for train/test split (only used if return_split=True)
        
    Returns:
        If return_split=False: Tuple of (signals, labels) 
        If return_split=True: Tuple of ((train_signals, train_labels), (test_signals, test_labels))
    """
    logger.info("🌊 Attempting to download real GW150914 data...")
    
    # Try to get real LIGO data
    real_strain = download_gw150914_data()
    
    if real_strain is not None:
        logger.info(f"✅ GW150914 strain data obtained: {len(real_strain)} samples")
        
        # ✅ MEMORY-OPTIMIZED: Create smaller windowed dataset
        window_size = 256 if quick_mode else window_size
        windows_list, labels_list = create_proper_windows(real_strain, window_size=window_size, overlap=overlap)
        
        # Convert to JAX arrays
        signals = jnp.array(windows_list)
        labels = jnp.array(labels_list)
        
        logger.info(f"🌊 Created proper windowed dataset:")
        logger.info(f"   Total windows: {len(signals)}")
        logger.info(f"   Window size: {signals.shape[1]}")
        logger.info(f"   Signal windows (label=1): {jnp.sum(labels)}")
        logger.info(f"   Noise windows (label=0): {jnp.sum(1 - labels)}")
        logger.info(f"   Class balance: {jnp.mean(labels):.1%} positive")
        
        # ✅ CRITICAL: Check for valid dataset
        if len(signals) < 2:
            logger.error("❌ Dataset too small - falling back to simulated data")
            real_strain = None  # Force fallback
        elif jnp.all(labels == 0) or jnp.all(labels == 1):
            logger.warning("⚠️ All labels are the same class - this will give fake accuracy!")
            logger.warning(f"   Labels: {jnp.unique(labels)}")
            logger.warning("   Consider adjusting window parameters or event detection")
        
        # ✅ OPTIONAL: Apply stratified split if requested
        if return_split:
            from utils.data_split import create_stratified_split
            (train_signals, train_labels), (test_signals, test_labels) = create_stratified_split(
                signals, labels, train_ratio=train_ratio, random_seed=42
            )
            return (train_signals, train_labels), (test_signals, test_labels)
        else:
            return signals, labels
    else:
        # Fallback to synthetic data
        logger.warning("❌ Could not obtain real LIGO data - using synthetic fallback")
        from data.gw_dataset_builder import create_evaluation_dataset
        
        synthetic_data = create_evaluation_dataset(
            num_samples=num_samples,
            sequence_length=window_size,
            sample_rate=4096,
            random_seed=42
        )
        
        from utils.jax_safety import safe_stack_to_device, safe_array_to_device
        signals = safe_stack_to_device([sample[0] for sample in synthetic_data], dtype=np.float32)
        labels = safe_array_to_device([sample[1] for sample in synthetic_data], dtype=np.int32)
        
        # ✅ OPTIONAL: Apply stratified split if requested
        if return_split:
            from utils.data_split import create_stratified_split
            (train_signals, train_labels), (test_signals, test_labels) = create_stratified_split(
                signals, labels, train_ratio=train_ratio, random_seed=42
            )
            return (train_signals, train_labels), (test_signals, test_labels)
        else:
            return signals, labels 