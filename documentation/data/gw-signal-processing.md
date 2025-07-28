# Gravitational Wave Signal Processing

## Signal Characteristics

Gravitational wave (GW) signals from compact binary coalescences (like GW150914) have distinct characteristics that are crucial for their detection. The primary signal type is a "chirp," which is a waveform that increases in both frequency and amplitude over time as the two objects spiral closer together and eventually merge.


The key parameters that define a GW signal are:

*   **Chirp Mass (M_c)**: A combination of the two component masses that determines the rate of frequency increase (the "chirp rate").
*   **Distance (D_L)**: The luminosity distance to the source, which affects the overall amplitude of the signal.
*   **Phase (φ)**: The initial phase of the waveform.
*   **Time of Coalescence (t_c)**: The GPS time at which the two objects merge.


The strain signal `h(t)` can be approximated by a post-Newtonian (PN) expansion, but for simplicity and robustness, the system uses a phenomenological model based on a frequency-modulated sine wave.


## Preprocessing Pipeline

The raw strain data from the LIGO detectors is heavily contaminated with noise, including instrumental noise and environmental disturbances (glitches). The preprocessing pipeline is designed to clean the data and prepare it for the model.


### 1. Data Loading and Combination
The first step is to load the raw strain data from the H1 and L1 detectors using the ReadLIGO library. The data from both detectors is combined by taking their average. This simple combination enhances the signal-to-noise ratio (SNR) because the astrophysical signal is correlated between the two detectors, while much of the noise is uncorrelated.


### 2. Whitening
Whitening is a critical preprocessing step that flattens the power spectral density (PSD) of the data. LIGO data has a highly non-uniform noise spectrum, with much more power at low frequencies. Whitening transforms the data so that all frequency components have equal power, making it easier for the model to learn features across the entire frequency band.


The whitening process involves:
1.  Estimating the PSD of a long segment of noise data.
2.  Taking the Fourier transform of the data segment.
3.  Dividing the Fourier coefficients by the square root of the PSD at the corresponding frequency.
4.  Taking the inverse Fourier transform to return to the time domain.


### 3. Bandpass Filtering
After whitening, a bandpass filter is applied to isolate the frequency band where GW signals are expected (e.g., 20-500 Hz). This removes very low-frequency drift and high-frequency noise that are not relevant for the detection task.


### 4. Windowing and Labeling
The continuous, preprocessed strain data is then divided into overlapping windows of a fixed length (e.g., 256 or 512 samples). Each window is labeled as positive (1) if it contains the time of coalescence `t_c` within a certain margin, and negative (0) otherwise. This creates a supervised learning dataset from the raw time series.


## Implementation

The core signal processing functions are implemented in the `data` module, particularly in `gw_preprocessor.py` and `real_ligo_integration.py`.


```python
import numpy as np
import scipy.signal as signal
from scipy import fft

def estimate_psd(strain_data: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the Power Spectral Density (PSD) using Welch's method."""
    freqs, psd = signal.welch(strain_data, fs=sample_rate, nperseg=4096)
    return freqs, psd

def whiten(strain_data: np.ndarray, sample_rate: float, psd: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Whiten the strain data using the provided PSD."""
    # Interpolate PSD to match the frequency resolution of the FFT
    psd_interp = np.interp(fft.rfftfreq(len(strain_data), 1/sample_rate), freqs, psd)
    
    # Take the Fourier transform
    strain_fft = fft.rfft(strain_data)
    
    # Divide by the square root of the PSD (with a small epsilon for stability)
    strain_fft_whitened = strain_fft / np.sqrt(psd_interp + 1e-10)
    
    # Inverse Fourier transform
    strain_whitened = fft.irfft(strain_fft_whitened, n=len(strain_data))
    
    return strain_whitened

def bandpass_filter(strain_data: np.ndarray, sample_rate: float, lowcut: float = 20.0, highcut: float = 500.0) -> np.ndarray:
    """Apply a bandpass filter to the strain data."""
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    strain_filtered = signal.filtfilt(b, a, strain_data)
    return strain_filtered

def create_windows(strain_data: np.ndarray, window_size: int = 256, overlap: float = 0.5) -> np.ndarray:
    """Create overlapping windows from the strain data."""
    step = int(window_size * (1 - overlap))
    windows = []
    for start in range(0, len(strain_data) - window_size + 1, step):
        windows.append(strain_data[start:start + window_size])
    return np.array(windows)

def label_windows(windows: np.ndarray, window_times: np.ndarray, event_time: float, margin: float = 0.1) -> np.ndarray:
    """Label windows as 1 (signal) if they contain the event time, else 0 (noise)."""
    labels = np.zeros(len(windows))
    # window_times[i] is the start time of windows[i]
    end_times = window_times + (len(windows[0]) / 4096)  # Assuming 4096 Hz sample rate
    # A window contains the event if event_time is between its start and end
    labels = ((window_times <= event_time + margin) & (end_times >= event_time - margin)).astype(int)
    return labels
```

## Usage

These functions are orchestrated by the `create_real_ligo_dataset` function to create the final training dataset.

```python
# Example of the full preprocessing pipeline for a single segment
sample_rate = 4096.0  # Hz

# 1. Load and combine data (example)
# strain_h1, time_h1 = ... # From ReadLIGO
# strain_l1, time_l1 = ... # From ReadLIGO
# combined_strain = (strain_h1 + strain_l1) / 2.0

# 2. Estimate PSD on a noise segment (e.g., before the event)
noise_segment = combined_strain[:10000]  # First 2.4 seconds
freqs, psd = estimate_psd(noise_segment, sample_rate)

# 3. Whiten the entire data segment
strain_whitened = whiten(combined_strain, sample_rate, psd, freqs)

# 4. Apply bandpass filter
strain_filtered = bandpass_filter(strain_whitened, sample_rate)

# 5. Create overlapping windows
windows = create_windows(strain_filtered, window_size=256, overlap=0.5)

# 6. Label the windows
event_time = 1126259462.4  # GPS time of GW150914
time_per_sample = 1 / sample_rate
window_start_times = time_h1[::128]  # Start time of each window (step=128 for 50% overlap)
labels = label_windows(windows, window_start_times, event_time)

print(f"Created {len(windows)} windows, {np.sum(labels)} labeled as signal")
```

This preprocessing pipeline transforms the raw, noisy detector output into a clean, structured dataset that is suitable for training the neuromorphic model.