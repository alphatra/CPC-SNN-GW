import h5py
import numpy as np
import torch
import scipy.signal
import os

def verify_reconstruction():
    # Parameters matches Data Generation
    fs = 4096.0
    window_sec = 0.25
    overlap_frac = 0.5
    
    nperseg = int(window_sec * fs) # 1024
    noverlap = int(nperseg * overlap_frac) # 512
    
    print(f"Verifying reconstruction with: nperseg={nperseg}, noverlap={noverlap}, fs={fs}")
    
    # Load Data
    h5_path = "data/cpc_snn_train.h5"
    if not os.path.exists(h5_path):
        print("H5 file not found.")
        return

    with h5py.File(h5_path, 'r') as h5:
        # Get first available key
        gid = list(h5.keys())[0]
        print(f"Testing on sample ID: {gid}")
        
        grp = h5[gid]["H1"]
        mag = grp["mag"][()]
        cos = grp["cos"][()]
        sin = grp["sin"][()]
        f = h5[gid]["f"][()] # Frequencies
        
    # --- SCIPY RECONSTRUCTION (Ref) ---
    Zxx_complex = mag * (cos + 1j * sin)
    
    # Map to full spectrum
    k_indices = np.round(f * nperseg / fs).astype(int)
    n_freq_full = nperseg // 2 + 1
    
    T_bins = Zxx_complex.shape[0]
    Zxx_full = np.zeros((T_bins, n_freq_full), dtype=np.complex64)
    
    valid = (k_indices >= 0) & (k_indices < n_freq_full)
    Zxx_full[:, k_indices[valid]] = Zxx_complex[:, valid]
    Zxx_full_T = Zxx_full.T
    
    t, x_scipy = scipy.signal.istft(
        Zxx_full_T,
        fs=fs,
        window='hann',
        nperseg=1024,
        noverlap=512,
        input_onesided=True,
        boundary=True # Matches center=True
    )
    
    # --- TORCH RECONSTRUCTION ---
    Zxx_torch = torch.from_numpy(Zxx_full_T).unsqueeze(0)
    window_torch = torch.hann_window(1024)
    
    x_torch = torch.istft(
        Zxx_torch,
        n_fft=1024,
        hop_length=512,
        win_length=1024,
        window=window_torch,
        center=True,
        normalized=False,
        onesided=True,
        length=None
    )
    
    s_scipy = x_scipy
    s_torch = x_torch.squeeze().numpy()
    
    print(f"Scipy raw shape: {s_scipy.shape}")
    print(f"Torch raw shape: {s_torch.shape}")
    
    # Check RMS scaling
    rms_scipy = np.sqrt(np.mean(s_scipy**2))
    rms_torch = np.sqrt(np.mean(s_torch**2))
    
    scaling_factor = rms_scipy / (rms_torch + 1e-9)
    print(f"RMS Ratio (Scipy/Torch): {scaling_factor:.4f}")
    
    # Estimate nperseg/2 scaling?
    print(f"nperseg/2: {1024/2}")
    
    # Align
    lags = scipy.signal.correlation_lags(len(s_scipy), len(s_torch))
    corr = scipy.signal.correlate(s_scipy, s_torch * scaling_factor)
    best_lag = lags[np.argmax(corr)]
    
    print(f"Optimal Lag: {best_lag}")
    
    # Apply corrections
    s_torch_corrected = s_torch * scaling_factor
    
    # If lag is 0, they are aligned
    if abs(best_lag) < 5:
        print("Lag is negligible. Signals aligned!")
        mse = np.mean((s_scipy - s_torch_corrected)**2)
        final_corr = np.corrcoef(s_scipy, s_torch_corrected)[0, 1]
        print(f"Final Correlation: {final_corr:.6f}")
        print(f"Final MSE: {mse:.6e}")
        if final_corr > 0.99:
            print("SUCCESS: Reconstruction consistency verified (with scaling).")
    else:
        print("Lag is significant. Check alignment.")

if __name__ == "__main__":
    verify_reconstruction()
