from __future__ import annotations
from typing import Dict, List, Union, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

class HDF5SFTPairDataset(Dataset):
    """
    Dataset for a pair of detectors (e.g., H1, L1) with stored T-F (SFT/STFT).

    HDF5 Structure (per sample /id):
      /id/H1/mag      [T,F]
      /id/H1/cos      [T,F]
      /id/H1/sin      [T,F]
      /id/L1/mag      [T,F]
      /id/L1/cos      [T,F]
      /id/L1/sin      [T,F]
      /id/f           [F]
      /id/mask_H1     [T]  (0/1)
      /id/mask_L1     [T]  (0/1)
      /id/label       scalar (0.0 or 1.0)

    Parameters:
      use_phase:       whether to use [mag, cosφ, sinφ] (True) or only [mag] (False)
      add_mask_channel:whether to include mask as an additional 2D channel
      enforce_same_shape: crops H1/L1 to common T if necessary
    """

    def __init__(
        self,
        h5_path: str,
        index_list: List[Union[str, int]],
        use_phase: bool = True,
        add_mask_channel: bool = True,
        enforce_same_shape: bool = True,
        dtype: torch.dtype = torch.float32,
        return_time_series: bool = False,
        fs: float = 4096.0,
        window_sec: float = 0.25,
        overlap_frac: float = 0.5,
        injection_mode: bool = False,
        signal_bank: Optional[List[Dict[str, np.ndarray]]] = None
    ) -> None:
        super().__init__()
        self.h5_path = h5_path
        self.ids = [str(i) for i in index_list]
        self.use_phase = use_phase
        self.add_mask_channel = add_mask_channel
        self.enforce_same_shape = enforce_same_shape
        self.dtype = dtype
        self.return_time_series = return_time_series
        self.fs = fs
        self.window_sec = window_sec
        self.overlap_frac = overlap_frac
        
        # Injection params
        self.injection_mode = injection_mode
        self.signal_bank = signal_bank
        self.current_snr = 20.0 # Default, can be updated via set_snr()

    def set_snr(self, snr: float):
        self.current_snr = snr

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def _read_ifo(h5: h5py.File, gid: str, ifo: str) -> Dict[str, np.ndarray]:
        g = h5[f"{gid}/{ifo}"]
        mag = g["mag"][()]
        if "cos" in g and "sin" in g:
            cos = g["cos"][()]
            sin = g["sin"][()]
        elif "re" in g and "im" in g:
            re = g["re"][()]
            im = g["im"][()]
            mag = np.sqrt(re**2 + im**2)
            pha = np.arctan2(im, re)
            cos = np.cos(pha)
            sin = np.sin(pha)
        else:
            raise KeyError(f"Missing [cos,sin] or [re,im] in group {gid}/{ifo}")
        return {"mag": mag, "cos": cos, "sin": sin}

    def _reconstruct_time_series(self, mag: np.ndarray, cos: np.ndarray, sin: np.ndarray, f_stored: np.ndarray) -> torch.Tensor:
        from scipy.signal import istft
        
        # Reconstruct complex STFT (Time, Freq_stored)
        Zxx_stored = mag * (cos + 1j * sin)
        
        nperseg = int(self.window_sec * self.fs)
        noverlap = int(nperseg * self.overlap_frac)
        
        # Full frequency range
        # stft returns nperseg//2 + 1 bins for onesided
        n_freq_full = nperseg // 2 + 1
        
        # Create full Zxx (Time, Freq_full)
        # But we need (Freq_full, Time) for istft?
        # Let's work in (Time, Freq) first then transpose.
        T = Zxx_stored.shape[0]
        Zxx_full = np.zeros((T, n_freq_full), dtype=np.complex64)
        
        # Map stored frequencies to indices
        # f = k * fs / nperseg
        # k = f * nperseg / fs
        k_indices = np.round(f_stored * nperseg / self.fs).astype(int)
        
        # Filter valid indices (just in case)
        valid = (k_indices >= 0) & (k_indices < n_freq_full)
        
        Zxx_full[:, k_indices[valid]] = Zxx_stored[:, valid]
        
        # Transpose to (Freq, Time) for istft
        Zxx_full = Zxx_full.T
        
        _, x_rec = istft(
            Zxx_full,
            fs=self.fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            input_onesided=True,
            boundary=None # Match stft params
        )
        
        return torch.from_numpy(x_rec).to(self.dtype)


    def _pack_channels(
        self,
        mag: np.ndarray,
        cos: np.ndarray,
        sin: np.ndarray,
        mask_t: np.ndarray,
    ) -> torch.Tensor:
        T, F = mag.shape
        channels = []
        if self.use_phase:
            channels.extend([mag, cos, sin])
        else:
            channels.append(mag)

        if self.add_mask_channel:
            # mask_t: [T] -> [T,F]
            mask_2d = np.repeat(mask_t[:, None], F, axis=1)
            channels.append(mask_2d)

        x = np.stack(channels, axis=0)  # [C,T,F]
        return torch.from_numpy(x).to(self.dtype)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gid = self.ids[idx]
        with h5py.File(self.h5_path, "r") as h5:
            H1 = self._read_ifo(h5, gid, "H1")
            L1 = self._read_ifo(h5, gid, "L1")
            mask_H1 = h5[f"{gid}/mask_H1"][()]
            mask_L1 = h5[f"{gid}/mask_L1"][()]
            f = h5[f"{gid}/f"][()]
            y = h5[f"{gid}/label"][()] if f"{gid}/label" in h5 else None

        # --- On-the-Fly Injection Logic Removed ---
        # Injection is now handled on GPU in the training loop for performance.

        if self.enforce_same_shape:
            T = min(H1["mag"].shape[0], L1["mag"].shape[0],
                    mask_H1.shape[0], mask_L1.shape[0])
            for d in (H1, L1):
                for k in ("mag", "cos", "sin"):
                    d[k] = d[k][:T]
            mask_H1 = mask_H1[:T]
            mask_L1 = mask_L1[:T]

        if self.return_time_series:
            # Reconstruct time series
            ts_H1 = self._reconstruct_time_series(H1["mag"], H1["cos"], H1["sin"], f)
            ts_L1 = self._reconstruct_time_series(L1["mag"], L1["cos"], L1["sin"], f)
            
            # Ensure same length

            min_len = min(ts_H1.shape[0], ts_L1.shape[0])
            ts_H1 = ts_H1[:min_len]
            ts_L1 = ts_L1[:min_len]
            
            # Stack: [Channels, Time]
            x_out = torch.stack([ts_H1, ts_L1], dim=0)
            
            # Mask? We don't have time-domain mask easily, but we can interpolate or ignore.
            # For now, ignore mask for time series.
            
            out = {"y": x_out} # Use 'y' key for input signal to match smoke test expectation? 
            # No, usually 'x' is input, 'y' is label.
            # But smoke test used x = batch['y'].
            # Let's stick to standard: 'x' = input, 'y' = label.
            out["x"] = x_out
            
        else:
            out = {
                "H1": self._pack_channels(H1["mag"], H1["cos"], H1["sin"], mask_H1),
                "L1": self._pack_channels(L1["mag"], L1["cos"], L1["sin"], mask_L1),
                "mask_H1": torch.from_numpy(mask_H1).to(self.dtype),
                "mask_L1": torch.from_numpy(mask_L1).to(self.dtype),
                "f": torch.from_numpy(f).to(self.dtype),
            }

        if y is not None:
            out["label"] = torch.tensor(float(y), dtype=self.dtype) # Rename to 'label' to avoid confusion

        return out

    @staticmethod
    def batch_reconstruct_torch(batch: Dict[str, torch.Tensor], fs: float = 4096.0, window_sec: float = 0.25, overlap_frac: float = 0.5, device=torch.device('cpu')) -> torch.Tensor:
        """
        Reconstructs time series from a batch of SFT data using torch.istft on the specified device.
        
        Args:
            batch: Dictionary from DataLoader (return_time_series=False).
                   Keys: "H1", "L1", "f", ...
                   H1/L1 shape: (B, C, T_sft, F_stored)
                   f shape: (B, F_stored)
            fs: Sampling frequency.
            window_sec: Window length in seconds.
            overlap_frac: Overlap fraction.
            device: Target device (e.g., 'mps', 'cuda').
            
        Returns:
            x_out: (B, 2, Time)
        """
        # Unpack
        # H1: (B, C, T_bins, F_bins)
        # We need to handle H1 and L1
        
        recons = []
        for ifo in ["H1", "L1"]:
            data = batch[ifo].to(device) # (B, C, T_bins, F_bins)
            f_batch = batch["f"].to(device) # (B, F_bins)
            
            # Assuming C=3 (mag, cos, sin) or C=4 (mag, cos, sin, mask)
            # We need mag, cos, sin. They are at indices 0, 1, 2.
            mag = data[:, 0, :, :]
            cos = data[:, 1, :, :]
            sin = data[:, 2, :, :]
            
            # Zxx_stored: (B, T_bins, F_bins) (Complex)
            Zxx_stored = torch.complex(mag * cos, mag * sin)
            
            B, T_bins, F_stored = Zxx_stored.shape
            
            nperseg = int(window_sec * fs)
            noverlap = int(nperseg * overlap_frac)
            n_freq_full = nperseg // 2 + 1
            
            # Prepare full Zxx: (B, F_full, T_bins) for torch.istft
            # Note: torch.istft expects (B, F, T)
            
            Zxx_full = torch.zeros((B, n_freq_full, T_bins), dtype=torch.complex64, device=device)
            
            # Map frequencies
            # We assume f is same for all batch items, take first
            f_vals = f_batch[0] # (F_stored)
            k_indices = torch.round(f_vals * nperseg / fs).long()
            
            # Filter valid
            valid_mask = (k_indices >= 0) & (k_indices < n_freq_full)
            valid_k = k_indices[valid_mask]
            
            # Assign
            # Zxx_stored is (B, T, F). Transpose to (B, F, T)
            Zxx_stored_t = Zxx_stored.permute(0, 2, 1) # (B, F_stored, T)
            
            Zxx_full[:, valid_k, :] = Zxx_stored_t[:, valid_mask, :]
            
            # iSTFT
            # Use center=True to fix NOLA condition (padding at boundaries)
            # And run on device (MPS) for speed
            window = torch.hann_window(nperseg, device=device)
            
            x_rec = torch.istft(
                Zxx_full,
                n_fft=nperseg,
                hop_length=nperseg - noverlap,
                win_length=nperseg,
                window=window,
                center=False, 
                normalized=False,
                onesided=True,
                return_complex=False
            )
            
            recons.append(x_rec)
            
        # Stack H1, L1 -> (B, 2, Time)
        # Check lengths
        min_len = min(recons[0].shape[-1], recons[1].shape[-1])
        x_out = torch.stack([recons[0][..., :min_len], recons[1][..., :min_len]], dim=1)
        
        return x_out

class HDF5TimeSeriesDataset(Dataset):
    """
    Dataset for pre-processed Time Series data (H1, L1).
    Structure:
      /id/H1 [T]
      /id/L1 [T]
      /id/label scalar
    """
    def __init__(self, h5_path, index_list, dtype=torch.float32):
        self.h5_path = h5_path
        self.ids = [str(i) for i in index_list]
        self.dtype = dtype
        
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, idx):
        gid = self.ids[idx]
        with h5py.File(self.h5_path, "r") as h5:
            H1 = h5[f"{gid}/H1"][()]
            L1 = h5[f"{gid}/L1"][()]
            y = h5[f"{gid}/label"][()] if f"{gid}/label" in h5 else None
            
        # Stack: [2, T]
        x = np.stack([H1, L1], axis=0)
        
        out = {
            "x": torch.from_numpy(x).to(self.dtype)
        }
        
        if y is not None:
            out["label"] = torch.tensor(float(y), dtype=self.dtype)
            
        return out

class InMemoryHDF5Dataset(Dataset):
    """
    Loads the entire HDF5 dataset into RAM for zero-copy access.
    Best for datasets that fit in memory (<16GB).
    """
    def __init__(self, h5_path, index_list, device='cpu', use_phase=True, add_mask_channel=True, dtype=torch.float32, fs=4096.0, window_sec=0.25, overlap_frac=0.5):
        print(f"Loading entire dataset from {h5_path} into RAM...")
        self.data = []
        self.dtype = dtype
        self.use_phase = use_phase
        self.add_mask_channel = add_mask_channel
        self.fs = fs
        self.window_sec = window_sec
        self.overlap_frac = overlap_frac
        
        # Injection params (placeholder for compatibility)
        self.injection_mode = False
        self.signal_bank = None
        self.current_snr = 20.0
        
        from tqdm import tqdm
        
        with h5py.File(h5_path, 'r') as h5:
            for idx in tqdm(index_list, desc="Loading to RAM"):
                gid = str(idx)
                
                # Read H1
                h1_g = h5[f"{gid}/H1"]
                h1_mag = h1_g["mag"][()]
                
                # Determine T
                T = h1_mag.shape[0]
                
                # Read L1
                l1_g = h5[f"{gid}/L1"]
                l1_mag = l1_g["mag"][()]
                
                # Read Masks
                mask_h1 = h5[f"{gid}/mask_H1"][()]
                mask_l1 = h5[f"{gid}/mask_L1"][()]
                
                # Crop to min T
                T = min(h1_mag.shape[0], l1_mag.shape[0], mask_h1.shape[0], mask_l1.shape[0])
                
                # Helper to pack channels
                def pack(g, mask, t_len):
                    mag = g["mag"][:t_len]
                    if "cos" in g:
                        cos = g["cos"][:t_len]
                        sin = g["sin"][:t_len]
                    else:
                        # Re/Im fallback
                        re = g["re"][:t_len]
                        im = g["im"][:t_len]
                        pha = np.arctan2(im, re)
                        cos = np.cos(pha)
                        sin = np.sin(pha)
                        
                    channels = [mag]
                    if use_phase:
                        channels.extend([cos, sin])
                    if add_mask_channel:
                         mask_2d = np.repeat(mask[:t_len, None], mag.shape[1], axis=1)
                         channels.append(mask_2d)
                    return np.stack(channels, axis=0) # (C, T, F)

                h1_packed = pack(h1_g, mask_h1, T)
                l1_packed = pack(l1_g, mask_l1, T)
                
                sample = {
                    "H1": torch.from_numpy(h1_packed).to(dtype),
                    "L1": torch.from_numpy(l1_packed).to(dtype),
                    "f": torch.from_numpy(h5[f"{gid}/f"][()]).to(dtype),
                    "mask_H1": torch.from_numpy(mask_h1[:T]).to(dtype),
                    "mask_L1": torch.from_numpy(mask_l1[:T]).to(dtype)
                }
                
                if f"{gid}/label" in h5:
                    sample["label"] = torch.tensor(float(h5[f"{gid}/label"][()]), dtype=dtype)
                    
                self.data.append(sample)
                
        print(f"Loaded {len(self.data)} samples.")

    def set_snr(self, snr: float):
        self.current_snr = snr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class InMemoryGPU_Dataset(Dataset):
    """
    Loads the entire dataset directly into GPU memory (MPS/CUDA) as complex tensors.
    Eliminates CPU-GPU transfer latency.
    Maps cropped SFTs to full frequency grid (513 bins) for compatibility with injection.
    """
    def __init__(self, h5_path, index_list, device, fs=4096.0, window_sec=0.25, overlap_frac=0.5):
        print(f"Loading {len(index_list)} samples directly to {device}...")
        
        # Constants for Full SFT
        nperseg = int(window_sec * fs)
        n_freq_full = nperseg // 2 + 1
        
        temp_data = []
        
        with h5py.File(h5_path, 'r') as h5:
            from tqdm import tqdm
            for idx in tqdm(index_list, desc="Loading to RAM"):
                gid = str(idx)
                
                # Read f to map frequencies
                f = h5[f"{gid}/f"][()]
                
                # Calculate indices
                k_indices = np.round(f * nperseg / fs).astype(int)
                valid_mask = (k_indices >= 0) & (k_indices < n_freq_full)
                valid_k = k_indices[valid_mask]
                
                # Read H1
                h1_mag = torch.from_numpy(h5[f"{gid}/H1"]['mag'][()])
                T = h1_mag.shape[0]
                
                if "cos" in h5[f"{gid}/H1"]:
                    h1_cos = torch.from_numpy(h5[f"{gid}/H1"]['cos'][()])
                    h1_sin = torch.from_numpy(h5[f"{gid}/H1"]['sin'][()])
                else:
                    re = torch.from_numpy(h5[f"{gid}/H1"]['re'][()])
                    im = torch.from_numpy(h5[f"{gid}/H1"]['im'][()])
                    pha = torch.atan2(im, re)
                    h1_cos = torch.cos(pha)
                    h1_sin = torch.sin(pha)
                    
                # H1 Complex: (T, F_stored)
                h1_c = torch.complex(h1_mag * h1_cos, h1_mag * h1_sin)
                
                # Read L1
                l1_mag = torch.from_numpy(h5[f"{gid}/L1"]['mag'][()])
                if "cos" in h5[f"{gid}/L1"]:
                    l1_cos = torch.from_numpy(h5[f"{gid}/L1"]['cos'][()])
                    l1_sin = torch.from_numpy(h5[f"{gid}/L1"]['sin'][()])
                else:
                    re = torch.from_numpy(h5[f"{gid}/L1"]['re'][()])
                    im = torch.from_numpy(h5[f"{gid}/L1"]['im'][()])
                    pha = torch.atan2(im, re)
                    l1_cos = torch.cos(pha)
                    l1_sin = torch.sin(pha)
                    
                # L1 Complex: (T, F_stored)
                l1_c = torch.complex(l1_mag * l1_cos, l1_mag * l1_sin)
                
                # Ensure same T
                T = min(h1_c.shape[0], l1_c.shape[0])
                h1_c = h1_c[:T, :]
                l1_c = l1_c[:T, :]
                
                # Map to Full SFT: (2, F_full, T)
                # Create zero tensor
                full_sft = torch.zeros((2, n_freq_full, T), dtype=torch.complex64)
                
                # Assign H1 (transpose to F, T)
                full_sft[0, valid_k, :] = h1_c[:, valid_mask].T
                
                # Assign L1
                full_sft[1, valid_k, :] = l1_c[:, valid_mask].T
                
                temp_data.append(full_sft)

        # Convert to single tensor and move to device
        print("Moving data to GPU memory...")
        # Stack: (N, 2, F_full, T)
        self.data_complex = torch.stack(temp_data).to(device)
        
        print(f"Dataset ready on {device}: {self.data_complex.shape} ({self.data_complex.element_size() * self.data_complex.numel() / 1e9:.2f} GB)")

    def __len__(self):
        return len(self.data_complex)

    def __getitem__(self, idx):
        # Returns dictionary with 'x' as complex tensor on GPU
        # and 'label' as 0.0 (noise)
        return {
            "x": self.data_complex[idx], 
            "label": torch.tensor(0.0, device=self.data_complex.device)
        }