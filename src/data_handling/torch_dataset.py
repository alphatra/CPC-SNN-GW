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
                center=True, 
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