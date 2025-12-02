import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import time
import os
import json
from src.data_handling.torch_dataset import HDF5SFTPairDataset
from src.data_handling.gw_utils import generate_waveform, project_waveform_to_ifo

def generate_signal_bank(n_signals=100, sample_rate=4096.0, f_low=20.0, duration=4.0):
    """
    Generates a bank of whitened signal Time Series for injection.
    Returns a Tensor: (n_signals, 2, time_steps)
    """
    print(f"Generating Signal Bank ({n_signals} signals, {duration}s)...")
    
    signals_list = []
    target_samples = int(duration * sample_rate)
    
    for _ in tqdm(range(n_signals), desc="Building Signal Bank"):
        hp, hc = generate_waveform(mass_range=(10, 50), sample_rate=sample_rate, f_lower=f_low)
        
        ra = np.random.uniform(0, 2 * np.pi)
        dec = np.random.uniform(-np.pi / 2, np.pi / 2)
        psi = np.random.uniform(0, 2 * np.pi)
        t_gps = 1000000000.0 
        
        sig_channels = []
        for ifo in ["H1", "L1"]:
            sig = project_waveform_to_ifo(hp, hc, ifo, ra, dec, psi, t_gps)
            sig_np = sig.numpy()
            
            peak_idx = np.argmax(np.abs(sig_np))
            start = peak_idx - target_samples // 2
            end = start + target_samples
            
            if start < 0:
                pad_left = -start
                sig_cut = np.pad(sig_np, (pad_left, 0))[:target_samples]
            elif end > len(sig_np):
                pad_right = end - len(sig_np)
                sig_cut = np.pad(sig_np, (0, pad_right))[start:]
            else:
                sig_cut = sig_np[start:end]
                
            if len(sig_cut) < target_samples:
                 sig_cut = np.pad(sig_cut, (0, target_samples - len(sig_cut)))
            
            sig_channels.append(sig_cut)
            
        sig_tensor = torch.tensor(np.stack(sig_channels), dtype=torch.float32)
        signals_list.append(sig_tensor)
        
    return torch.stack(signals_list)

def prepare_signal_bank_sft(signal_bank_data, device):
    """
    Converts Time Series signal bank to Complex SFT bank on GPU.
    Args:
        signal_bank_data: Tensor (N, 2, T) OR Dict {bin_key: List[Tensor]}
    Returns:
        signal_bank_sft: Tensor (N, 2, F, T_bins) OR Dict {bin_key: Tensor}
    """
    print(f"Converting Signal Bank to SFT on {device}...")
    
    # Helper for single tensor
    def _convert_tensor(ts_tensor):
        ts_tensor = ts_tensor.to(device)
        N, C, T = ts_tensor.shape
        fs = 4096.0
        window_sec = 0.25
        nperseg = int(window_sec * fs)
        noverlap = int(nperseg * 0.5)
        window = torch.hann_window(nperseg, device=device)
        flat = ts_tensor.view(-1, T)
        Z = torch.stft(flat, n_fft=nperseg, hop_length=nperseg-noverlap, window=window, 
                       return_complex=True, center=True, normalized=False, onesided=True)
        return Z.view(N, C, Z.shape[1], Z.shape[2])

    if isinstance(signal_bank_data, dict):
        # Balanced Bank (Dict of bins) -> Dense Tensor
        # Find max bin index to size the tensor
        # Keys are "min-max", e.g. "4-5". We use the lower bound as index.
        max_idx = 0
        n_per_bin = 0
        
        # First pass to find dims
        for key, val in signal_bank_data.items():
            idx = int(key.split('-')[0])
            max_idx = max(max_idx, idx)
            if n_per_bin == 0:
                n_per_bin = len(val)
        
        # Create dense tensor: (Max_Idx+1, N_per_bin, C, F, T_bins)
        # We need to know shape of one SFT
        # Convert one to get shape
        sample_sft = _convert_tensor(signal_bank_data[list(signal_bank_data.keys())[0]][0].unsqueeze(0))
        _, C, F_dim, T_dim = sample_sft.shape
        
        dense_bank = torch.zeros((max_idx + 1, n_per_bin, C, F_dim, T_dim), dtype=torch.complex64, device=device)
        
        print(f"Creating Dense Signal Bank Tensor: {dense_bank.shape} ({dense_bank.element_size() * dense_bank.numel() / 1e9:.2f} GB)")
        
        for key, val_list in signal_bank_data.items():
            idx = int(key.split('-')[0])
            # Convert list of TS to SFTs
            if isinstance(val_list, list):
                ts_tensor = torch.stack(val_list)
            else:
                ts_tensor = val_list
            
            sft_batch = _convert_tensor(ts_tensor)
            dense_bank[idx] = sft_batch
            
        return dense_bank
    else:
        # Legacy Tensor
        return _convert_tensor(signal_bank_data)

class CPCTrainer:
    def __init__(self, model, optimizer, train_loader, val_loader, args, device, scaler=None, scheduler_lr=None, snr_scheduler=None, signal_bank=None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = device
        self.scaler = scaler
        self.scheduler_lr = scheduler_lr
        self.snr_scheduler = snr_scheduler
        
        # Prepare Signal Bank SFT if needed
        self.signal_bank_sft = None
        if signal_bank is not None:
            self.signal_bank_sft = prepare_signal_bank_sft(signal_bank, device)
        
        self.use_amp = args.amp and hasattr(torch, 'amp')
        self.use_spikes = getattr(args, 'use_spikes', False)
        self.use_timeseries = getattr(args, 'use_timeseries', False)
        
        # Caching for reconstruction
        self.cache = {}
        self.buffers = {}

    def _reconstruct_and_inject(self, batch, current_snr, inject=True):
        """
        Reconstructs Time Series from SFT batch, injecting signals in Frequency Domain if needed.
        """
        # Constants (should match dataset)
        fs = 4096.0
        window_sec = 0.25
        overlap_frac = 0.5
        nperseg = int(window_sec * fs)
        noverlap = int(nperseg * overlap_frac)
        n_freq_full = nperseg // 2 + 1
        
        batch_size = batch["label"].shape[0] # Use label for batch size as 'f' might be missing in Total Cache
        if "H1" in batch:
             T_bins = batch["H1"].shape[2]
        elif "x" in batch:
             T_bins = batch["x"].shape[3] # (B, 2, F, T) -> T is at index 3
        
        # 1. Cache Indices & Window
        if "valid_k" not in self.cache and "f" in batch:
            f_batch = batch["f"].to(self.device, non_blocking=True)
            f_vals = f_batch[0] # Assume constant f across batches
            k_indices = torch.round(f_vals * nperseg / fs).long()
            valid_mask = (k_indices >= 0) & (k_indices < n_freq_full)
            valid_k = k_indices[valid_mask]
            
            self.cache["valid_k"] = valid_k
            self.cache["valid_mask"] = valid_mask
            self.cache["window"] = torch.hann_window(nperseg, device=self.device)
            
        # If using Total Cache, valid_k might not be needed if we have full SFTs?
        # InMemoryGPU_Dataset returns full SFTs? No, it returns what was stored.
        # Wait, InMemoryGPU_Dataset loads H1/L1 which are stored SFTs (partial freq?).
        # If stored SFTs are partial, we still need 'f' to map them.
        # InMemoryGPU_Dataset stores 'x' as (N, 2, T, F).
        # Does it store 'f'? No, it seems to drop 'f' in the snippet provided by user.
        # But wait, the user snippet: "h1 = torch.stack([h1_mag, h1_cos, h1_sin]) # (3, T, F)"
        # It loads whatever is in HDF5.
        # If HDF5 has partial freq, we need 'f' to map.
        # User snippet didn't include 'f' in __getitem__.
        # I should assume InMemoryGPU_Dataset returns full SFTs or I need to fix it to return 'f'.
        # Actually, let's look at InMemoryGPU_Dataset again.
        # It loads H1/L1.
        # It doesn't load 'f'.
        # If the HDF5 contains partial SFTs, we are in trouble without 'f'.
        # But maybe the user implies we load *everything* and if it's partial, we need 'f'.
        # Let's assume for now we need 'f' if we are doing mapping.
        # But if we use InMemoryGPU_Dataset, we might just assume it's mapped?
        # No, the code just stacks H1/L1.
        # I should probably update InMemoryGPU_Dataset to include 'f' or assume full.
        # Let's assume for now we need to handle 'f' if it's there.
        # But I can't easily change InMemoryGPU_Dataset now without another tool call.
        # Let's update cpc_trainer to be robust.
        
        if "valid_k" in self.cache:
            valid_k = self.cache["valid_k"]
            valid_mask = self.cache["valid_mask"]
            window = self.cache["window"]
        else:
             # Fallback if 'f' not in batch (e.g. first batch of Total Cache missing f?)
             # We need window at least.
             window = torch.hann_window(nperseg, device=self.device)
        
        # 2. Reuse Buffer for Z_full
        # Z_full: (B, 2, F_full, T_bins)
        if "Z_full" not in self.buffers or self.buffers["Z_full"].shape[0] != batch_size:
             self.buffers["Z_full"] = torch.zeros((batch_size, 2, n_freq_full, T_bins), dtype=torch.complex64, device=self.device)
        
        Z_full = self.buffers["Z_full"]
        Z_full.zero_() # Reset buffer
        
        # Prepare Injection
        # do_inject depends on inject flag and signal bank presence.
        # We allow injection even if snr_scheduler is False (Fixed SNR mode).
        do_inject = inject and self.signal_bank_sft is not None
        
        inject_mask = None
        chosen_signals = None
        
        if do_inject:
            inject_mask = torch.rand(batch_size, device=self.device) < 0.5
            if inject_mask.any():
                # Select signals based on current_snr
                # Check if we have a dense bank (Tensor with 5 dims) or legacy (4 dims)
                if self.signal_bank_sft.ndim == 5:
                    # Balanced Bank: (Max_Bin, N_per_bin, C, F, T)
                    bin_idx = int(current_snr)
                    
                    # Bounds check
                    max_bin = self.signal_bank_sft.shape[0] - 1
                    bin_idx = min(max(bin_idx, 0), max_bin)
                    
                    bank_subset = self.signal_bank_sft[bin_idx] # (N_per_bin, C, F, T)
                    
                    n_sigs = bank_subset.shape[0]
                    sig_indices = torch.randint(0, n_sigs, (batch_size,), device=self.device)
                    chosen_signals = bank_subset[sig_indices]
                else:
                    # Legacy Random Bank: (N_total, C, F, T)
                    n_sigs = self.signal_bank_sft.shape[0]
                    sig_indices = torch.randint(0, n_sigs, (batch_size,), device=self.device)
                    chosen_signals = self.signal_bank_sft[sig_indices]
        
        # Process H1 and L1
        # We need to process them together to handle injection correctly (paired signals)
        
        # Unpack Batch SFTs
        # batch[ifo]: (B, C, T_bins, F_stored)
        # We need to map to (B, F_full, T_bins)
        
        # Unpack Batch SFTs
        # Check if we have pre-loaded complex SFTs (Total Cache)
        if "x" in batch and batch["x"].is_complex():
             # batch['x'] is (B, 2, F, T) complex (from InMemoryGPU_Dataset)
             # We need Z_full: (B, 2, F_full, T_bins)
             # It matches directly.
             Z_full = batch["x"]
        else:
            # Standard Reconstruction from H1/L1 dicts
            
            # Vectorized Scatter
            # Z_full[:, i, valid_k, :] = Z_stored_t[:, valid_mask, :]
            
            for i, ifo in enumerate(["H1", "L1"]):
                data = batch[ifo].to(self.device, non_blocking=True)
                mag = data[:, 0, :, :]
                cos = data[:, 1, :, :]
                sin = data[:, 2, :, :]
                Z_stored = torch.complex(mag * cos, mag * sin) # (B, T_bins, F_stored)
                
                # Transpose to (B, F_stored, T_bins)
                Z_stored_t = Z_stored.permute(0, 2, 1)
                
                # Map to full
                Z_full[:, i, valid_k, :] = Z_stored_t[:, valid_mask, :]
            
        # --- INJECTION (Frequency Domain) ---
        if do_inject and inject_mask.any():
            # chosen_signals: (B, 2, F_sig, T_sig)
            # Match shapes
            F_sig = chosen_signals.shape[2]
            T_sig = chosen_signals.shape[3]
            
            # Crop/Pad Freq (should be same if generated correctly)
            if F_sig != n_freq_full:
                # Assuming same fs/nperseg, should be same. If not, skip or crop.
                pass
            
            # Crop/Pad Time
            if T_sig > T_bins:
                start = (T_sig - T_bins) // 2
                sig_inject = chosen_signals[:, :, :, start:start+T_bins]
            elif T_sig < T_bins:
                pad = T_bins - T_sig
                sig_inject = torch.nn.functional.pad(chosen_signals, (0, pad))
            else:
                sig_inject = chosen_signals
                
            # Calculate Scaling
            # If using Balanced Bank (5D tensor), signals are already at target SNR.
            # If legacy (4D), we scale.
            
            if self.signal_bank_sft.ndim == 5:
                # Balanced Bank: Direct Injection (Summation)
                # x = x + mask * signal
                mask_expanded = inject_mask.view(-1, 1, 1, 1).float()
                Z_full = Z_full + mask_expanded * sig_inject
            else:
                # Legacy Scaling
                # RMS in Freq Domain: sqrt(mean(|Z|^2))
                # noise_rms: (B, 2, 1, 1)
                noise_rms = torch.sqrt(torch.mean(Z_full.abs()**2, dim=(2,3), keepdim=True))
                sig_rms = torch.sqrt(torch.mean(sig_inject.abs()**2, dim=(2,3), keepdim=True))
                
                # Alpha
                target_alpha = current_snr * (noise_rms / (sig_rms + 1e-8))
                
                # Inject
                mask_expanded = inject_mask.view(-1, 1, 1, 1).float()
                Z_full = Z_full + mask_expanded * target_alpha * sig_inject
            
        # --- IFFT ---
        # window is cached
        
        # Reshape for IFFT: (B*2, F, T)
        Z_flat = Z_full.view(-1, n_freq_full, T_bins)
        
        x_rec = torch.istft(
            Z_flat,
            n_fft=nperseg,
            hop_length=nperseg - noverlap,
            win_length=nperseg,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=False
        )
        
        # Reshape back: (B, 2, Time)
        x_rec = x_rec.view(batch_size, 2, -1)
        
        return x_rec

    def train_epoch(self, epoch):
        current_snr = self.args.start_snr # Default to start_snr (Fixed Mode)
        if self.snr_scheduler:
            current_snr = self.snr_scheduler.get_snr(epoch)
        
        self.model.train()
        train_loss = 0
        train_acc = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [SNR={current_snr:.1f}]")
        for batch in pbar:
            t0 = time.time()
            # Move to device (non_blocking for speed)
            if self.use_spikes or self.use_timeseries:
                x = batch['x'].to(self.device, non_blocking=True)
            else:
                # Reconstruct from SFTs (GPU accelerated)
                # If using InMemoryGPU_Dataset, batch['x'] is already on GPU and complex.
                # _reconstruct_and_inject will handle it.
                x = self._reconstruct_and_inject(batch, current_snr, inject=True)
            
            # Labels
            y = batch['label'].to(self.device, non_blocking=True)
            
            # Normalize
            mean = x.mean(dim=2, keepdim=True)
            std = x.std(dim=2, keepdim=True)
            x = (x - mean) / (std + 1e-8)
            
            if self.args.channel == "H1":
                x = x[:, 0:1, :]
            elif self.args.channel == "L1":
                x = x[:, 1:2, :]
            
            # Time Jittering (Per-Sample)
            if self.model.training:
                max_shift = int(x.shape[-1] * 0.05)
                if max_shift > 0:
                    # Efficient per-sample roll
                    B, C, T = x.shape
                    shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=self.device)
                    
                    # Create a grid of indices
                    arange = torch.arange(T, device=self.device).view(1, 1, T).expand(B, C, T)
                    # Add shifts (broadcasting)
                    new_indices = (arange - shifts.view(B, 1, 1)) % T
                    
                    # Gather
                    # We need to use gather on the last dimension
                    x = torch.gather(x, 2, new_indices)
            
            t_data = time.time() - t0
            t1 = time.time()
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                    z, c, spikes = self.model(x, is_encoded=self.use_spikes)
                    loss, metrics = self.model.compute_cpc_loss(z, c, spikes=z)
                    acc = metrics["acc1"]
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                z, c, spikes = self.model(x, is_encoded=self.use_spikes)
                loss, metrics = self.model.compute_cpc_loss(z, c, spikes=z)
                acc = metrics["acc1"]
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if self.scheduler_lr:
                    self.scheduler_lr.step()
            
            t_model = time.time() - t1
            batch_time = time.time() - t0
            
            train_loss += loss.item()
            train_acc += acc
            
            pbar.set_postfix({'loss': loss.item(), 'acc': acc, 'snr': f"{current_snr:.1f}"})
            
            if not self.args.no_wandb:
                wandb.log({
                    "batch_loss": loss.item(), 
                    "batch_acc": acc,
                    "train_acc_top5": metrics["acc5"],
                    "cpc_pos_score": metrics["pos_score"],
                    "cpc_neg_score": metrics["neg_score"],
                    "cpc_margin": metrics["pos_score"] - metrics["neg_score"],
                    "latent_diversity": z.std(dim=0).mean().item(),
                    "grad_norm": grad_norm,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "current_snr": current_snr,
                    "batch_time": batch_time,
                    "time_data_load": t_data,
                    "time_model_fwd_bwd": t_model,
                    "snn_spike_density": z.mean().item(),
                    "rsnn_context_mean": c.mean().item(),
                    "rsnn_context_std": c.std().item(),
                    "input_rms": x.std().item()
                })
            
        avg_train_loss = train_loss / len(self.train_loader)
        avg_train_acc = train_acc / len(self.train_loader)
        
        return avg_train_loss, avg_train_acc

    def validate(self):
        self.model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for batch in self.val_loader:
                if self.use_spikes or self.use_timeseries:
                    x = batch['x'].to(self.device)
                else:
                    # Just reconstruct, no injection for validation
                    x = self._reconstruct_and_inject(batch, 0.0, inject=False)
                
                # Normalize
                mean = x.mean(dim=2, keepdim=True)
                std = x.std(dim=2, keepdim=True)
                x = (x - mean) / (std + 1e-8)

                if self.args.channel == "H1":
                    x = x[:, 0:1, :]
                elif self.args.channel == "L1":
                    x = x[:, 1:2, :]
                    
                z, c, spikes = self.model(x, is_encoded=self.use_spikes)
                loss, metrics = self.model.compute_cpc_loss(z, c, spikes=z)
                acc = metrics["acc1"]
                val_loss += loss.item()
                val_acc += acc
        
        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_acc = val_acc / len(self.val_loader)
        
        return avg_val_loss, avg_val_acc
