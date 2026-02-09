import torch
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
from src.data_handling.torch_dataset import HDF5SFTPairDataset, HDF5TimeSeriesDataset

def compute_stats(x):
    """Compute basic stats for a tensor x (B, C, T) or (1, C, T)."""
    # Flatted over Batch and Time, keep Channels separate if we wanted, 
    # but user asks for general "x". Let's do per-channel stats if possible or aggregate.
    # Usually H1/L1 are similar physics, maybe aggregate for high level check.
    # Let's aggregate for now.
    
    x_flat = x.float().flatten()
    
    mean = x_flat.mean().item()
    std = x_flat.std().item()
    rms = torch.sqrt(torch.mean(x_flat**2)).item()
    
    # Percentiles
    # quantile requires float32/64
    p01 = (-x_flat).quantile(0.99).item() * -1 # p1 roughly
    # torch.quantile is slow on huge data, but ok for batches
    quantiles = torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99]).to(x.device)
    qs = torch.quantile(x_flat, quantiles)
    
    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "p01": qs[0].item(),
        "p05": qs[1].item(),
        "p50": qs[2].item(),
        "p95": qs[3].item(),
        "p99": qs[4].item(),
        "min": x_flat.min().item(),
        "max": x_flat.max().item()
    }

def main():
    parser = argparse.ArgumentParser()
    
    # Paths (using ../../ logic from train_cpc.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    default_h5 = os.path.join(project_root, "data/cpc_snn_train.h5")
    default_noise = os.path.join(project_root, "data/indices_noise.json")
    default_signal = os.path.join(project_root, "data/indices_signal.json")
    
    parser.add_argument("--h5_path", type=str, default=default_h5)
    parser.add_argument("--noise_indices", type=str, default=default_noise)
    parser.add_argument("--signal_indices", type=str, default=default_signal)
    parser.add_argument("--samples", type=int, default=100, help="Samples per class to check")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.h5_path}")
    print(f"Noise indices: {args.noise_indices}")
    print(f"Signal indices: {args.signal_indices}")
    
    with open(args.noise_indices, 'r') as f:
        noise_idx = json.load(f)[:args.samples]
    with open(args.signal_indices, 'r') as f:
        signal_idx = json.load(f)[:args.samples]
        
    # We use the on-the-fly reconstruction as that seems to be the default path in train_cpc.py 
    # when pre-processed files are not found (or logic is tricky).
    # Specifically, train_cpc.py uses HDF5SFTPairDataset(..., return_time_series=False).
    # But then batch_reconstruct_torch is called inside the loop.
    # To mimic training EXACTLY, we should load raw SFT and reconstruct.
    
    dataset_noise = HDF5SFTPairDataset(
        h5_path=args.h5_path,
        index_list=noise_idx,
        return_time_series=False
    )
    
    dataset_signal = HDF5SFTPairDataset(
        h5_path=args.h5_path,
        index_list=signal_idx,
        return_time_series=False
    )
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Helper to process a dataset
    def analyze_dataset(ds, name):
        print(f"\n--- Analyzing {name} ({len(ds)} samples) ---")
        
        # Accumulate stats? Or average of per-sample stats?
        # Let's accumulate all samples into one huge tensor for distribution analysis?
        # Might be heavy. Let's do running stats or just batch it.
        
        loader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=False)
        
        all_means = []
        all_stds = []
        all_rms = []
        
        # For quantiles, we might want to aggregate data
        # Let's take a few batches to estimate true distribution
        data_collector = []
        
        for batch in tqdm(loader):
            # Reconstruct exactly like train_cpc
            x = HDF5SFTPairDataset.batch_reconstruct_torch(batch, device=device)
            
            # Application of Instance Norm?
            # In train_cpc.py:
            # mean = x.mean(dim=2, keepdim=True)
            # std = x.std(dim=2, keepdim=True)
            # x = (x - mean) / (std + 1e-8)
            
            # The USER wants to know if input "carries signal" or is "dead".
            # Checking RAW x (before normalization) is crucial to see if it's all zeros.
            # Checking NORMALIZED x shows if normalization is working or killing it.
            
            # Let's check RAW first.
            x_raw = x.detach()
            
            # Collect for aggregation (careful with memory)
            if len(data_collector) < 1000: # Limit memory usage
                 data_collector.append(x_raw.cpu())
                 
        full_data = torch.cat(data_collector)
        stats = compute_stats(full_data)
        
        print(f"RAW Stats for {name}:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std : {stats['std']:.6f}")
        print(f"  RMS : {stats['rms']:.6f}")
        print(f"  Min/Max: {stats['min']:.6f} / {stats['max']:.6f}")
        print(f"  Percentiles: 1%={stats['p01']:.4f}, 50%={stats['p50']:.4f}, 99%={stats['p99']:.4f}")
        
        print(f"NORMALIZED Simulation (Instance Norm):")
        mean = full_data.mean(dim=2, keepdim=True)
        std = full_data.std(dim=2, keepdim=True)
        norm_data = (full_data - mean) / (std + 1e-8)
        
        stats_norm = compute_stats(norm_data)
        print(f"  Mean: {stats_norm['mean']:.6f}")
        print(f"  Std : {stats_norm['std']:.6f}")
        print(f"  RMS : {stats_norm['rms']:.6f}")
        
    analyze_dataset(dataset_noise, "NOISE")
    analyze_dataset(dataset_signal, "SIGNAL")

if __name__ == "__main__":
    main()
