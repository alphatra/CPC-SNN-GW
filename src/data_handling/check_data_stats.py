import h5py
import numpy as np
import torch
import os
import sys

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
sys.path.append(project_root)

from src.data_handling.torch_dataset import HDF5SFTPairDataset

def check_stats():
    # Try to load timeseries h5
    ts_path = os.path.join(project_root, "data/cpc_snn_train_timeseries.h5")
    
    if os.path.exists(ts_path):
        print(f"Checking stats from {ts_path}...")
        with h5py.File(ts_path, 'r') as f:
            keys = list(f.keys())[:100] # Check first 100
            
            all_data = []
            for k in keys:
                H1 = f[f"{k}/H1"][()]
                L1 = f[f"{k}/L1"][()]
                all_data.append(H1)
                all_data.append(L1)
                
            all_data = np.concatenate(all_data)
            print(f"Mean: {np.mean(all_data)}")
            print(f"Std: {np.std(all_data)}")
            print(f"Min: {np.min(all_data)}")
            print(f"Max: {np.max(all_data)}")
            
    else:
        print("Timeseries HDF5 not found. Reconstructing from raw...")
        # Load raw
        h5_path = os.path.join(project_root, "data/cpc_snn_train.h5")
        import json
        indices_path = os.path.join(project_root, "data/indices_noise.json")
        with open(indices_path, 'r') as f:
            indices = json.load(f)[:20]
            
        dataset = HDF5SFTPairDataset(h5_path, indices, return_time_series=False)
        
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=20, shuffle=False)
        
        batch = next(iter(loader))
        
        # Reconstruct
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        x = HDF5SFTPairDataset.batch_reconstruct_torch(batch, device=device)
        x_np = x.cpu().numpy()
        
        print(f"Mean: {np.mean(x_np)}")
        print(f"Std: {np.std(x_np)}")
        print(f"Min: {np.min(x_np)}")
        print(f"Max: {np.max(x_np)}")

if __name__ == "__main__":
    check_stats()
