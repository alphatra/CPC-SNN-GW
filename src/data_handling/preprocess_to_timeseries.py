import h5py
import torch
import numpy as np
import argparse
import os
import sys
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
sys.path.append(project_root)

from src.data_handling.torch_dataset import HDF5SFTPairDataset

def preprocess(args):
    # 1. Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 2. Load Indices
    with open(args.indices_path, 'r') as f:
        indices = json.load(f)
        
    if args.fast:
        indices = indices[:100]
        
    # 3. Dataset
    dataset = HDF5SFTPairDataset(
        h5_path=args.h5_path,
        index_list=indices,
        return_time_series=False
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0 
    )
    
    # Encoder if needed
    encoder = None
    if args.encode_spikes:
        from src.models.encoders import DeltaModulationEncoder
        encoder = DeltaModulationEncoder(threshold=args.threshold).to(device)
        print(f"Encoding spikes with threshold={args.threshold}...")
    
    # 4. Create Output HDF5
    mode = "w" if args.overwrite else "a"
    print(f"Writing to {args.output_path} (mode='{mode}')...")
    
    with h5py.File(args.output_path, mode) as out_h5:
        
        for i, batch in enumerate(tqdm(loader)):
            # Reconstruct on GPU
            x_rec = HDF5SFTPairDataset.batch_reconstruct_torch(batch, device=device)
            
            # Normalize (Instance Normalization per channel)
            # x_rec: (B, C, T)
            # Mean/Std over Time dimension
            mean = x_rec.mean(dim=2, keepdim=True)
            std = x_rec.std(dim=2, keepdim=True)
            x_rec = (x_rec - mean) / (std + 1e-8)
            
            if encoder is not None:
                # Encode to spikes
                # x_rec: (B, 2, T)
                with torch.no_grad():
                    spikes = encoder(x_rec) # (B, 2, T) {-1, 0, 1}
                
                # Move to CPU and cast to int8
                data_np = spikes.cpu().to(torch.int8).numpy()
            else:
                data_np = x_rec.cpu().numpy()
            
            # Save each sample
            current_batch_size = data_np.shape[0]
            global_idx_start = i * args.batch_size
            
            for b in range(current_batch_size):
                idx_str = str(indices[global_idx_start + b])
                
                g = out_h5.create_group(idx_str)
                
                g.create_dataset("H1", data=data_np[b, 0])
                g.create_dataset("L1", data=data_np[b, 1])
                
                if "label" in batch:
                    g.create_dataset("label", data=batch["label"][b].item())

    print("Pre-processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    default_h5 = os.path.join(project_root, "data/cpc_snn_train.h5")
    default_indices = os.path.join(project_root, "data/indices_noise.json")
    default_output = os.path.join(project_root, "data/cpc_snn_train_timeseries.h5")
    
    parser.add_argument("--h5_path", type=str, default=default_h5)
    parser.add_argument("--indices_path", type=str, default=default_indices)
    parser.add_argument("--output_path", type=str, default=default_output)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--encode_spikes", action="store_true", help="Pre-encode to spikes")
    parser.add_argument("--threshold", type=float, default=0.05, help="Delta threshold")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing file")
    
    args = parser.parse_args()
    
    if args.encode_spikes and "_timeseries" in args.output_path:
        args.output_path = args.output_path.replace("_timeseries.h5", "_spikes.h5")
        
    preprocess(args)
