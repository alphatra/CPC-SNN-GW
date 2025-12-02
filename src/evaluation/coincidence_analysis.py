import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.cpc_snn import CPCSNN
from src.data_handling.torch_dataset import HDF5SFTPairDataset

def compute_anomaly_score_trace(model, x, device):
    """
    Computes anomaly score trace (loss over time).
    """
    model.eval()
    with torch.no_grad():
        z, c, _ = model(x)
        batch_size, time_steps, _ = z.shape
        
        scores = torch.zeros(batch_size, time_steps).to(device)
        counts = torch.zeros(batch_size, time_steps).to(device)
        
        for k in range(1, model.prediction_steps + 1):
            if time_steps <= k: continue
            
            W_k = model.predictors[k-1]
            c_t = c[:, :-k, :]
            z_tk = z[:, k:, :]
            z_pred = W_k(c_t)
            
            logits = torch.einsum('bth, ath -> bta', z_pred, z_tk)
            
            labels = torch.arange(batch_size).to(device)
            labels = labels.unsqueeze(1).expand(batch_size, time_steps - k)
            
            logits_flat = logits.reshape(-1, batch_size)
            labels_flat = labels.reshape(-1)
            
            loss_flat = F.cross_entropy(logits_flat, labels_flat, reduction='none')
            loss_per_t = loss_flat.reshape(batch_size, time_steps - k)
            
            scores[:, :-k] += loss_per_t
            counts[:, :-k] += 1
            
        counts[counts == 0] = 1
        scores /= counts
        return scores

def coincidence_analysis(args):
    # 1. Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 2. Load Models
    def load_model(path, in_channels=1):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = CPCSNN(
            in_channels=in_channels,
            hidden_dim=config['hidden_dim'],
            context_dim=config['context_dim'],
            prediction_steps=config['prediction_steps'],
            delta_threshold=config['delta_threshold']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    print(f"Loading H1 model from {args.model_h1}...")
    model_h1 = load_model(args.model_h1)
    
    print(f"Loading L1 model from {args.model_l1}...")
    model_l1 = load_model(args.model_l1)
    
    # 3. Load Test Data
    with open(args.signal_indices, 'r') as f: signal_ids = json.load(f)
    
    # Use a subset for demo
    test_ids = signal_ids[:50]
    
    dataset = HDF5SFTPairDataset(
        h5_path=args.h5_path,
        index_list=test_ids,
        return_time_series=True
    )
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 4. Inference & Coincidence
    print("Running coincidence analysis...")
    
    results = []
    
    for batch in tqdm(loader):
        x = batch['x'].to(device) # (B, 2, T)
        
        # Split channels
        x_h1 = x[:, 0:1, :]
        x_l1 = x[:, 1:2, :]
        
        # Get scores
        scores_h1 = compute_anomaly_score_trace(model_h1, x_h1, device)
        scores_l1 = compute_anomaly_score_trace(model_l1, x_l1, device)
        
        # Coincidence Logic
        # Simple thresholding + AND gate
        # In a real scenario, we'd look for peaks within 10ms window.
        # Here, we just multiply the scores (soft AND) or check if both exceed threshold.
        
        # Let's visualize the first sample in batch
        if len(results) == 0:
            # Save traces for plotting
            results.append({
                "h1": scores_h1[0].cpu().numpy(),
                "l1": scores_l1[0].cpu().numpy(),
                "strain_h1": x_h1[0, 0].cpu().numpy(),
                "strain_l1": x_l1[0, 0].cpu().numpy()
            })

    # 5. Plotting
    res = results[0]
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Strain
    ax[0].plot(res['strain_h1'], label='H1 Strain', color='orange', alpha=0.7)
    ax[0].plot(res['strain_l1'], label='L1 Strain', color='blue', alpha=0.7)
    ax[0].set_title("Strain Signals")
    ax[0].legend()
    
    # Individual Scores
    ax[1].plot(res['h1'], label='H1 Anomaly Score', color='orange')
    ax[1].plot(res['l1'], label='L1 Anomaly Score', color='blue')
    ax[1].set_title("Independent Anomaly Scores")
    ax[1].legend()
    
    # Coincidence Score (Product)
    coincidence_score = res['h1'] * res['l1']
    ax[2].plot(coincidence_score, label='Coincidence Score (H1 * L1)', color='green')
    ax[2].set_title("Coincidence Score")
    ax[2].legend()
    
    plt.tight_layout()
    plt.savefig("coincidence_results.png")
    print("Saved coincidence plot to coincidence_results.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Resolve paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    default_h5 = os.path.join(project_root, "data/cpc_snn_train.h5")
    default_signal = os.path.join(project_root, "data/indices_signal.json")
    
    parser.add_argument("--model_h1", type=str, required=True)
    parser.add_argument("--model_l1", type=str, required=True)
    parser.add_argument("--h5_path", type=str, default=default_h5)
    parser.add_argument("--signal_indices", type=str, default=default_signal)
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    coincidence_analysis(args)
