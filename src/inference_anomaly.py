import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.models.cpc_snn import CPCSNN
from src.data_handling.torch_dataset import HDF5SFTPairDataset

def compute_anomaly_score_trace(model, x, device):
    """
    Computes the anomaly score (loss) trace for a single sample or batch.
    Returns: (Batch, Time) score trace.
    """
    model.eval()
    with torch.no_grad():
        # Forward
        z, c, _ = model(x) # z: (B, T, H), c: (B, T, C)
        
        batch_size, time_steps, hidden_dim = z.shape
        
        # We want a score for each time step t.
        # Score[t] = Average Loss of predicting z_{t+k} from c_t across all k.
        # Or simpler: just accumulate loss contributions.
        
        # Initialize score array
        scores = torch.zeros(batch_size, time_steps).to(device)
        counts = torch.zeros(batch_size, time_steps).to(device)
        
        for k in range(1, model.prediction_steps + 1):
            if time_steps <= k: continue
            
            W_k = model.predictors[k-1]
            
            c_t = c[:, :-k, :]      # Context at t
            z_tk = z[:, k:, :]      # Target at t+k
            
            z_pred = W_k(c_t)       # Prediction for t+k
            
            # Compute contrastive loss for each step t
            # We need negatives. 
            # If Batch=1, we can't use batch negatives easily unless we use a memory bank.
            # But here we assume we process a batch of test samples (some noise, some signal).
            # So we can use the batch as negatives.
            
            # logits: (B, T-k, B)
            # z_pred: (B, T-k, H)
            # z_tk: (B, T-k, H) -> (A, T-k, H) for negatives
            
            logits = torch.einsum('bth, ath -> bta', z_pred, z_tk)
            
            # Targets: diagonal (b==a)
            labels = torch.arange(batch_size).to(device)
            labels = labels.unsqueeze(1).expand(batch_size, time_steps - k) # (B, T-k)
            
            # Flatten for CrossEntropy
            logits_flat = logits.reshape(-1, batch_size) # (B*(T-k), B)
            labels_flat = labels.reshape(-1)
            
            # Loss per element (no reduction)
            loss_flat = F.cross_entropy(logits_flat, labels_flat, reduction='none')
            
            # Reshape back to (B, T-k)
            loss_per_t = loss_flat.reshape(batch_size, time_steps - k)
            
            # Add to scores at time t (since c_t is responsible for prediction)
            # Or should we attribute it to t+k (the time of the anomaly)?
            # If a GW happens at t+k, c_t (from noise) won't predict it.
            # So the error will be high for the pair (t, t+k).
            # We can attribute it to t (predictive failure) or t+k (unexpected event).
            # Attributing to t+k seems more intuitive for "detection at time of event".
            # Let's attribute to t+k.
            
            # scores[:, k:] += loss_per_t
            # counts[:, k:] += 1
            
            # Actually, let's attribute to t (the context time) for now, 
            # as it measures "how confused is the model right now about the future".
            scores[:, :-k] += loss_per_t
            counts[:, :-k] += 1
            
        # Average
        counts[counts == 0] = 1
        scores /= counts
        
        return scores

def run_inference(
    h5_path="data/cpc_snn_train.h5",
    noise_indices_path="data/indices_noise.json",
    signal_indices_path="data/indices_signal.json",
    model_path="checkpoints/cpc_snn_noise_model.pth",
    n_samples=16
):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load Indices
    with open(noise_indices_path, 'r') as f:
        noise_indices = json.load(f)
    with open(signal_indices_path, 'r') as f:
        signal_indices = json.load(f)
        
    # Select a subset
    test_indices = noise_indices[:n_samples//2] + signal_indices[:n_samples//2]
    labels = [0]*(n_samples//2) + [1]*(n_samples//2) # 0=Noise, 1=Signal
    
    dataset = HDF5SFTPairDataset(
        h5_path=h5_path,
        index_list=test_indices,
        return_time_series=True
    )
    
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=False) # Single batch
    
    # Load Model
    model = CPCSNN(
        in_channels=2,
        hidden_dim=64,
        context_dim=64,
        prediction_steps=12,
        delta_threshold=0.5
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded.")
    
    # Inference
    batch = next(iter(loader))
    x = batch['x'].to(device)
    
    scores = compute_anomaly_score_trace(model, x, device)
    scores_np = scores.cpu().numpy()
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    # Plot a few Noise samples
    plt.subplot(2, 1, 1)
    for i in range(4): # First 4 are noise
        plt.plot(scores_np[i], label=f"Noise {i}", alpha=0.7)
    plt.title("Anomaly Scores - Noise Samples")
    plt.ylabel("Contrastive Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot a few Signal samples
    plt.subplot(2, 1, 2)
    for i in range(n_samples//2, n_samples//2 + 4): # First 4 signals
        plt.plot(scores_np[i], label=f"Signal {i}", alpha=0.7)
    plt.title("Anomaly Scores - Signal Samples")
    plt.ylabel("Contrastive Loss")
    plt.xlabel("Time Step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("anomaly_detection_results.png")
    print("Results saved to anomaly_detection_results.png")

if __name__ == "__main__":
    import argparse
    import os
    
    # Resolve paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../"))
    
    default_h5 = os.path.join(project_root, "data/cpc_snn_train.h5")
    default_noise = os.path.join(project_root, "data/indices_noise.json")
    default_signal = os.path.join(project_root, "data/indices_signal.json")
    default_model = os.path.join(project_root, "checkpoints/cpc_snn_noise_model.pth")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, default=default_h5)
    parser.add_argument("--noise_indices", type=str, default=default_noise)
    parser.add_argument("--signal_indices", type=str, default=default_signal)
    parser.add_argument("--model_path", type=str, default=default_model)
    parser.add_argument("--n_samples", type=int, default=16)
    
    args = parser.parse_args()
    
    run_inference(
        h5_path=args.h5_path,
        noise_indices_path=args.noise_indices,
        signal_indices_path=args.signal_indices,
        model_path=args.model_path,
        n_samples=args.n_samples
    )
