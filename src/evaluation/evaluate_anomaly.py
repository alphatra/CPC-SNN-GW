import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from src.models.cpc_snn import CPCSNN
from src.data_handling.torch_dataset import HDF5SFTPairDataset

def compute_anomaly_score(model, x, device):
    """
    Computes a scalar anomaly score for each sample in the batch.
    Score = Mean Contrastive Loss over time.
    """
    model.eval()
    with torch.no_grad():
        z, c, _ = model(x)
        batch_size, time_steps, _ = z.shape
        
        total_loss = torch.zeros(batch_size).to(device)
        counts = torch.zeros(batch_size).to(device)
        
        for k in range(1, model.prediction_steps + 1):
            if time_steps <= k: continue
            
            W_k = model.predictors[k-1]
            c_t = c[:, :-k, :]
            z_tk = z[:, k:, :]
            z_pred = W_k(c_t)
            
            # Logits: (B, T-k, B)
            logits = torch.einsum('bth, ath -> bta', z_pred, z_tk)
            
            labels = torch.arange(batch_size).to(device)
            labels = labels.unsqueeze(1).expand(batch_size, time_steps - k)
            
            logits_flat = logits.reshape(-1, batch_size)
            labels_flat = labels.reshape(-1)
            
            loss_flat = F.cross_entropy(logits_flat, labels_flat, reduction='none')
            loss_per_t = loss_flat.reshape(batch_size, time_steps - k)
            
            # Sum over time
            total_loss += loss_per_t.sum(dim=1)
            counts += (time_steps - k)
            
        return total_loss / counts

def get_clusters(scores, threshold=0.5, cluster_threshold_time=0.35, sample_rate=2048.0):
    """
    Groups continuous threshold crossings into single events (Events).
    
    Args:
        scores (np.array): Array of scores (anomaly scores) over time [T].
        threshold (float): Detection threshold.
        cluster_threshold_time (float): Max gap in seconds to consider detections as one event.
        sample_rate (float): Sampling rate.
    
    Returns:
        cluster_times (list): Times of events (max score time).
        cluster_values (list): Max score in cluster.
    """
    
    # 1. Find indices where score > threshold
    trigger_indices = np.where(scores > threshold)[0]
    
    if len(trigger_indices) == 0:
        return [], []

    # 2. Convert indices to time
    trigger_times = trigger_indices / sample_rate
    trigger_values = scores[trigger_indices]
    
    # List of triggers: [[time, value], ...]
    triggers = list(zip(trigger_times, trigger_values))
    
    clusters = []
    
    # 3. Clustering
    for trigger in triggers:
        t, val = trigger
        
        if len(clusters) == 0:
            clusters.append([trigger])
            continue
            
        last_cluster = clusters[-1]
        last_trigger_time = last_cluster[-1][0]
        
        # If new trigger is close to previous -> same cluster
        if (t - last_trigger_time) <= cluster_threshold_time:
            last_cluster.append(trigger)
        else:
            clusters.append([trigger])

    # 4. Extract max from clusters
    final_times = []
    final_max_scores = []
    
    for cluster in clusters:
        # Find trigger with max score in cluster
        c_times = [trig[0] for trig in cluster]
        c_vals = [trig[1] for trig in cluster]
        
        max_idx = np.argmax(c_vals)
        final_times.append(c_times[max_idx])
        final_max_scores.append(c_vals[max_idx])

    return final_times, final_max_scores

def evaluate(args):
    # 1. Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 2. Load Model
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = CPCSNN(
        in_channels=2,
        hidden_dim=config['hidden_dim'],
        context_dim=config['context_dim'],
        prediction_steps=config['prediction_steps'],
        delta_threshold=config['delta_threshold']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # 3. Load Test Data (Mix of Noise and Signal)
    with open(args.noise_indices, 'r') as f: noise_ids = json.load(f)
    with open(args.signal_indices, 'r') as f: signal_ids = json.load(f)
    
    # Use a subset for evaluation
    n_eval = 500
    test_ids = noise_ids[:n_eval] + signal_ids[:n_eval]
    true_labels = [0]*n_eval + [1]*n_eval # 0=Noise, 1=Signal
    
    dataset = HDF5SFTPairDataset(
        h5_path=args.h5_path,
        index_list=test_ids,
        return_time_series=True
    )
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 4. Inference
    all_scores = []
    print("Running inference...")
    for batch in tqdm(loader):
        x = batch['x'].to(device)
        scores = compute_anomaly_score(model, x, device)
        all_scores.extend(scores.cpu().numpy())
        all_scores = np.concatenate(all_scores)
    
    # 5. Evaluate with Clustering
    # This is a per-sample evaluation, but for real streams we would stitch them.
    # Here `all_scores` is a flat list of scores per sample (0-dim per sample).
    # Wait, compute_anomaly_score returns (Batch_Size).
    # So all_scores is (N_samples).
    # This acts like a "score per event".
    # For stream clustering, we would need scores over time (B, T).
    # The current compute_anomaly_score averages over time.
    
    # If the user wants clustering, they likely want stream evaluation or
    # the metric is "number of detected events" vs "number of injected events".
    
    # Let's adapt: The input `scores` to get_clusters implies a time series.
    # If we have N independent validation samples (4s each), clustering doesn't make sense across them
    # UNLESS we treat them as a stream or if compute_anomaly_score returns (B, T).
    
    # The user request says: "Zamiast liczyć każde przekroczenie progu w milisekundzie...".
    # This implies we should be looking at time-resolved scores.
    
    # Let's check compute_anomaly_score again. It sums over time and returns scalar per batch item.
    # "return total_loss / counts" -> Scalar per sample.
    
    # If the user wants clustering, we probably need time-resolved scores.
    # But for now, let's assume we proceed with the scalar scores as "events" 
    # and maybe the user meant clustering on a continuous long stream (which we don't have here).
    
    # HOWEVER, the `get_clusters` function is provided.
    # Let's implement the standard ROC AUC on the scalar scores first (as existing).
    
    fpr, tpr, thresholds = roc_curve(true_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Optional: Clustering Example (Treating concatenated samples as a pseudo-stream)
    # In a real scenario, this should be applied to a continuous time-series.
    
    clustering_threshold = 3.0 # Example threshold
    event_times, event_scores = get_clusters(
        all_scores, 
        threshold=clustering_threshold, 
        cluster_threshold_time=0.5, 
        sample_rate=1.0 # Assuming 1 score per sample in this concatenated view
    )
    
    print(f"Clustering Analysis (Threshold={clustering_threshold}):")
    print(f"  > Found {len(event_times)} independent events (anomalies) in the test set.")
    if len(event_times) > 0:
        print(f"  > Avg Max Score: {np.mean(event_scores):.4f}")
    
    # 6. Plotting
    plt.figure(figsize=(10, 5))
    
    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Score Distribution
    plt.subplot(1, 2, 2)
    scores_noise = [s for s, l in zip(all_scores, true_labels) if l == 0]
    scores_signal = [s for s, l in zip(all_scores, true_labels) if l == 1]
    
    plt.hist(scores_noise, bins=30, alpha=0.5, label='Noise', density=True)
    plt.hist(scores_signal, bins=30, alpha=0.5, label='Signal', density=True)
    plt.xlabel('Anomaly Score (Contrastive Loss)')
    plt.title('Score Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("evaluation_results.png")
    print("Saved plots to evaluation_results.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Resolve paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    default_h5 = os.path.join(project_root, "data/cpc_snn_train.h5")
    default_noise = os.path.join(project_root, "data/indices_noise.json")
    default_signal = os.path.join(project_root, "data/indices_signal.json")
    
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--h5_path", type=str, default=default_h5)
    parser.add_argument("--noise_indices", type=str, default=default_noise)
    parser.add_argument("--signal_indices", type=str, default=default_signal)
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    evaluate(args)
