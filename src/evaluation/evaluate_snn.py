#!/usr/bin/env python
"""
SNN Binary Classification Evaluation

Evaluates trained SNN model on noise vs gravitational wave signal classification.
Uses logits/probabilities for ROC-AUC, not CPC-based anomaly scores.
"""
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
from src.evaluation.metrics import compute_tpr_at_fpr, compute_ece, compute_brier_score


def evaluate_snn(args):
    """
    Evaluate SNN binary classifier on test data.
    """
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
    config = checkpoint.get('config', {})
    
    model = CPCSNN(
        in_channels=config.get('in_channels', 1),
        hidden_dim=config.get('hidden_dim', 64),
        context_dim=config.get('context_dim', 64),
        prediction_steps=config.get('prediction_steps', 6),
        delta_threshold=config.get('delta_threshold', 0.1),
        temperature=config.get('temperature', 0.07),
        beta=config.get('beta', 0.85)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # 3. Load Test Data
    with open(args.noise_indices, 'r') as f:
        noise_ids = json.load(f)
    with open(args.signal_indices, 'r') as f:
        signal_ids = json.load(f)
    
    # Use subset for evaluation
    n_eval = args.n_samples
    test_noise = noise_ids[:n_eval]
    test_signal = signal_ids[:n_eval]
    
    test_ids = test_noise + test_signal
    true_labels = [0] * len(test_noise) + [1] * len(test_signal)
    
    dataset = HDF5SFTPairDataset(
        h5_path=args.h5_path,
        index_list=test_ids,
        return_time_series=True
    )
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 4. Inference
    all_probs = []
    all_labels = []
    all_logits = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch['x'].to(device)
            labels = batch['label']
            
            # Normalize
            mean = x.mean(dim=2, keepdim=True)
            std = x.std(dim=2, keepdim=True)
            x = (x - mean) / (std + 1e-8)
            
            # Forward pass
            logits, c, z = model(x)
            
            # Convert to probabilities
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_logits.extend(logits.cpu().numpy().flatten())
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    
    # 5. Compute Metrics
    from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, average_precision_score, confusion_matrix
    
    # Basic metrics
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    
    # Predictions at 0.5 threshold
    preds = (all_probs >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    
    # GW-specific: TPR at extremely low FPR
    tpr_metrics = compute_tpr_at_fpr(all_labels, all_probs, fpr_thresholds=[1e-3, 1e-4, 1e-5])
    
    # Calibration metrics
    ece = compute_ece(all_labels, all_probs)
    brier = compute_brier_score(all_labels, all_probs)
    
    # Confusion matrix at optimal threshold (FPR=1e-4)
    optimal_thresh = tpr_metrics[1e-4]['threshold']
    preds_optimal = (all_probs >= optimal_thresh).astype(int)
    cm = confusion_matrix(all_labels, preds_optimal)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"Evaluation Results")
    print(f"{'='*70}")
    print(f"Total samples: {len(all_labels)}")
    print(f"  Noise: {np.sum(all_labels == 0)}, Signal: {np.sum(all_labels == 1)}")
    print(f"\nBasic Metrics:")
    print(f"  Accuracy (thresh=0.5): {acc:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"\nCalibration:")
    print(f"  ECE: {ece:.4f} (lower is better)")
    print(f"  Brier Score: {brier:.4f} (lower is better)")
    print(f"\nGW Detection Performance (TPR at low FPR):")
    for fpr_val in sorted(tpr_metrics.keys()):
        metrics = tpr_metrics[fpr_val]
        print(f"  FPR={fpr_val:.0e}: TPR={metrics['tpr']:.4f} (threshold={metrics['threshold']:.4f})")
    print(f"\nConfusion Matrix at FPR=1e-4 (threshold={optimal_thresh:.4f}):")
    print(f"  TN={cm[0,0]:<6} FP={cm[0,1]:<6}")
    print(f"  FN={cm[1,0]:<6} TP={cm[1,1]:<6}")
    
    # Per-class statistics
    noise_probs = all_probs[all_labels == 0]
    signal_probs = all_probs[all_labels == 1]
    
    print(f"\nProbability Distributions:")
    print(f"  Noise: mean={noise_probs.mean():.3f}, std={noise_probs.std():.3f}")
    print(f"  Signal: mean={signal_probs.mean():.3f}, std={signal_probs.std():.3f}")
    
    # 6. Plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)
    
    # Probability Distribution
    axes[1].hist(noise_probs, bins=50, alpha=0.5, label='Noise', density=True, color='blue')
    axes[1].hist(signal_probs, bins=50, alpha=0.5, label='Signal', density=True, color='red')
    axes[1].axvline(0.5, color='black', linestyle='--', label='Threshold')
    axes[1].set_xlabel('Predicted Probability')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Probability Distributions')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Logits Distribution
    noise_logits = all_logits[all_labels == 0]
    signal_logits = all_logits[all_labels == 1]
    axes[2].hist(noise_logits, bins=50, alpha=0.5, label='Noise', density=True, color='blue')
    axes[2].hist(signal_logits, bins=50, alpha=0.5, label='Signal', density=True, color='red')
    axes[2].axvline(0, color='black', linestyle='--', label='Decision')
    axes[2].set_xlabel('Logits')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Logit Distributions')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    # Calibration plot
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(all_labels, all_probs, n_bins=15)
    axes[3].plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=6, label=f'Model (ECE={ece:.3f})')
    axes[3].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')
    axes[3].fill_between(prob_pred, prob_true, prob_pred, alpha=0.2)
    axes[3].set_xlabel('Predicted Probability', fontsize=11)
    axes[3].set_ylabel('True Probability', fontsize=11)
    axes[3].set_title('Calibration Curve', fontsize=12, fontweight='bold')
    axes[3].legend(loc='upper left')
    axes[3].grid(alpha=0.3, linestyle='--')
    axes[3].set_xlim([0, 1])
    axes[3].set_ylim([0, 1])
    
    plt.tight_layout()
    save_path = args.output or "snn_evaluation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plots to {save_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SNN Binary Classifier")
    
    # Resolve paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    default_h5 = os.path.join(project_root, "data/cpc_snn_train.h5")
    default_noise = os.path.join(project_root, "data/indices_noise.json")
    default_signal = os.path.join(project_root, "data/indices_signal.json")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--h5_path", type=str, default=default_h5,
                        help="Path to HDF5 data file")
    parser.add_argument("--noise_indices", type=str, default=default_noise,
                        help="Path to noise indices JSON")
    parser.add_argument("--signal_indices", type=str, default=default_signal,
                        help="Path to signal indices JSON")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of samples per class to evaluate")
    parser.add_argument("--output", type=str, default="snn_evaluation.png",
                        help="Output plot filename")
    
    args = parser.parse_args()
    evaluate_snn(args)
