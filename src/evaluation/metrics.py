"""
Advanced Metrics for Gravitational Wave Detection and SNN Monitoring

Provides production-grade metrics for:
- GW detection quality at extremely low FPR
- Model calibration for reliable confidence scores  
- SNN health monitoring (spike statistics)
"""

import numpy as np
import torch
from typing import Dict, List, Union, Tuple



def compute_tpr_at_fpr(
    labels: np.ndarray, 
    scores: np.ndarray, 
    fpr_thresholds: List[float] = [1e-3, 1e-4, 1e-5, 1e-6]
) -> Dict[float, Dict[str, float]]:
    """
    Compute TPR at extremely low FPR (critical for GW detection).
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    results = {}
    for target_fpr in fpr_thresholds:
        # Find first point where FPR >= target
        idx = np.where(fpr >= target_fpr)[0]
        if len(idx) == 0:
            idx = len(fpr) - 1
        else:
            idx = idx[0]
        
        results[target_fpr] = {
            'tpr': float(tpr[idx]),
            'fnr': 1.0 - float(tpr[idx]),
            'threshold': float(thresholds[idx]),
            'actual_fpr': float(fpr[idx])
        }
    
    return results


def compute_stability_metrics(
    labels: np.ndarray,
    scores: np.ndarray
) -> Dict[str, float]:
    """
    Compute stability metrics (noise tails, d-prime, KS).
    """
    noise_scores = scores[labels == 0]
    signal_scores = scores[labels == 1]
    
    metrics = {}
    
    if len(noise_scores) > 0:
        metrics['noise_max'] = float(np.max(noise_scores))
        metrics['noise_q99.9'] = float(np.percentile(noise_scores, 99.9))
        metrics['noise_q99.99'] = float(np.percentile(noise_scores, 99.99))
    else:
        metrics['noise_max'] = 0.0
        metrics['noise_q99.9'] = 0.0
        metrics['noise_q99.99'] = 0.0
        
    # d-prime (separation)
    if len(noise_scores) > 1 and len(signal_scores) > 1:
        mu0 = np.mean(noise_scores)
        mu1 = np.mean(signal_scores)
        var0 = np.var(noise_scores, ddof=1)
        var1 = np.var(signal_scores, ddof=1)
        # pooled std dev
        pooled_std = np.sqrt(0.5 * (var0 + var1))
        metrics['d_prime'] = float((mu1 - mu0) / (pooled_std + 1e-10))
    else:
        metrics['d_prime'] = 0.0
        
    # KS Statistic
    try:
        from scipy.stats import ks_2samp
        ks_stat, _ = ks_2samp(noise_scores, signal_scores)
        metrics['ks_stat'] = float(ks_stat)
    except ImportError:
        metrics['ks_stat'] = 0.0
        
    return metrics



def compute_ece(
    labels: np.ndarray, 
    probs: np.ndarray, 
    n_bins: int = 15
) -> float:
    """
    Expected Calibration Error - measures confidence calibration.
    
    Critical for setting detection thresholds in real deployments.
    A calibrated model's predicted probability P(signal|x)=0.8 should
    actually contain signals 80% of the time.
    
    Args:
        labels: Ground truth binary labels
        probs: Predicted probabilities (must be in [0, 1])
        n_bins: Number of bins for binning predictions
    
    Returns:
        ECE value (lower is better, 0 = perfect calibration)
        
    Reference:
        Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Samples in this bin
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def compute_brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    """
    Brier score - MSE between predicted probabilities and true labels.
    
    Combines calibration and refinement. Lower is better.
    
    Args:
        labels: Ground truth binary labels
        probs: Predicted probabilities
    
    Returns:
        Brier score (0 = perfect, 1 = worst)
    """
    return float(np.mean((probs - labels) ** 2))


def compute_snn_stats(spike_tensor: torch.Tensor, eps: float = 1e-8) -> Dict[str, Union[float, np.ndarray]]:
    """
    Comprehensive SNN spike statistics for monitoring network health.
    
    Detects common SNN pathologies:
    - Dead neurons (never spike)
    - Saturated neurons (always spike)  
    - Unbalanced temporal activity
    
    Args:
        spike_tensor: Binary spike tensor, shape (B, C, T) or (B, T, C)
                     where B=batch, C=channels, T=time
    
    Returns:
        Dict with:
        - firing_rate_mean/std/min/max: Statistics of spike rates per channel
        - dead_neurons_ratio: Fraction of channels with zero spikes
        - saturation_ratio: Fraction of channels spiking >90% of time
        - temporal_sparsity: Spike rate over time (T,) for visualization
    """
    if spike_tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got shape {spike_tensor.shape}")
    
    # Ensure (B, C, T) format
    B, dim1, dim2 = spike_tensor.shape
    if dim1 > dim2:  # Heuristic: time (T) usually > channels (C)
        spike_tensor = spike_tensor.permute(0, 2, 1)
        B, C, T = spike_tensor.shape
    else:
        C, T = dim1, dim2
    
    # Firing rate per channel (averaged over batch and time)
    firing_rate = spike_tensor.float().mean(dim=(0, 2))  # (C,)
    
    # Dead neurons: channels with zero total spikes
    total_spikes_per_channel = spike_tensor.sum(dim=(0, 2))  # (C,)
    dead_ratio = float((total_spikes_per_channel == 0).float().mean())
    
    # Saturated neurons: channels that spike >90% of time
    spike_rate_per_channel = spike_tensor.float().mean(dim=2)  # (B, C)
    saturated = (spike_rate_per_channel > 0.9).any(dim=0)  # (C,)
    saturation_ratio = float(saturated.float().mean())
    
    # Temporal profile: spike activity over time
    temporal_profile = spike_tensor.float().mean(dim=(0, 1))  # (T,)
    
    return {
        'firing_rate_mean': float(firing_rate.mean()),
        'firing_rate_std': float(firing_rate.std()),
        'firing_rate_min': float(firing_rate.min()),
        'firing_rate_max': float(firing_rate.max()),
        'dead_neurons_ratio': dead_ratio,
        'saturation_ratio': saturation_ratio,
        'temporal_sparsity': temporal_profile.cpu().numpy(),
        'num_channels': C,
        'num_timesteps': T
    }


def compute_per_layer_stats(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 10
) -> Dict[str, Dict]:
    """
    Compute SNN statistics for each layer by running inference.
    
    Args:
        model: SNN model with SpikingCNN encoder
        data_loader: DataLoader for computing stats
        device: Device to run on
        max_batches: Number of batches to average over
    
    Returns:
        Dict mapping layer names to their spike statistics
    """
    model.eval()
    
    # Storage for spikes from each layer
    layer_spikes = {
        'lif1': [],
        'lif2': [],
        'lif3': []
    }
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= max_batches:
                break
            
            x = batch['x'].to(device)
            
            # Enable spike collection (would need hooks in actual implementation)
            # This is placeholder - real implementation needs forward hooks
            # For now, return empty dict
            pass
    
    # Aggregate and compute stats
    results = {}
    for layer_name, spike_list in layer_spikes.items():
        if spike_list:
            all_spikes = torch.cat(spike_list, dim=0)
            results[layer_name] = compute_snn_stats(all_spikes)
    
    return results
