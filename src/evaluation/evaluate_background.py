import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
from scipy import stats

from tqdm import tqdm
from scipy import stats

from src.train.data_setup import HDF5SFTPairDataset
from src.models.cpc_snn import CPCSNN
from src.models.tf_encoder import TFEncoder

def load_model(checkpoint_path, device):
    """Loads the model and arguments from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Reconstruct arguments from checkpoint if possible, or use defaults + overrides
    # Ideally, we should save args in checkpoint. Assuming 'args' dict or we just parse logic.
    # For now, we assume the user provides matching CLI args to instantiate the model structure.
    # We just load state_dict.
    
    # We DO need the exact args used for architecture (use_tf2d, etc).
    # If not in checkpoint, user MUST provide them.
    
    return checkpoint

def bootstrap_tpr_at_fpr(signal_scores, noise_scores, fpr_levels=[1e-4, 1e-3], n_boot=1000):
    """
    Computes Separability/TPR at fixed FPRs with Bootstrapping.
    Returns: Dict {fpr: {'tpr': mean, 'ci_low': v, 'ci_high': v}}
    """
    results = {}
    
    noise_scores = np.array(noise_scores)
    signal_scores = np.array(signal_scores)
    n_noise = len(noise_scores)
    
    for fpr in fpr_levels:
        # Check measurability
        if fpr < 1.0/n_noise:
            results[fpr] = {'tpr': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'status': 'Below Resolution'}
            continue
            
        tprs = []
        for _ in range(n_boot):
            # Resample Noise
            noise_sample = np.random.choice(noise_scores, n_noise, replace=True)
            # Resample Signal? Or just Noise threshold? Typically both for full confidence.
            # Usually threshold is fixed by Noise. Signal variance matters too.
            sig_sample = np.random.choice(signal_scores, len(signal_scores), replace=True)
            
            # Find Threshold
            # Sort Noise Descending
            noise_sorted = np.sort(noise_sample)[::-1]
            # Index for FPR
            idx = int(fpr * n_noise)
            threshold = noise_sorted[idx]
            
            # Calc TPR
            tp = (sig_sample > threshold).sum()
            tpr = tp / len(sig_sample)
            tprs.append(tpr)
            
        tprs = np.array(tprs)
        results[fpr] = {
            'tpr': np.mean(tprs),
            'ci_low': np.percentile(tprs, 2.5),
            'ci_high': np.percentile(tprs, 97.5),
            'threshold_mean': np.mean([np.sort(np.random.choice(noise_scores, n_noise, replace=True))[::-1][int(fpr*n_noise)] for _ in range(100)]) # Approx
        }
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best model checkpoint")
    parser.add_argument("--noise_h5", type=str, default="data/cpc_snn_train.h5", help="Path to H5 with Noise samples")
    parser.add_argument("--indices_noise", type=str, default="data/indices_noise.json")
    parser.add_argument("--indices_signal", type=str, default="data/indices_signal.json")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="mps")
    
    # Architecture args (must match training!)
    parser.add_argument("--use_tf2d", action="store_true", default=True)
    parser.add_argument("--channel", type=str, default=None) # H1 or L1 or None (Both)
    parser.add_argument("--no_mask", action="store_true", default=False, help="Use 6-channel model (trained without mask)")
    parser.add_argument("--ablate_mask", action="store_true", default=False, help="Zero out mask channel for leakage test (keeps 8-channel model)")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Model hidden dimension")
    parser.add_argument("--context_dim", type=int, default=64, help="Model context dimension")
    parser.add_argument("--prediction_steps", type=int, default=12, help="CPC Prediction steps")
    parser.add_argument("--top_k_noise", type=int, default=50, help="Number of Top Noise samples to save for inspection")
    
    # Advanced Eval Strategies
    parser.add_argument("--method", type=str, default="standard", choices=["standard", "swapped_pairs"], help="Eval method")
    parser.add_argument("--swaps", type=int, default=10, help="Number of swaps/shifts for swapped_pairs method")
    parser.add_argument("--ablate_ifo", type=str, default=None, choices=["H1", "L1"], help="Zero out specific IFO")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print(f"Loading Checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Init Model
    # Determine In_Channels
    comps = 3 if args.no_mask else 4
    in_channels = comps * (1 if args.channel else 2)
    
    model = CPCSNN(
        in_channels=in_channels,
        hidden_dim=args.hidden_dim, 
        context_dim=args.context_dim,
        use_tf2d=args.use_tf2d,
        prediction_steps=args.prediction_steps
    ).to(device)
    
    # Load Weights
    # Handle 'module.' prefix if saved from DataParallel, or strict matching
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Standard load failed, trying loose: {e}")
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()
    
    # Datasets
    print("Loading Data Indices...")
    with open(args.indices_noise, 'r') as f:
        all_noise = json.load(f)
    with open(args.indices_signal, 'r') as f:
        all_signal = json.load(f)
        
    print(f"Total Noise Pool: {len(all_noise)}")
    print(f"Total Signal Pool: {len(all_signal)}")
    
    # Create Full Datasets (No Split - Eval on All Available for Max Resolution)
    ds = HDF5SFTPairDataset(
        args.noise_h5,
        all_noise + all_signal, # Combine them but we will filter by label or track indices
        add_mask_channel=not args.no_mask,
        return_time_series=False
    )
    # Wait, simple list:
    ds_noise = HDF5SFTPairDataset(args.noise_h5, all_noise, add_mask_channel=not args.no_mask)
    ds_signal = HDF5SFTPairDataset(args.noise_h5, all_signal, add_mask_channel=not args.no_mask)
    
    loader_noise = DataLoader(ds_noise, batch_size=args.batch_size, num_workers=4)
    loader_signal = DataLoader(ds_signal, batch_size=args.batch_size, num_workers=4)
    
    # Inference Loop
    noise_probs = []
    noise_logits_list = [] # Store logits for EVT
    noise_ids = [] # Capture IDs for Top-K
    signal_probs = []
    signal_logits_list = []
    
    print("Evaluating Noise Background...")
    
    if args.method == "standard":
        with torch.no_grad():
            for batch in tqdm(loader_noise):
                # Prep Input
                feat_list = []
                slice_idx = 3 if args.no_mask else 4
                
                def get_feats(ifo):
                    d = batch[ifo].to(device)
                    if args.ablate_mask:
                         d[:, 3, :, :] = 0.0
                         return d[:, 0:4, :, :]
                    
                    # IFO Ablation
                    if args.ablate_ifo == ifo:
                        return torch.zeros_like(d[:, 0:slice_idx, :, :])
                    
                    return d[:, 0:slice_idx, :, :]
                
                if args.channel != "L1": feat_list.append(get_feats("H1"))
                if args.channel != "H1": feat_list.append(get_feats("L1"))
                
                x = torch.cat(feat_list, dim=1)
                mean = x.mean(dim=(2,3), keepdim=True)
                std = x.std(dim=(2,3), keepdim=True)
                x = (x - mean) / (std + 1e-8)
                
                logits, _, _ = model(x)
                logits_np = logits.detach().cpu().numpy().flatten()
                probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
                noise_probs.extend(probs)
                noise_logits_list.extend(logits_np)
                if 'id' in batch:
                    noise_ids.extend(batch['id'])
                else:
                    start_idx = len(noise_probs) - len(probs)
                    noise_ids.extend([f"BatchIdx_{i+start_idx}" for i in range(len(probs))])
                    
    elif args.method == "swapped_pairs":
        print(f"Generating Background via Time-Slides (Swapped Pairs). Swaps: {args.swaps}")
        # Load ALL Noise into RAM
        all_h1 = []
        all_l1 = []
        all_ids = []
        
        # Helper to slice correct channels
        def process_tensor(t):
             # t is (B, 4, T, F)
             slice_idx = 3 if args.no_mask else 4
             t = t[:, 0:slice_idx, :, :]
             if args.ablate_mask and slice_idx==4:
                  t[:, 3, :, :] = 0.0
             return t

        print("Loading Noise Bank into RAM...")
        with torch.no_grad():
             for batch in tqdm(loader_noise):
                 h1 = batch['H1'].to(device) # (B, 4, T, F)
                 l1 = batch['L1'].to(device)
                 
                 # Apply Channel Selection / Mask Ablation NOW to save RAM
                 h1 = process_tensor(h1).cpu()
                 l1 = process_tensor(l1).cpu()
                 
                 all_h1.append(h1)
                 all_l1.append(l1)
                 if 'id' in batch: all_ids.extend(batch['id'])
                 else: all_ids.extend(['?']*h1.shape[0])
                 
        full_h1 = torch.cat(all_h1, dim=0) # (N, C, T, F)
        full_l1 = torch.cat(all_l1, dim=0)
        N = full_h1.shape[0]
        print(f"Loaded {N} noise samples.")
        
        # Swapping Loop
        batch_size = args.batch_size
        
        with torch.no_grad():
            for k in range(args.swaps):
                # Roll L1 indices by LARGE RANDOM SHIFT (N/4 to N) to break local correlation
                # Avoid small shifts (e.g. 1..50) which preserve glitches
                min_shift = N // 4
                max_shift = N - 1
                if min_shift >= max_shift: min_shift = 1 # Fallback for small N
                
                shift = np.random.randint(min_shift, max_shift)
                l1_shifted = torch.roll(full_l1, shifts=shift, dims=0)
                # Convert ids list to array for rolling or just list slice
                # List roll:
                shifted_ids_list = all_ids[-shift:] + all_ids[:-shift]
                
                print(f"Swap {k+1}/{args.swaps} (Shift={shift})...")
                
                # Batch Processing
                for i in tqdm(range(0, N, batch_size)):
                    b_h1 = full_h1[i:i+batch_size].to(device)
                    b_l1 = l1_shifted[i:i+batch_size].to(device)
                    
                    # IFO Ablation (Applied late)
                    if args.ablate_ifo == "H1": b_h1.zero_()
                    if args.ablate_ifo == "L1": b_l1.zero_()
                    
                    # Stack H1, L1
                    feats = []
                    if args.channel != "L1": feats.append(b_h1)
                    if args.channel != "H1": feats.append(b_l1)
                    
                    x = torch.cat(feats, dim=1)
                    
                    # Norm
                    mean = x.mean(dim=(2,3), keepdim=True)
                    std = x.std(dim=(2,3), keepdim=True)
                    x = (x - mean) / (std + 1e-8)
                    
                    logits, _, _ = model(x)
                    l_np = logits.detach().cpu().numpy().flatten()
                    probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
                    
                    noise_probs.extend(probs)
                    noise_logits_list.extend(l_np)
                    
                    # Synthetic IDs
                    curr_ids_h1 = all_ids[i:i+batch_size]
                    curr_ids_l1 = shifted_ids_list[i:i+batch_size]
                    syn_ids = [f"{hid}_vs_{lid}" for hid, lid in zip(curr_ids_h1, curr_ids_l1)]
                    noise_ids.extend(syn_ids)
            
    print("Evaluating Signals...")
    with torch.no_grad():
         for batch in tqdm(loader_signal):
            feat_list = []
            slice_idx = 3 if args.no_mask else 4
            
            def get_feats(ifo):
                d = batch[ifo].to(device)
                if args.ablate_mask:
                     d[:, 3, :, :] = 0.0
                     return d[:, 0:4, :, :]
                if args.ablate_ifo == ifo:
                     return torch.zeros_like(d[:, 0:slice_idx, :, :])
                return d[:, 0:slice_idx, :, :]
            
            if args.channel != "L1": feat_list.append(get_feats("H1"))
            if args.channel != "H1": feat_list.append(get_feats("L1"))
            
            x = torch.cat(feat_list, dim=1)
            mean = x.mean(dim=(2,3), keepdim=True)
            std = x.std(dim=(2,3), keepdim=True)
            x = (x - mean) / (std + 1e-8)
            
            logits, _, _ = model(x)
            logits_np = logits.detach().cpu().numpy().flatten()
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            signal_probs.extend(probs)
            signal_logits_list.extend(logits_np)
            
    # Analysis
    noise_probs = np.array(noise_probs)
    signal_probs = np.array(signal_probs)
    noise_logits = np.array(noise_logits_list)
    signal_logits = np.array(signal_logits_list)

    print("\n--- Evaluation Results ---")
    print(f"Noise Samples: {len(noise_probs)}")
    print(f"Signal Samples: {len(signal_probs)}")
    print(f"Noise Max Score: {noise_probs.max():.6f}")
    print(f"Noise Mean: {noise_probs.mean():.6f}")
    print(f"Signal Mean: {signal_probs.mean():.6f}")
    
    # --- Top-K Noise Analysis (Hard Negative Mining) ---
    # Combine ID and Score
    if len(noise_ids) == len(noise_probs):
        noise_pairs = list(zip(noise_ids, noise_probs))
        # Sort by Score Descending
        noise_pairs.sort(key=lambda x: x[1], reverse=True)
        
        top_k = args.top_k_noise
        print(f"\n### Top {top_k} Hard Negatives (Potential Leak/Glitches) ###")
        top_k_data = []
        for i in range(min(top_k, len(noise_pairs))):
            nid, score = noise_pairs[i]
            top_k_data.append({'id': nid, 'score': float(score)})
            if i < 10: # Print Top 10
                 print(f"Rank {i+1}: ID={nid} Score={score:.6f}")
                 
        with open("top_k_noise.json", "w") as f:
            json.dump(top_k_data, f, indent=4)
        print(f"Saved Top-{top_k} Noise IDs to top_k_noise.json")
    else:
        print("\n[Warning] ID mismatch, skipping Top-K ID report.")
    
    # Bootstrapping
    fprs = [1e-2, 1e-3, 1e-4]
    if len(noise_probs) > 10000: fprs.append(1e-5)
    if len(noise_probs) > 100000: fprs.append(1e-6)
    
    # Bootstrapping (Empirical)
    fprs = [1e-2, 1e-3, 1e-4]
    if len(noise_probs) > 10000: fprs.append(1e-5)
    if len(noise_probs) > 100000: fprs.append(1e-6)
    
    metrics = bootstrap_tpr_at_fpr(signal_probs, noise_probs, fpr_levels=fprs)
    
    print("\n### TPR @ FPR (Bootstrap 95% CI)")
    for fpr, res in metrics.items():
        if 'status' in res:
             print(f"FPR={fpr:.0e}: {res['status']}")
        else:
             print(f"FPR={fpr:.0e}: {res['tpr']:.4f} [{res['ci_low']:.4f} - {res['ci_high']:.4f}] (Thr: {res['threshold_mean']:.4f})")

    # --- Extreme Value Theory (EVT) Extrapolation on LOGITS ---
    # Fit Generalized Pareto to the tail (top 5%) of LOGITS
    # Only if we have enough samples (>1000)
    evt_metrics = {}
    if len(noise_logits) > 1000:
        try:
            from scipy.stats import genpareto
            sorted_logits = np.sort(noise_logits) # Sort Logits
            # Threshold u for tail: 95th percentile
            u = sorted_logits[int(0.95 * len(sorted_logits))]
            tail_scores = sorted_logits[sorted_logits > u] - u # Shift to 0
            
            # Fit GPD: shape (c/xi), loc=0, scale (sigma)
            c, loc, scale = genpareto.fit(tail_scores, floc=0)
            
            print(f"\n### EVT Tail Extrapolation (Logits Top 5% Fit, u={u:.4f}) ###")
            print(f"GPD Params: shape(xi)={c:.4f}, scale={scale:.4f}")
            
            n_total = len(noise_logits)
            n_tail = len(tail_scores)
            
            target_fprs = [1e-4, 1e-5, 1e-6, 1e-8]
            evt_results = {}
            
            def sigmoid(x): return 1 / (1 + np.exp(-x))
            
            for fpr in target_fprs:
                if c != 0:
                    term = ((n_total * fpr) / n_tail) ** (-c) - 1
                    th_logit = u + (scale / c) * term
                else:
                    # Exponential limit (xi -> 0)
                    th_logit = u - scale * np.log((n_total * fpr) / n_tail)
                
                # Convert Threshold to Probability
                th_prob = sigmoid(th_logit)
                
                # Check TPR at this EVT threshold (using probabilities)
                tpr_evt = (signal_probs > th_prob).mean()
                evt_results[fpr] = {'thr_logit': float(th_logit), 'thr_prob': float(th_prob), 'tpr': float(tpr_evt)}
                print(f"EVT FPR={fpr:.0e} -> Thr(Logit)={th_logit:.4f} | Thr(Prob)={th_prob:.6f} | TPR={tpr_evt:.4f}")
            
            evt_metrics = evt_results
            
            # Sanity Check warning
            if c > 0.5:
                print("[Warning] GPD shape xi > 0.5 implies very heavy tail. Extrapolation may be unstable.")
                
        except Exception as e:
            print(f"EVT fitting failed: {e}")

    # Save Report
    # Convert metrics to float for JSON
    json_metrics = {}
    for k, v in metrics.items():
        if 'status' in v:
            json_metrics[str(k)] = v
        else:
            json_metrics[str(k)] = {nk: float(nv) for nk, nv in v.items()}
            
    # Add EVT to report
    json_evt = {str(k): v for k, v in evt_metrics.items()}

    report = {
        'noise_stats': {'max': float(noise_probs.max()), 'mean': float(noise_probs.mean())},
        'signal_stats': {'mean': float(signal_probs.mean())},
        'metrics': json_metrics,
        'evt_extrapolation': json_evt
    }
    
    with open("large_scale_eval_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print("\nReport saved to large_scale_eval_report.json")

if __name__ == "__main__":
    main()
