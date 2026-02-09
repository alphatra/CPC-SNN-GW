import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import json
import os
import numpy as np
import h5py
from src.data_handling.torch_dataset import HDF5SFTPairDataset

def _load_hard_negative_ids(path, max_count=0):
    with open(path, "r") as f:
        payload = json.load(f)

    items = []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        # Common wrappers.
        for key in ("hard_negatives", "items", "data"):
            if key in payload and isinstance(payload[key], list):
                items = payload[key]
                break

    hard_ids = []
    for rec in items:
        if isinstance(rec, (str, int)):
            hard_ids.append(str(rec))
        elif isinstance(rec, dict) and "id" in rec:
            hard_ids.append(str(rec["id"]))

    if max_count and max_count > 0:
        hard_ids = hard_ids[:max_count]

    # Preserve order while dropping duplicates.
    return list(dict.fromkeys(hard_ids))

def _build_hard_negative_sampler(train_indices, combined_indices, labels, args):
    if not args.hard_negatives_json:
        return None
    if args.hard_negative_boost <= 1.0:
        print("[HardNeg] hard_negative_boost <= 1.0, sampler disabled.")
        return None
    if not os.path.exists(args.hard_negatives_json):
        print(f"[HardNeg] File not found: {args.hard_negatives_json}. Sampler disabled.")
        return None

    hard_ids = _load_hard_negative_ids(args.hard_negatives_json, max_count=args.hard_negative_max)
    if len(hard_ids) == 0:
        print("[HardNeg] No valid IDs loaded from JSON. Sampler disabled.")
        return None
    hard_set = set(hard_ids)

    train_weights = np.ones(len(train_indices), dtype=np.float64)
    hard_in_train = 0
    noise_in_train = 0

    for local_idx, global_idx in enumerate(train_indices):
        lbl = int(labels[global_idx])
        if lbl != 0:
            continue
        noise_in_train += 1
        sid = str(combined_indices[global_idx])
        if sid in hard_set:
            train_weights[local_idx] = float(args.hard_negative_boost)
            hard_in_train += 1

    if hard_in_train == 0:
        print("[HardNeg] No hard-negative IDs overlapped with train split. Sampler disabled.")
        return None

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(train_weights, dtype=torch.double),
        num_samples=len(train_indices),
        replacement=True,
    )
    print(
        f"[HardNeg] Weighted sampler ON | loaded={len(hard_ids)} | "
        f"noise_in_train={noise_in_train} | hard_in_train={hard_in_train} | "
        f"boost={args.hard_negative_boost}"
    )
    return sampler

def setup_dataloaders(args, device):
    """
    Sets up train and validation dataloaders.
    
    Args:
        args: Parsed arguments containing paths and training configs.
        device: Torch device (used for pin_memory logic).
        
    Returns:
        train_loader, val_loader
    """
    # 1. Load Data (Mixed: Noise + Signal)
    with open(args.noise_indices, 'r') as f:
        noise_indices = json.load(f)
    with open(args.signal_indices, 'r') as f:
        signal_indices = json.load(f)
        
    print(f"Loaded {len(noise_indices)} noise samples and {len(signal_indices)} signal samples.")
    
    # 2. Subset / Mode Logic
    if args.fast:
        print("FAST MODE: Using subset of data.")
        noise_indices = noise_indices[:100]
        signal_indices = signal_indices[:100]
        # args.epochs = 1 # Handled in main script logic if passed by reference, or caller handles it
    
    if args.stability_test:
        print("STABILITY TEST MODE: 256 signal + 256 noise.")
        noise_indices = noise_indices[:256]
        signal_indices = signal_indices[:256]
    
    combined_indices = noise_indices + signal_indices
        
    # Check for pre-processed data (Spikes > TimeSeries > On-the-fly)
    spikes_path = args.h5_path.replace(".h5", "_spikes.h5")
    ts_path = args.h5_path.replace(".h5", "_timeseries.h5")
    
    print("-" * 40)
    print(f"[DEBUG] Spikes Path: {spikes_path} | Exists: {os.path.exists(spikes_path)}")
    print(f"[DEBUG] TimeSeries Path: {ts_path} | Exists: {os.path.exists(ts_path)}")
    print(f"[DEBUG] Force Reconstruct: {args.force_reconstruct} (Type: {type(args.force_reconstruct)})")
    
    # 3. Stratified Shuffle / Block Split
    n_noise = len(noise_indices)
    n_signal = len(signal_indices)
    
    # Sanity Check: Noise vs Noise
    if args.sanity_noise_only:
        print("[SANITY] Noise-Only Mode: Training on Noise vs Noise (expect AUC ~0.5)")
        # Use noise indices for both 'signal' positions and 'noise' positions
        # If we don't have enough noise, reuse/cycle them
        # Just concat noise_indices with itself if needed, or simple slice
        # combined_indices = noise_indices (first half) + noise_indices (second half)
        # To match expected length n_noise + n_signal
        
        # Simple strategy: Just grab more noise if available, or duplicate
        # For simplicity, we just use the loaded noise_indices for the "signal" part too
        # Need to ensure we don't crash if len(noise) < n_signal (unlikely for big data)
        signal_part_indices = (noise_indices * 2)[:n_signal] # rough fallback
        combined_indices = noise_indices + signal_part_indices
        labels = [0] * n_noise + [1] * n_signal # Labels still 0/1 to separate "classes"
    else:
        labels = [0] * n_noise + [1] * n_signal  # 0=noise, 1=signal

    if args.split_strategy == "random":
        # Shuffle with fixed seed for reproducibility (Standard)
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(combined_indices))
        combined_indices = [combined_indices[i] for i in perm]
        labels = [labels[i] for i in perm]
        print(f"Strategy: RANDOM Split. Shuffled {len(combined_indices)} samples.")
    else:
        # Time/Block Split: GLOBAL Sort by Index (Time Proxy)
        # 1. Zip indices and labels
        zipped = list(zip(combined_indices, labels))
        # 2. Sort by Index (ID string)
        zipped.sort(key=lambda x: x[0])
        # 3. Unzip
        combined_indices, labels = zip(*zipped)
        combined_indices = list(combined_indices)
        labels = list(labels)
        
        print(f"Strategy: GLOBAL TIME Split. Sorted {len(combined_indices)} samples by ID. Split will be strictly chronological.")

    
    # 4. Create Dataset
    forced_labels = labels if args.sanity_noise_only else None
    
    # If using use_sft, we MUST use HDF5SFTPairDataset to get 'f', 'mag', etc.
    # So we disable cache variables if use_sft is True.
    use_spikes = os.path.exists(spikes_path) and not args.force_reconstruct and not args.use_sft and not args.use_tf2d
    use_timeseries = os.path.exists(ts_path) and not args.force_reconstruct and not args.use_sft and not args.use_tf2d

    # Ensure indices/labels are consistent with selected source file.
    selected_h5 = spikes_path if use_spikes else (ts_path if use_timeseries else args.h5_path)
    with h5py.File(selected_h5, "r") as h5:
        available_ids = set(h5.keys())
    filtered_pairs = [
        (idx, lbl) for idx, lbl in zip(combined_indices, labels) if str(idx) in available_ids
    ]
    missing = len(combined_indices) - len(filtered_pairs)
    if missing > 0:
        print(f"Warning: {missing} samples missing in {selected_h5}; filtering them out before split.")
    if not filtered_pairs:
        raise RuntimeError(f"No overlapping IDs between provided indices and {selected_h5}")
    combined_indices, labels = zip(*filtered_pairs)
    combined_indices = list(combined_indices)
    labels = list(labels)
    if not args.sanity_noise_only:
        n_pos = int(sum(labels))
        if n_pos == 0 or n_pos == len(labels):
            raise RuntimeError(
                "Selected training source has only one class after ID filtering. "
                f"source={selected_h5}, positives={n_pos}, total={len(labels)}. "
                "Use --force_reconstruct True or rebuild *_timeseries/_spikes with both noise and signal indices."
            )
    
    if use_spikes:
        print(f"Using pre-encoded spikes from {spikes_path}")
        from src.data_handling.torch_dataset import HDF5TimeSeriesDataset
        dataset = HDF5TimeSeriesDataset(
            h5_path=spikes_path,
            index_list=combined_indices,
            forced_labels=forced_labels
        )
    elif use_timeseries:
        print(f"Using pre-processed time series data from {ts_path}")
        from src.data_handling.torch_dataset import HDF5TimeSeriesDataset
        dataset = HDF5TimeSeriesDataset(
            h5_path=ts_path,
            index_list=combined_indices,
            forced_labels=forced_labels
        )
    else:
        print("Using on-the-fly reconstruction (Slower)")
        dataset = HDF5SFTPairDataset(
            h5_path=args.h5_path,
            index_list=combined_indices,
            return_time_series=False,
            forced_labels=forced_labels
        )
    
    # 5. Split Logic
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    
    if args.split_strategy == "random":
        # Random Stratified Split
        from sklearn.model_selection import train_test_split
        train_indices, val_indices, train_labels, val_labels = train_test_split(
            indices, labels, 
            test_size=0.2, 
            stratify=labels, 
            random_state=42
        )
    else:
        # Time Split (Global Sorted Slice)
        split_idx = int(0.8 * dataset_size)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Labels for verification (optional)
        train_labels = labels[:split_idx]
        val_labels = labels[split_idx:] 

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)
    
    # Log class distribution
    print(f"Train: {sum(train_labels)}/{len(train_labels)} signal samples ({100*sum(train_labels)/len(train_labels):.1f}%)")
    print(f"Val: {sum(val_labels)}/{len(val_labels)} signal samples ({100*sum(val_labels)/len(val_labels):.1f}%)")

    train_sampler = _build_hard_negative_sampler(train_indices, combined_indices, labels, args)
    
    # 6. DataLoader Setup
    loader_kwargs = {
        "batch_size": args.batch_size,
        "pin_memory": True if str(device) != "mps" else False, 
    }
    
    if args.workers > 0:
        loader_kwargs["num_workers"] = args.workers
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
        # On macOS, spawn is safer
        if torch.backends.mps.is_available():
             multiprocessing_context = 'spawn'
        else:
             multiprocessing_context = None
    else:
        loader_kwargs["num_workers"] = 0
        multiprocessing_context = None

    if multiprocessing_context:
        train_loader = DataLoader(
            train_set,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            multiprocessing_context=multiprocessing_context,
            **loader_kwargs,
        )
        val_loader = DataLoader(val_set, shuffle=False, multiprocessing_context=multiprocessing_context, **loader_kwargs)
    else:
        train_loader = DataLoader(
            train_set,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            **loader_kwargs,
        )
        val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
        
    return train_loader, val_loader, use_spikes, use_timeseries
