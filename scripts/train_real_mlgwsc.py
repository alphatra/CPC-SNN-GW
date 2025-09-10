#!/usr/bin/env python3
"""
REAL training script with MLGWSC-1 dataset
This uses the actual 583k+ GW injection events
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_real_mlgwsc_data(num_samples=10000):
    """Load REAL MLGWSC-1 data with proper structure."""
    
    logger.info("=" * 70)
    logger.info("🚀 Loading REAL MLGWSC-1 Dataset")
    logger.info("=" * 70)
    
    data_dir = Path("/teamspace/studios/this_studio/data/dataset-4/v2")
    
    # Load background noise
    bg_file = data_dir / "train_background_s24w61w_1.hdf"
    fg_file = data_dir / "train_foreground_s24w61w_1.hdf"
    inj_file = data_dir / "train_injections_s24w61w_1.hdf"
    
    all_samples = []
    all_labels = []
    
    logger.info("\n📊 Loading background noise...")
    with h5py.File(bg_file, 'r') as f:
        # Get H1 detector data
        h1_group = f['H1']
        for gps_time in list(h1_group.keys())[:1]:  # Use first GPS segment
            h1_data = h1_group[gps_time][:]
            logger.info(f"   H1 segment {gps_time}: {len(h1_data):,} samples")
            
            # Create sliding windows (1 second @ 4096 Hz)
            window_size = 4096
            stride = 2048  # 50% overlap
            
            for i in range(0, len(h1_data) - window_size, stride):
                window = h1_data[i:i+window_size]
                all_samples.append(window)
                all_labels.append(0)  # Noise = 0
                
                if len(all_samples) >= num_samples // 2:
                    break
    
    noise_count = len(all_samples)
    logger.info(f"   Loaded {noise_count:,} noise windows")
    
    logger.info("\n📊 Loading GW signals...")
    with h5py.File(fg_file, 'r') as f:
        if 'H1' in f:
            h1_fg = f['H1']
            
            # Foreground contains multiple GW events
            if isinstance(h1_fg, h5py.Group):
                event_keys = list(h1_fg.keys())[:num_samples // 2]
                
                for event_key in event_keys:
                    event_data = h1_fg[event_key][:]
                    
                    # Take central part of event
                    if len(event_data) >= 4096:
                        center = len(event_data) // 2
                        window = event_data[center-2048:center+2048]
                        all_samples.append(window)
                        all_labels.append(1)  # GW signal = 1
            elif isinstance(h1_fg, h5py.Dataset):
                # Alternative format - continuous data with injections
                fg_data = h1_fg[:]
                
                # Load injection times
                with h5py.File(inj_file, 'r') as inj_f:
                    if 'tc' in inj_f:  # Coalescence times
                        tc_times = inj_f['tc'][:num_samples // 2]
                        
                        for tc in tc_times:
                            # Extract window around coalescence
                            idx = int(tc * 4096) % len(fg_data)  # Convert to sample index
                            
                            if idx > 2048 and idx < len(fg_data) - 2048:
                                window = fg_data[idx-2048:idx+2048]
                                all_samples.append(window)
                                all_labels.append(1)
    
    signal_count = len(all_samples) - noise_count
    logger.info(f"   Loaded {signal_count:,} GW signal windows")
    
    # Convert to JAX arrays
    X = jnp.array(all_samples)
    y = jnp.array(all_labels)
    
    # Shuffle
    key = jax.random.PRNGKey(42)
    perm = jax.random.permutation(key, len(X))
    X = X[perm]
    y = y[perm]
    
    # Split train/test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info("\n" + "=" * 70)
    logger.info("📊 REAL MLGWSC-1 Dataset Loaded:")
    logger.info(f"   Train: {len(X_train):,} samples")
    logger.info(f"   Test: {len(X_test):,} samples")
    logger.info(f"   Class balance: {np.mean(y_train):.1%} positive")
    logger.info(f"   Sample shape: {X_train[0].shape}")
    logger.info("=" * 70)
    
    return (X_train, y_train), (X_test, y_test)

def main():
    """Run REAL training with proper MLGWSC-1 data."""
    
    # Load REAL data
    (X_train, y_train), (X_test, y_test) = load_real_mlgwsc_data(num_samples=50000)
    
    logger.info("\n🚀 Starting REAL training with MLGWSC-1...")
    logger.info(f"   This is NOT mock data!")
    logger.info(f"   Real samples: {len(X_train):,}")
    
    # Import and run unified trainer
    from training.unified_trainer import UnifiedTrainer
    
    # Create simple config dict
    config = {
        'seed': 42,
        'epochs': 60,
        'batch_size': 32,
        'learning_rate': 5e-4,
        'device': 'gpu',
        'name': 'mlgwsc_real_training'
    }
    
    # Convert to namespace for compatibility
    from types import SimpleNamespace
    config = SimpleNamespace(**config)
    
    trainer = UnifiedTrainer(config)
    
    # Train on REAL data
    logger.info("\n🔥 Training on REAL MLGWSC-1 data...")
    results = trainer.train(X_train, y_train, X_test, y_test)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ REAL Training Complete!")
    logger.info(f"   Final Test Accuracy: {results.get('test_accuracy', 0):.1%}")
    logger.info(f"   Final ROC-AUC: {results.get('roc_auc', 0):.3f}")
    logger.info("=" * 70)
    
    return results

if __name__ == "__main__":
    main()
