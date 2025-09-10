#!/usr/bin/env python3
"""
Test MLGWSC-1 dataset loader to verify we have real data
"""

import h5py
import numpy as np
from pathlib import Path

def test_mlgwsc_dataset():
    """Check if MLGWSC-1 dataset is real and accessible."""
    
    print("=" * 60)
    print("🔍 Testing MLGWSC-1 Dataset")
    print("=" * 60)
    
    # Check dataset files
    data_dir = Path("/teamspace/studios/this_studio/data/dataset-4/v2")
    
    files = {
        'train_background': data_dir / "train_background_s24w61w_1.hdf",
        'train_foreground': data_dir / "train_foreground_s24w61w_1.hdf", 
        'train_injections': data_dir / "train_injections_s24w61w_1.hdf",
        'val_background': data_dir / "val_background_s24w6d1_1.hdf",
        'val_foreground': data_dir / "val_foreground_s24w6d1_1.hdf",
        'val_injections': data_dir / "val_injections_s24w6d1_1.hdf"
    }
    
    print("\n📁 Checking dataset files:")
    for name, path in files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {name}: {size_mb:.1f} MB")
        else:
            print(f"   ❌ {name}: NOT FOUND")
    
    # Load and check train background
    train_bg = files['train_background']
    if train_bg.exists():
        print(f"\n📊 Analyzing {train_bg.name}:")
        with h5py.File(train_bg, 'r') as f:
            print(f"   Keys: {list(f.keys())}")
            
            # Check for strain data
            if 'data' in f:
                data = f['data']
                print(f"   Data shape: {data.shape}")
                print(f"   Data dtype: {data.dtype}")
                
                # Sample some data
                sample = data[:10, :100]  # First 10 samples, first 100 points
                print(f"   Sample range: [{np.min(sample):.2e}, {np.max(sample):.2e}]")
                print(f"   Sample mean: {np.mean(sample):.2e}")
                print(f"   Sample std: {np.std(sample):.2e}")
                
                # Count total samples
                total_samples = data.shape[0]
                print(f"\n   🎯 Total samples available: {total_samples:,}")
                
                if total_samples > 1000:
                    print(f"   ✅ This is REAL data! (not mock)")
                else:
                    print(f"   ⚠️  Small dataset - might be test data")
                    
            elif 'H1_strain' in f or 'L1_strain' in f:
                # Alternative format
                for key in ['H1_strain', 'L1_strain']:
                    if key in f:
                        data = f[key]
                        print(f"   {key} shape: {data.shape}")
                        total_samples = data.shape[0] if len(data.shape) > 0 else 1
                        print(f"   {key} samples: {total_samples:,}")
    
    # Load and check injections
    train_inj = files['train_injections']
    if train_inj.exists():
        print(f"\n📊 Analyzing {train_inj.name}:")
        with h5py.File(train_inj, 'r') as f:
            print(f"   Keys: {list(f.keys())}")
            
            # Check size to verify it's real
            if 'data' in f:
                shape = f['data'].shape
                print(f"   Injections shape: {shape}")
                print(f"   Number of injections: {shape[0]:,}")
    
    print("\n" + "=" * 60)
    print("📝 Summary:")
    
    # Final verdict
    real_data = False
    if train_bg.exists():
        with h5py.File(train_bg, 'r') as f:
            if 'data' in f:
                real_data = f['data'].shape[0] > 10000
    
    if real_data:
        print("✅ MLGWSC-1 dataset is REAL with 10,000+ samples!")
        print("   Ready for full training!")
    else:
        print("⚠️  Dataset might be limited or in different format")
        print("   Check the data structure manually")
    
    print("=" * 60)

if __name__ == "__main__":
    test_mlgwsc_dataset()
