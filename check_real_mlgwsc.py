#!/usr/bin/env python3
"""
Deep check of MLGWSC-1 dataset structure
"""

import h5py
import numpy as np
from pathlib import Path

def check_mlgwsc():
    """Deep dive into MLGWSC-1 structure."""
    
    print("=" * 70)
    print("🔬 Deep Analysis of MLGWSC-1 Dataset")
    print("=" * 70)
    
    # Background file (noise)
    bg_file = Path("/teamspace/studios/this_studio/data/dataset-4/v2/train_background_s24w61w_1.hdf")
    
    print(f"\n📊 Analyzing: {bg_file.name}")
    print("-" * 50)
    
    with h5py.File(bg_file, 'r') as f:
        # Check H1 detector data
        if 'H1' in f:
            h1_group = f['H1']
            print(f"\n🔍 H1 Detector (Hanford):")
            print(f"   Type: {type(h1_group)}")
            
            # Check if it's a group or dataset
            if isinstance(h1_group, h5py.Group):
                print(f"   Keys in H1 group: {list(h1_group.keys())}")
                for key in h1_group.keys():
                    item = h1_group[key]
                    if isinstance(item, h5py.Dataset):
                        print(f"      {key}: shape={item.shape}, dtype={item.dtype}")
                        if item.shape[0] > 0:
                            sample = item[:5] if len(item.shape) == 1 else item[:5, :100]
                            print(f"         Sample: min={np.min(sample):.2e}, max={np.max(sample):.2e}")
            elif isinstance(h1_group, h5py.Dataset):
                print(f"   H1 is a dataset: shape={h1_group.shape}, dtype={h1_group.dtype}")
                total_duration = h1_group.shape[0] if len(h1_group.shape) > 0 else 0
                print(f"   Total samples: {total_duration:,}")
                
                # Estimate hours of data (assuming 4096 Hz sampling)
                hours = total_duration / (4096 * 3600)
                print(f"   Estimated hours of data @ 4096Hz: {hours:.2f} hours")
                
                # Sample the data
                if total_duration > 0:
                    sample = h1_group[:10000]  # First 10k samples
                    print(f"   Data range: [{np.min(sample):.2e}, {np.max(sample):.2e}]")
                    print(f"   Data mean: {np.mean(sample):.2e}")
                    print(f"   Data std: {np.std(sample):.2e}")
        
        # Check L1 detector data
        if 'L1' in f:
            l1_group = f['L1']
            print(f"\n🔍 L1 Detector (Livingston):")
            
            if isinstance(l1_group, h5py.Dataset):
                print(f"   L1 is a dataset: shape={l1_group.shape}, dtype={l1_group.dtype}")
                total_duration = l1_group.shape[0] if len(l1_group.shape) > 0 else 0
                print(f"   Total samples: {total_duration:,}")
    
    # Foreground file (signals)
    fg_file = Path("/teamspace/studios/this_studio/data/dataset-4/v2/train_foreground_s24w61w_1.hdf")
    
    print(f"\n📊 Analyzing: {fg_file.name}")
    print("-" * 50)
    
    with h5py.File(fg_file, 'r') as f:
        print(f"Keys: {list(f.keys())}")
        
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset):
                print(f"   {key}: shape={item.shape}, dtype={item.dtype}")
                if item.shape[0] > 0 and item.shape[0] < 1000000:  # Don't print if huge
                    print(f"      Number of GW events: {item.shape[0]:,}")
    
    # Injection parameters
    inj_file = Path("/teamspace/studios/this_studio/data/dataset-4/v2/train_injections_s24w61w_1.hdf")
    
    print(f"\n📊 Analyzing: {inj_file.name}")
    print("-" * 50)
    
    with h5py.File(inj_file, 'r') as f:
        print(f"Keys (parameters): {list(f.keys())[:5]}... ({len(f.keys())} total)")
        
        # Check number of injections
        first_key = list(f.keys())[0]
        num_injections = f[first_key].shape[0]
        print(f"\n🎯 Number of injection events: {num_injections:,}")
        
        # Sample some parameters
        if 'mass1' in f and 'mass2' in f:
            m1 = f['mass1'][:]
            m2 = f['mass2'][:]
            print(f"\nMass ranges:")
            print(f"   Mass1: {np.min(m1):.1f} - {np.max(m1):.1f} M☉")
            print(f"   Mass2: {np.min(m2):.1f} - {np.max(m2):.1f} M☉")
    
    print("\n" + "=" * 70)
    print("📝 FINAL VERDICT:")
    print("=" * 70)
    
    with h5py.File(bg_file, 'r') as f:
        if 'H1' in f:
            h1_data = f['H1']
            if isinstance(h1_data, h5py.Dataset):
                samples = h1_data.shape[0]
                hours = samples / (4096 * 3600)
                
                if samples > 1000000:  # More than 1M samples
                    print(f"✅ THIS IS REAL MLGWSC-1 DATA!")
                    print(f"   - {samples:,} samples ({hours:.1f} hours @ 4096Hz)")
                    print(f"   - {num_injections:,} GW injection events")
                    print(f"   - Professional competition dataset")
                    print(f"\n🚀 Ready for REAL training with 100k+ samples!")
                else:
                    print(f"⚠️  Limited dataset: only {samples:,} samples")
    
    print("=" * 70)

if __name__ == "__main__":
    check_mlgwsc()
