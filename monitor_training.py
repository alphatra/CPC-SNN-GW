#!/usr/bin/env python3
"""
Monitor training progress
"""

import os
import time
from pathlib import Path
import sys

def monitor():
    """Monitor training outputs."""
    output_dir = Path("outputs/fixed_training")
    
    print("=" * 60)
    print("📊 Training Monitor")
    print("=" * 60)
    print("\nWatching for training progress...\n")
    
    # Check if training has started
    if not output_dir.exists():
        print("⏳ Waiting for training to start...")
        print("   Output directory not yet created")
        return
    
    # Look for log files
    log_files = list(output_dir.glob("*.log"))
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        print(f"📝 Latest log: {latest_log.name}")
        
        # Show last few lines
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            if lines:
                print("\nLast 10 lines:")
                print("-" * 40)
                for line in lines[-10:]:
                    print(line.rstrip())
    
    # Check for checkpoints
    checkpoint_dir = output_dir / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        if checkpoints:
            print(f"\n✅ Found {len(checkpoints)} checkpoint(s)")
            latest = max(checkpoints, key=os.path.getctime)
            print(f"   Latest: {latest.name}")
    
    # Check for metrics
    metrics_file = output_dir / "metrics.json"
    if metrics_file.exists():
        import json
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            if 'roc_auc' in metrics:
                print(f"\n📈 Current ROC-AUC: {metrics['roc_auc']:.3f}")
            if 'tpr_at_far' in metrics:
                print(f"   TPR@FAR=1/30d: {metrics['tpr_at_far']:.3f}")

if __name__ == "__main__":
    while True:
        os.system('clear')
        monitor()
        print("\n(Press Ctrl+C to stop monitoring)")
        
        try:
            time.sleep(5)  # Update every 5 seconds
        except KeyboardInterrupt:
            print("\n\n✋ Monitoring stopped")
            sys.exit(0)
