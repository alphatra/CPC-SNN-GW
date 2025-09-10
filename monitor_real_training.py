#!/usr/bin/env python3
"""
Monitor REAL training progress
"""

import os
import time
from pathlib import Path
import sys

def monitor():
    """Monitor real training outputs."""
    
    print("=" * 70)
    print("📊 REAL Training Monitor (MLGWSC-1 Dataset)")
    print("=" * 70)
    
    # Check log file
    log_file = Path("real_training.log")
    
    if log_file.exists():
        print("\n📝 Training Log:")
        print("-" * 50)
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
            # Show status
            if lines:
                # Look for key indicators
                for line in lines:
                    if "Loading REAL MLGWSC-1" in line:
                        print("✅ Loading real dataset...")
                    elif "Train:" in line and "samples" in line:
                        print(f"📊 {line.strip()}")
                    elif "Test:" in line and "samples" in line:
                        print(f"📊 {line.strip()}")
                    elif "Class balance:" in line:
                        print(f"⚖️  {line.strip()}")
                    elif "Starting REAL training" in line:
                        print("🚀 Training started!")
                    elif "Epoch" in line and "/" in line:
                        print(f"🔥 {line.strip()}")
                    elif "Loss:" in line:
                        print(f"   {line.strip()}")
                    elif "accuracy" in line.lower():
                        print(f"   📈 {line.strip()}")
                    elif "ROC-AUC" in line:
                        print(f"   🎯 {line.strip()}")
                
                # Show last 5 lines
                print("\nLast 5 lines:")
                print("-" * 30)
                for line in lines[-5:]:
                    print(line.rstrip())
    else:
        print("⏳ Waiting for training to start...")
        print("   Log file not yet created")
    
    # Check if process is still running
    import subprocess
    result = subprocess.run(['pgrep', '-f', 'train_real_mlgwsc.py'], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("\n✅ Training process is RUNNING (PID: {})".format(result.stdout.strip()))
    else:
        if log_file.exists() and os.path.getsize(log_file) > 0:
            print("\n🏁 Training COMPLETED or stopped")
        else:
            print("\n⚠️  Training not yet started")

if __name__ == "__main__":
    print("\n🔄 Monitoring real training... (Press Ctrl+C to stop)\n")
    
    while True:
        os.system('clear')
        monitor()
        
        try:
            time.sleep(3)  # Update every 3 seconds
        except KeyboardInterrupt:
            print("\n\n✋ Monitoring stopped")
            sys.exit(0)
