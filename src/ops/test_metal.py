import torch
import time
import os

print("Compiling Metal Extension...")
try:
    from src.ops.fused_lif import fused_lif_metal_update
    print("Extension loaded successfully.")
except Exception as e:
    print(f"Failed to load extension: {e}")
    exit(1)

def test_lif():
    # Parameters
    B, C, T = 2, 4, 10
    beta = 0.5
    threshold = 1.0
    
    # Inputs (CPU, Pinned for Unified Memory speed)
    currents = torch.ones(B, C, T, dtype=torch.float32)
    
    # Run Metal Op
    print("Running Metal Kernel...")
    t0 = time.time()
    spikes = fused_lif_metal_update(currents, beta, threshold)
    t1 = time.time()
    
    print(f"Execution time: {(t1-t0)*1000:.2f} ms")
    print(f"Output shape: {spikes.shape}")
    print("Spikes subset:\n", spikes[0, 0, :])
    
    # Verification
    # Logic: input=1.0, beta=0.5, thr=1.0
    # t=0: mem=1.0. Spike=0 (mem > 1.0 is False? Or >=? Code: mem > threshold).
    # If mem > 1.0:
    # t=0: mem = 1.0. Spike=0.
    # t=1: mem = 1.0*0.5 + 1.0 = 1.5. Spike=1. mem=0.5.
    # t=2: mem = 0.5*0.5 + 1.0 = 1.25. Spike=1. mem=0.25.
    # t=3: mem = 0.25*0.5 + 1.0 = 1.125. Spike=1. mem=0.125.
    # t=4: mem = 0.125*0.5 + 1.0 = 1.0625. Spike=1. mem=0.0625.
    # ...
    
    expected_spikes = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32)
    diff = (spikes[0, 0, :] - expected_spikes).abs().sum()
    
    if diff == 0:
        print("Verification PASSED: Spikes match expected sequence.")
    else:
        print("Verification FAILED.")
        print("Expected:", expected_spikes)
        print("Got:", spikes[0, 0, :])

if __name__ == "__main__":
    test_lif()
