import torch
import time
from src.models.architectures import SpikingCNN
from src.ops.fused_lif import fused_lif_metal_update

def benchmark():
    # Setup
    B, C, T = 128, 2, 2048 # Realistic Batch 128, 1s sample
    in_channels = 2
    hidden_dim = 64
    steps = 20
    
    device = torch.device("mps")
    
    # 1. Baseline Model
    model_base = SpikingCNN(in_channels, hidden_dim, use_metal=False).to(device)
    model_base.eval()
    
    # 2. Metal Model
    model_metal = SpikingCNN(in_channels, hidden_dim, use_metal=True).to(device)
    model_metal.eval()
    
    # Input
    x = torch.randn(B, C, T).to(device)
    
    print(f"Benchmarking Inference (Batch={B}, Time={T})...")
    
    # Warmup
    print("Warmup...")
    with torch.no_grad():
        for _ in range(5):
            model_base(x)
            if model_metal.use_metal: model_metal(x)
            
    # Test Baseline
    torch.mps.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(steps):
            model_base(x)
    torch.mps.synchronize()
    t_base = (time.time() - t0) / steps
    print(f"Baseline (PyTorch Loop): {t_base*1000:.2f} ms / batch")
    
    # Test Metal
    torch.mps.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(steps):
            model_metal(x)
    torch.mps.synchronize()
    t_metal = (time.time() - t0) / steps
    print(f"Metal (Fused Kernel):    {t_metal*1000:.2f} ms / batch")
    
    speedup = t_base / t_metal
    print(f"Speedup: {speedup:.2f}x 🚀")

if __name__ == "__main__":
    benchmark()
