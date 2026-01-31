import torch
from torch.utils.cpp_extension import load
import os

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
mps_src = os.path.join(current_dir, "metal/lif_op.mm")
metal_src = os.path.join(current_dir, "metal/lif_kernel.metal")

# Read Metal Kernel Source
if os.path.exists(metal_src):
    with open(metal_src, 'r') as f:
        kernel_source = f.read()
else:
    kernel_source = ""
    print(f"Warning: Metal kernel not found at {metal_src}")

# JIT Compile the extension
# We link against Metal and Foundation frameworks.
lif_metal = load(
    name="lif_metal",
    sources=[mps_src],
    extra_cflags=["-std=c++17"],
    extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
    verbose=True
)

def fused_lif_metal_update(currents: torch.Tensor, beta: float, threshold: float = 1.0) -> torch.Tensor:
    """
    Fused LIF update using Native Metal Kernel.
    
    Args:
        currents (torch.Tensor): Input currents (Batch, Channels, Time) or (Batch, Time, Channels) on CPU.
        beta (float): Decay rate.
        threshold (float): Firing threshold.
        
    Returns:
        spikes (torch.Tensor): Binary spikes on CPU (Unified Memory).
    """
    if currents.device.type != 'cpu':
        # Fallback or error?
        # User requested native metal to bypass overhead. 
        # Best approach: Input should be pinned CPU tensor.
        currents = currents.cpu()
        
    # FIXED: Ensure float32 and contiguous memory before C++ call
    # C++ code assumes float32, but AMP might pass bfloat16/fp16
    currents = currents.to(dtype=torch.float32, memory_format=torch.contiguous_format)
    
    # FIXED: Convert beta/threshold to python floats if they are tensors/Parameters
    if torch.is_tensor(beta):
        beta = float(beta.item())
    else:
        beta = float(beta)
    
    if torch.is_tensor(threshold):
        threshold = float(threshold.item())
    else:
        threshold = float(threshold)
        
    return lif_metal.fused_lif_forward(currents, beta, threshold, kernel_source)
