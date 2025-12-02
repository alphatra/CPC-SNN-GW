import torch
import torch.nn as nn

# Static JIT-compiled function for Delta Modulation
@torch.jit.script
def delta_encode_jit(x: torch.Tensor, threshold: float) -> torch.Tensor:
    # x: (Batch, Channels, Time)
    B, C, T = x.shape
    
    # Pre-allocate output tensors
    spikes = torch.zeros_like(x)
    recon = torch.zeros((B, C), device=x.device, dtype=x.dtype)
    
    # Loop runs in C++ (TorchScript)
    for t in range(T):
        current_input = x[:, :, t]
        error = current_input - recon
        
        pos_spikes = (error > threshold).float()
        neg_spikes = (error < -threshold).float()
        
        # Update
        net_spike = pos_spikes - neg_spikes
        recon += net_spike * threshold
        
        # Store spikes
        spikes[:, :, t] = net_spike
        
    return spikes

class DeltaModulationEncoder(nn.Module):
    """
    Delta Modulation Encoder for converting continuous signals into spikes.
    Uses JIT compilation for performance.
    """
    def __init__(self, threshold: float = 0.1):
        super().__init__()
        # Register threshold as buffer for JIT compatibility
        self.register_buffer('threshold', torch.tensor(float(threshold)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Time)
            
        Returns:
            spikes (torch.Tensor): Spike tensor of shape (Batch, Channels, Time) with values {-1, 0, 1}.
        """
        # Ensure threshold is within reasonable bounds
        # Note: In JIT, we pass the float value
        thresh_val = float(torch.clamp(self.threshold, min=0.01, max=0.5).item())
        return delta_encode_jit(x, thresh_val)

class ThresholdCrossingEncoder(nn.Module):
    """
    Simple Threshold Crossing Encoder.
    Generates a spike if the signal derivative (diff) exceeds a threshold.
    
    Args:
        threshold (float): Threshold for the derivative.
    """
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Time)
        """
        # Calculate temporal difference
        diff = x[:, :, 1:] - x[:, :, :-1]
        
        # Pad the first time step to maintain length
        zeros = torch.zeros_like(x[:, :, 0:1])
        diff = torch.cat([zeros, diff], dim=2)
        
        pos_spikes = (diff > self.threshold).float()
        neg_spikes = (diff < -self.threshold).float()
        
        return pos_spikes - neg_spikes
