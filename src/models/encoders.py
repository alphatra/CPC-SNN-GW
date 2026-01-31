import torch
import torch.nn as nn

class DeltaModulationEncoder(nn.Module):
    """
    Delta Modulation Encoder for converting continuous signals into spikes.
    
    Implements a temporal delta modulator:
    - Tracks a reconstruction of the signal.
    - If input > reconstruction + threshold -> Positive Spike (+1), Reconstruction += threshold
    - If input < reconstruction - threshold -> Negative Spike (-1), Reconstruction -= threshold
    
    Args:
        threshold (float): The delta threshold for spike generation.
        channels (int): Number of input channels.
    """
    def __init__(self, threshold=0.1):
        super().__init__()
        # Buffer for JIT compatibility (not a parameter to be learned via gradient descent here)
        self.register_buffer('threshold', torch.tensor(float(threshold)))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Time)
            
        Returns:
            spikes (torch.Tensor): Spike tensor of shape (Batch, Channels, Time) with values {-1, 0, 1}.
        """
        return delta_encode_loop(x, self.threshold.item())

# JIT-compiled loop for performance
@torch.jit.script
def delta_encode_loop(x: torch.Tensor, threshold: float) -> torch.Tensor:
    # x: (Batch, Channels, Time)
    batch, channels, time = x.shape
    
    # Pre-allocate output
    spikes_out = torch.zeros_like(x)
    
    # Initialize reconstruction
    recon = torch.zeros((batch, channels), device=x.device, dtype=x.dtype)
    
    # Clamp threshold
    thr = max(threshold, 0.01) # Simple safety check
    
    for t in range(time):
        current_input = x[:, :, t]
        error = current_input - recon
        
        pos_spikes = (error > thr).float()
        neg_spikes = (error < -thr).float()
        
        recon += pos_spikes * thr
        recon -= neg_spikes * thr
        
        # In-place update
        spikes_out[:, :, t] = pos_spikes - neg_spikes
            
    return spikes_out


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
