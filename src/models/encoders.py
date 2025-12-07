import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # Make threshold a learnable parameter
        self.threshold = nn.Parameter(torch.tensor(float(threshold)))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Time)
            
        Returns:
            spikes (torch.Tensor): Spike tensor of shape (Batch, Channels, Time) with values {-1, 0, 1}.
        """
        # Ensure threshold is within [0.01, 0.5] to prevent saturation
        threshold = torch.clamp(self.threshold, min=0.01, max=0.5)
        
        # Initialize reconstruction
        recon = torch.zeros_like(x[:, :, 0])
        spikes = []
        
        # Iterate over time steps
        for t in range(x.shape[2]):
            current_input = x[:, :, t]
            
            # Calculate error
            error = current_input - recon
            
            # Generate spikes
            pos_spikes = (error > threshold).float()
            neg_spikes = (error < -threshold).float()
            
            # Update reconstruction
            recon += pos_spikes * threshold
            recon -= neg_spikes * threshold
            
            # Combine spikes (1 for pos, -1 for neg)
            # Note: In rare cases where threshold is 0, both could be true, but threshold > 0 usually.
            net_spikes = pos_spikes - neg_spikes
            spikes.append(net_spikes)
            
        # Stack along time dimension
        return torch.stack(spikes, dim=2)

class FastDeltaEncoder(nn.Module):
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        # Approximation of Delta Modulation using differences (vectorized)
        # Instead of tracking state, we look at local changes
        diff = x[:, :, 1:] - x[:, :, :-1]
        
        # Pad to maintain time dimension
        diff = F.pad(diff, (1, 0)) 
        
        spikes = torch.zeros_like(x)
        spikes[diff > self.threshold] = 1.0
        spikes[diff < -self.threshold] = -1.0
        
        return spikes

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
