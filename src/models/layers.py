import torch
import torch.nn as nn
import numpy as np

class DAIN_Layer(nn.Module):
    def __init__(self, input_dim, mode='adaptive_scale', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001):
        super(DAIN_Layer, self).__init__()
        print(f"Initializing DAIN_Layer with mode: {mode}")

        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive mean
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.eye(input_dim)

        # Parameters for adaptive scaling
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.eye(input_dim)

        # Parameters for gating
        self.gating_layer = nn.Linear(input_dim, input_dim)

        self.eps = 1e-8

    def forward(self, x):
        # Expecting x in shape (Batch, Channels, Time)
        # DAIN usually works on the feature dimension. 
        # Here Channels=Feature=2 (H1, L1).
        
        # Transpose to (Batch, Time, Channels) for Linear layers
        x = x.transpose(1, 2)
        
        # 1. Adaptive Mean
        avg = torch.mean(x, dim=1)  # (Batch, Channels)
        adaptive_avg = self.mean_layer(avg)  # (Batch, Channels)
        adaptive_avg = adaptive_avg.unsqueeze(1) # (Batch, 1, Channels)
        x = x - adaptive_avg

        # 2. Adaptive Scaling
        std = torch.std(x, dim=1) # (Batch, Channels)
        adaptive_std = self.scaling_layer(std) # (Batch, Channels)
        adaptive_std = adaptive_std.unsqueeze(1)
        adaptive_std = torch.clamp(adaptive_std, min=self.eps) # Avoid div by zero
        x = x / adaptive_std

        # 3. Gating (Optional but recommended)
        # Gating helps suppress noise
        if self.mode == 'full':
            gate = torch.sigmoid(self.gating_layer(x))
            x = x * gate
            
        # Transpose back to (Batch, Channels, Time)
        x = x.transpose(1, 2)
        
        return x
