import torch
import torch.nn as nn
import numpy as np

class DAIN_Layer(nn.Module):
    def __init__(self, input_dim=2, mode='full', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001):
        super(DAIN_Layer, self).__init__()
        
        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.eye(input_dim)

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.eye(input_dim)

        # Parameters for adaptive gating
        self.gating_layer = nn.Linear(input_dim, input_dim)

        self.eps = 1e-8

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        # DAIN oczekuje średniej po czasie (dim=2)
        
        # Step 1: Adaptive Centering
        avg = torch.mean(x, 2)
        adaptive_avg = self.mean_layer(avg)
        adaptive_avg = adaptive_avg.unsqueeze(2) # (B, C, 1)
        x = x - adaptive_avg

        # Step 2: Adaptive Scaling
        std = torch.mean(x ** 2, 2)
        std = torch.sqrt(std + self.eps)
        adaptive_std = self.scaling_layer(std)
        adaptive_std = torch.where(adaptive_std <= self.eps, torch.tensor(1.0, device=x.device), adaptive_std)
        adaptive_std = adaptive_std.unsqueeze(2) # (B, C, 1)
        x = x / adaptive_std

        # Step 3: Gating (Opcjonalne tłumienie szumu)
        avg = torch.mean(x, 2)
        gate = torch.sigmoid(self.gating_layer(avg))
        gate = gate.unsqueeze(2) # (B, C, 1)
        x = x * gate
        
        return x
