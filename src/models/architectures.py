import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class SpikingCNN(nn.Module):
    """
    Spiking CNN Feature Extractor.
    
    Structure:
    - Block 1: Conv1d -> LIF -> MaxPool
    - Block 2: Conv1d -> LIF -> MaxPool
    - Block 3: Conv1d -> LIF -> MaxPool
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (latent dimension).
        beta (float): Decay rate for LIF neurons.
    """
    def __init__(self, in_channels, out_channels=64, beta=0.9):
        super().__init__()
        
        # Use atan surrogate gradient for better stability
        spike_grad = surrogate.atan()
        
        # Block 1
        # Stride 4 to reduce time steps (1024 -> 256) for speed on MPS
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5, padding=2, stride=4)
        self.bn1 = nn.BatchNorm1d(16)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
        self.pool1 = nn.MaxPool1d(2)
        
        # Block 2
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
        self.pool2 = nn.MaxPool1d(2)
        
        # Block 3
        self.conv3 = nn.Conv1d(32, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
        self.pool3 = nn.MaxPool1d(2)

        
    def forward_block(self, x, conv, lif, pool, bn=None):
        # 1. Conv over whole sequence
        # x: (Batch, Channels, Time)
        currents = conv(x)
        
        # Apply BatchNorm if provided
        if bn is not None:
            currents = bn(currents)
        
        # 2. LIF integration
        # Transpose to (Time, Batch, Channels) for looping
        currents = currents.permute(2, 0, 1)
        spk_rec = []
        mem = lif.init_leaky()
        
        for step in range(currents.shape[0]):
            spk, mem = lif(currents[step], mem)
            spk_rec.append(spk)
            
        # Stack back to (Time, Batch, Channels)
        spikes = torch.stack(spk_rec, dim=0)
        
        # Transpose back to (Batch, Channels, Time)
        spikes = spikes.permute(1, 2, 0)
        
        # 3. Pooling
        out = pool(spikes)
        
        return out

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spikes (Batch, Channels, Time)
        Returns:
            z (torch.Tensor): Latent features (Batch, Out_Channels, Time')
        """
        x = self.forward_block(x, self.conv1, self.lif1, self.pool1, self.bn1)
        x = self.forward_block(x, self.conv2, self.lif2, self.pool2, self.bn2)
        x = self.forward_block(x, self.conv3, self.lif3, self.pool3, self.bn3)
        return x

class RSNN(nn.Module):
    """
    Recurrent Spiking Neural Network (RSNN) for Autoregressive Context.
    
    Uses RLeaky neurons to aggregate history.
    Returns membrane potential as continuous context vector (Readout).
    
    Args:
        input_size (int): Dimension of input features (z).
        hidden_size (int): Dimension of context vector (c).
        beta (float): Decay rate.
    """
    def __init__(self, input_size, hidden_size, beta=0.9):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Linear projection from input to hidden
        self.linear = nn.Linear(input_size, hidden_size)
        
        # LayerNorm to stabilize input to recurrent neuron
        self.ln = nn.LayerNorm(hidden_size)
        
        # Use atan surrogate gradient for better stability
        spike_grad = surrogate.atan()
        
        # Recurrent Leaky Neuron
        # all_to_all=True means fully connected recurrent weights
        self.rleaky = snn.RLeaky(beta=beta, spike_grad=spike_grad, all_to_all=True, linear_features=hidden_size, init_hidden=False)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Time, Input_Size)
        Returns:
            c (torch.Tensor): Context vectors (Batch, Time, Hidden_Size) - using Membrane Potential
        """
        batch_size, time_steps, _ = x.shape
        
        # Initialize hidden states
        spk, mem = self.rleaky.init_rleaky()
        spk = spk.to(x.device)
        mem = mem.to(x.device)
        
        # Handle batch size mismatch if init returns simplistic shapes or wrong batch size
        if spk.shape[0] != batch_size:
             if spk.dim() == 0 or (spk.dim() > 0 and spk.shape[0] == 1):
                 spk = spk.expand(batch_size, -1)
                 mem = mem.expand(batch_size, -1)
             else:
                 # If shape mismatch and not broadcastable (e.g. previous batch size 64 vs current 32)
                 # Re-initialize with zeros
                 spk = torch.zeros(batch_size, self.hidden_size).to(x.device)
                 mem = torch.zeros(batch_size, self.hidden_size).to(x.device)

        mem_rec = []
        
        for t in range(time_steps):
            # Input at time t
            inp = self.linear(x[:, t, :])
            
            # --- APLIKACJA LAYERNORM ---
            inp = self.ln(inp)
            # ---------------------------
            
            # Recurrent step
            spk, mem = self.rleaky(inp, spk, mem)
            
            # Clamp membrane potential
            if mem.requires_grad:
                mem.register_hook(lambda x: torch.clamp(x, -1.0, 1.0))
            
            mem_rec.append(mem)
            
        return torch.stack(mem_rec, dim=1)
