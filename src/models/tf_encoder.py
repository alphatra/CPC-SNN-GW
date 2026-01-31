import torch
import torch.nn as nn
import torch.nn.functional as F

class TFEncoder(nn.Module):
    """
    Time-Frequency 2D Encoder.
    
    Processes STFT spectrograms (Batch, Channels, Time, Freq) using 2D convolutions.
    Designed to compress the Frequency dimension while preserving the Time dimension
    for downstream sequence modeling (CPC/SNN).
    
    Architecture:
    - Input: [B, Cin, T, F]
    - Backbone: 4 blocks of Conv2d(k=(3,7), s=(1,2), p=(1,3)) -> Freq / 16
    - Output: [B, T, D] where D = Cout * F_final
    """
    def __init__(self, in_channels=6, base_channels=32, out_channels=128):
        super().__init__()
        
        self.in_channels = in_channels
        
        # Block 1
        # Stride (1, 2) -> T stays, F /= 2
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3))
        self.gn1 = nn.GroupNorm(4, base_channels)
        
        # Block 2
        self.conv2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3))
        self.gn2 = nn.GroupNorm(8, base_channels*2)
        
        # Block 3
        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3))
        self.gn3 = nn.GroupNorm(16, base_channels*4)
        
        # Block 4
        self.conv4 = nn.Conv2d(base_channels*4, base_channels*4, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3))
        self.gn4 = nn.GroupNorm(16, base_channels*4)
        
        # Mixing Block (1x1 Conv to adjust per-location features)
        self.conv_mix = nn.Conv2d(base_channels*4, out_channels, kernel_size=1)
        
        self.act = nn.GELU()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrogram (B, C, T, F)
            
        Returns:
            z (torch.Tensor): Sequence features (B, T, D)
        """
        # B, C, T, F
        
        # Block 1
        x = self.act(self.gn1(self.conv1(x)))
        
        # Block 2
        x = self.act(self.gn2(self.conv2(x)))
        
        # Block 3
        x = self.act(self.gn3(self.conv3(x)))
        
        # Block 4
        x = self.act(self.gn4(self.conv4(x)))
        
        # Mixing
        x = self.conv_mix(x) # (B, OutC, T, F')
        
        # Frequency Pooling (Mean)
        # x: (B, C_out, T, F_prime) -> (B, C_out, T)
        x = x.mean(dim=3)
        
        # Format for Sequence Modeling: (B, T, C_out)
        x = x.permute(0, 2, 1) # (B, T, D)
        
        return x

if __name__ == "__main__":
    # verification
    B, C, T, F = 2, 6, 31, 246
    model = TFEncoder(in_channels=C)
    
    x_test = torch.randn(B, C, T, F)
    z = model(x_test)
    
    print(f"Input: {x_test.shape}")
    print(f"Output: {z.shape}")
    
    Expected_F = F // 16
    print(f"Expected F_final approx: {Expected_F}")
    # Calculation:
    # 246 -> 123 -> 62 -> 31 -> 16
    
    print(f"Output Dim (D): {z.shape[2]}")
