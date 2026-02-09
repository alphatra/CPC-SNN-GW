import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.encoders import DeltaModulationEncoder
from src.models.architectures import SpikingCNN, RSNN
from src.models.layers import DAIN_Layer
from torch.utils.checkpoint import checkpoint
from src.models.tf_encoder import TFEncoder

class CPCSNN(nn.Module):
    """
    Modular CPC-SNN Model.
    """
    def __init__(self, 
                 in_channels=2, 
                 hidden_dim=64, 
                 context_dim=64, 
                 prediction_steps=12,
                 delta_threshold=0.1,
                 temperature=0.07,
                 beta=0.9,
                 use_checkpointing=False,
                 use_metal=True,
                 use_continuous_input=True,
                 no_dain=False,
                 use_tf2d=False,
                 use_layernorm_z=True,
                 use_mlp_head=True,
                 enable_predictors=True):
        super().__init__()
        
        self.prediction_steps = int(prediction_steps) if enable_predictors else 0
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.temperature = temperature
        self.use_checkpointing = use_checkpointing
        self.use_metal = use_metal
        self.use_continuous_input = use_continuous_input
        self.no_dain = no_dain
        self.use_tf2d = use_tf2d
        self.use_layernorm_z = use_layernorm_z
        self.use_mlp_head = use_mlp_head
        
        # 1. Input Architectures
        if self.use_tf2d:
             # TF 2D Encoder Path (Spectrogram -> Conv2D -> Sequence)
             # Input: (B, C, T, F) 
             self.tf_encoder = TFEncoder(in_channels=in_channels, out_channels=128) 
             # Adapter: Dynamic -> hidden_dim
             # Use LazyLinear to avoid hardcoded 2048 which depends on F
             self.adapter = nn.LazyLinear(hidden_dim)
             
             # Disable other paths
             self.dain = nn.Identity()
             self.encoder = nn.Identity()
             self.feature_extractor = nn.Identity()
             
        elif self.use_continuous_input:
            # Step C: Raw Input -> DAIN -> Conv1d (Learnable)
            if not self.no_dain:
                self.dain = DAIN_Layer(max(in_channels, 1), mode='full') 
            else:
                self.dain = nn.Identity()
            self.encoder = nn.Identity() # Bypass delta encoder
            self.feature_extractor = SpikingCNN(in_channels, hidden_dim, beta=beta, use_metal=use_metal)
        else:
            # Legacy: FastDeltaEncoder
            self.encoder = DeltaModulationEncoder(threshold=delta_threshold)
            self.feature_extractor = SpikingCNN(in_channels, hidden_dim, beta=beta, use_metal=use_metal)
            
        # 2. Normalization (LayerNorm on Z)
        self.ln_z = nn.LayerNorm(hidden_dim) if self.use_layernorm_z else nn.Identity()
        
        # 3. Context Network (RSNN)
        self.context_network = RSNN(hidden_dim, context_dim, beta=beta)
        
        # 4. Binary Classifier (Signal vs Noise)
        if self.use_mlp_head:
            self.classifier = nn.Sequential(
                nn.Linear(context_dim, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            )
        else:
            self.classifier = nn.Linear(context_dim, 1)
        
        # 5. CPC Predictors (InfoNCE)
        # Maps Context -> Future Latent Z (dim: context_dim -> hidden_dim)
        # We need K predictors for K steps
        self.predictors = nn.ModuleList()
        if self.prediction_steps > 0:
            self.predictors = nn.ModuleList([
                nn.Linear(context_dim, hidden_dim) for _ in range(self.prediction_steps)
            ])
        
    def predict_future(self, c):
        """
        Generates predictions for future Z steps based on context C.
        Args:
            c (torch.Tensor): Context tensor (B, T, ContextDim)
        Returns:
            List[torch.Tensor]: List of K tensors, each (B, T, HiddenDim)
        """
        preds = []
        for k in range(len(self.predictors)):
            # W_k(c_t) -> predicts z_{t+k}
            preds.append(self.predictors[k](c))
        return preds

    def forward(self, x, is_encoded=False):
        """
        Args:
            x (torch.Tensor): 
                - If use_tf2d=True: (B, 6, T, F)
                - Else: Raw (B, C, T) or Spikes
        """
        # 1. Feature Extraction
        if self.use_tf2d:
             # x is (B, 6, T, F)
             z_tf = self.tf_encoder(x) # (B, T, 2048)
             z = self.adapter(z_tf)   # (B, T, Hidden)
             # No permute needed, already (B, T, Hidden)
             
        else:
            # 1. Process Input
            if self.use_continuous_input and not is_encoded:
                if not self.no_dain:
                    x = self.dain(x)
                spikes = x 
            else:
                if is_encoded:
                    spikes = x
                else:
                    spikes = self.encoder(x)
            
            # 2. Extract features
            if self.training and self.use_checkpointing and spikes.requires_grad:
                 z = checkpoint(self.feature_extractor, spikes, use_reentrant=False)
            else:
                 z = self.feature_extractor(spikes)
            
            # z: (B, Hidden, T_pooled) -> (B, T_pooled, Hidden)
            z = z.permute(0, 2, 1)

        # Apply LayerNorm to z (B, T, Hidden)
        z = self.ln_z(z)

        # 3. Compute Context
        # z is (B, T_seq, Hidden)
        
        # RSNN
        c = self.context_network(z) # (B, T_seq, Context)
        
        # 4. Classification
        # Pooling over Time (Mean) + MLP
        c_pooled = c.mean(dim=1) # (B, Context)
        logits = self.classifier(c_pooled) # (B, 1)
        
        return logits, c, z

