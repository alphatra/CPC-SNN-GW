import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.encoders import DeltaModulationEncoder
from src.models.architectures import SpikingCNN, RSNN

class CPCSNN(nn.Module):
    """
    Modular CPC-SNN Model.
    
    Pipeline:
    1. Raw Signal -> Delta Modulation -> Spikes
    2. Spikes -> Spiking CNN -> Latent Features (z)
    3. Latent Features -> RSNN -> Context (c)
    4. Context -> Linear Predictors -> Future Latents (z_pred)
    
    Args:
        in_channels (int): Number of input channels (e.g., 2 for H1, L1).
        hidden_dim (int): Dimension of latent features (z).
        context_dim (int): Dimension of context vector (c).
        prediction_steps (int): Number of future steps to predict (k).
        delta_threshold (float): Threshold for delta modulation.
    """
    def __init__(self, 
                 in_channels=2, 
                 hidden_dim=64, 
                 context_dim=64, 
                 prediction_steps=12,
                 delta_threshold=0.1,
                 temperature=0.07,
                 beta=0.9):
        super().__init__()
        
        self.prediction_steps = prediction_steps
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        # 1. Encoder (Delta Mod)
        self.encoder = DeltaModulationEncoder(threshold=delta_threshold)
        # Script the encoder instance for performance
        try:
            self.encoder = torch.jit.script(self.encoder)
        except Exception as e:
            print(f"WARNING: Failed to script encoder: {e}")
        
        # 2. Feature Extractor (Spiking CNN)
        self.feature_extractor = SpikingCNN(in_channels, hidden_dim, beta=beta)
        
        # 3. Context Network (RSNN)
        # Input to RSNN is z (hidden_dim)
        self.context_network = RSNN(hidden_dim, context_dim, beta=beta)
        
        # 4. Predictors
        # W_k transforms context c_t to predicted z_{t+k}
        self.predictors = nn.ModuleList([
            nn.Linear(context_dim, hidden_dim) for _ in range(prediction_steps)
        ])

    def forward(self, x, is_encoded=False):
        """
        Args:
            x (torch.Tensor): Raw input (Batch, Channels, Time) OR Spikes if is_encoded=True
            is_encoded (bool): If True, x is already spikes.
            
        Returns:
            z (torch.Tensor): Latent features (Batch, Time_pooled, Hidden)
            c (torch.Tensor): Context vectors (Batch, Time_pooled, Context)
        """
        # 1. Encode to spikes
        # x: (B, C, T) -> spikes: (B, C, T)
        if is_encoded:
            spikes = x
        else:
            spikes = self.encoder(x)
        
        # 2. Extract features
        # spikes: (B, C, T) -> z: (B, Hidden, T_pooled)
        z = self.feature_extractor(spikes)
        
        # 3. Compute Context
        # Transpose for RSNN: (B, T_pooled, Hidden)
        z = z.permute(0, 2, 1)
        
        # RSNN
        # c: (B, T_pooled, Context)
        c = self.context_network(z)
        
        return z, c, spikes

    def compute_cpc_loss(self, z, c, spikes=None):
        """
        Computes the InfoNCE loss for CPC.
        
        Args:
            z (torch.Tensor): Latent features (Batch, Time, Hidden)
            c (torch.Tensor): Context vectors (Batch, Time, Context)
            
        Returns:
            loss (torch.Tensor): Scalar loss.
            accuracy (float): Prediction accuracy.
        """
        batch_size, time_steps, _ = z.shape
        loss = 0
        correct = 0
        correct5 = 0
        total_pos = 0
        total_neg = 0
        valid_steps = 0
        total = 0
        
        # We can only predict up to time_steps - k
        # For each prediction step k
        for k in range(1, self.prediction_steps + 1):
            # Predictor for step k
            W_k = self.predictors[k-1]
            
            # Context at time t: c[:, :-k, :]
            # Target at time t+k: z[:, k:, :]
            
            if time_steps <= k:
                continue
                
            c_t = c[:, :-k, :]
            z_tk = z[:, k:, :]
            
            # Prediction: z_hat = W_k(c_t)
            z_pred = W_k(c_t) # (Batch, Time-k, Hidden)
            
            # InfoNCE Loss
            # We want z_pred[b, t] to be close to z_tk[b, t]
            # and far from z_tk[b', t] (other samples in batch)
            # or z_tk[b, t'] (other time steps - simpler version uses batch negatives)
            
            # Reshape for matmul
            # z_pred: (Batch * (Time-k), Hidden)
            # z_tk: (Batch * (Time-k), Hidden)
            
            z_pred_flat = z_pred.reshape(-1, self.hidden_dim)
            z_tk_flat = z_tk.reshape(-1, self.hidden_dim)
            
            # Logits: (Batch*Time, Batch*Time)
            # This is too big. Let's sample negatives or use batch-wise.
            # Standard CPC uses batch negatives.
            # Let's compute loss per time-step to save memory, or just subsample.
            # For simplicity here, let's just use the batch dimension at a random time step?
            # No, we should use all time steps.
            
            # Efficient implementation:
            # logits = z_pred . z_tk.T
            # But we only want negatives from the same time step across batch?
            # Or any negative?
            # Usually: negatives are other samples in the batch.
            
            # Let's iterate over time steps to avoid huge matrix
            # Or just take the mean over time steps of the loss.
            
            # Simplified: Compute loss for the last time step only (or a few random ones)
            # to speed up training? No, we want dense supervision.
            
            # Let's do it properly but efficiently.
            # We can treat (Batch * Time) as independent samples.
            # But (Batch * Time)^2 is huge.
            # Let's restrict negatives to be from the same time step but different batch indices.
            
            # z_pred: (B, T-k, H)
            # z_tk: (B, T-k, H)
            
            # Calculate scores: (B, T-k, B)
            # score[b, t, b'] = z_pred[b, t] . z_tk[b', t]
            
            # Normalize vectors for cosine similarity
            z_pred = F.normalize(z_pred, dim=-1)
            z_tk = F.normalize(z_tk, dim=-1)

            # Einstein summation: bth, bth -> btb (dot product over hidden, broadcast over batch)
            # z_pred[b, t, h] * z_tk[b', t, h]
            logits = torch.einsum('bth, ath -> bta', z_pred, z_tk)
            
            # Scale by temperature
            logits = logits / self.temperature
            
            # Target is diagonal: labels[b, t] = b
            labels = torch.arange(batch_size).to(z.device)
            labels = labels.unsqueeze(1).expand(batch_size, time_steps - k) # (B, T-k)
            
            # Logits: (B, T-k, B). We need to flatten T-k into Batch for CrossEntropy?
            # CrossEntropy expects (N, C). Here N = B*(T-k), C = B.
            
            logits_flat = logits.reshape(-1, batch_size) # (B*(T-k), B)
            labels_flat = labels.reshape(-1) # (B*(T-k))
            
            step_loss = F.cross_entropy(logits_flat, labels_flat)
            loss += step_loss
            
            # Accuracy Top-1
            preds = torch.argmax(logits_flat, dim=1)
            correct += (preds == labels_flat).sum().item()
            
            # Accuracy Top-5
            # logits_flat: (N, B)
            if logits_flat.size(1) >= 5:
                _, pred5 = logits_flat.topk(5, 1, True, True)
                pred5 = pred5.t()
                correct5_k = pred5.eq(labels_flat.view(1, -1).expand_as(pred5))
                correct5 += correct5_k[:5].reshape(-1).float().sum().item()
            else:
                # Fallback if batch size < 5
                correct5 += (preds == labels_flat).sum().item()

            # Margin Metrics
            # Positives: logits_flat[range(N), labels_flat]
            pos_scores = logits_flat.gather(1, labels_flat.unsqueeze(1)).squeeze()
            total_pos += pos_scores.mean().item()
            
            # Negatives: Mean of non-target logits
            # We can approximate or compute exactly.
            # Exact: (Sum(logits) - Sum(pos)) / (N * (B-1))
            sum_logits = logits_flat.sum(dim=1)
            sum_pos = pos_scores
            mean_neg = (sum_logits - sum_pos) / (batch_size - 1)
            total_neg += mean_neg.mean().item()
            
            valid_steps += 1
            
            total += labels_flat.size(0)
            
        loss /= self.prediction_steps
        
        # --- REGULARYZACJA ---
        if spikes is not None:
            # Celujemy w np. 5% aktywności (0.05)
            # spikes to tensor (Batch, Channels, Time)
            # Convert spikes to float for mean calculation if they are boolean/byte
            if spikes.dtype == torch.bool or spikes.dtype == torch.uint8:
                spikes = spikes.float()
                
            mean_activity = torch.mean(spikes)
            reg_loss = (mean_activity - 0.05) ** 2 
            
            # Dodajemy do głównego lossa z wagą lambda (np. 2.0)
            loss = loss + 2.0 * reg_loss 
        # ---------------------

        acc1 = correct / total if total > 0 else 0
        acc5 = correct5 / total if total > 0 else 0
        avg_pos = total_pos / valid_steps if valid_steps > 0 else 0
        avg_neg = total_neg / valid_steps if valid_steps > 0 else 0
        
        metrics = {
            "acc1": acc1,
            "acc5": acc5,
            "pos_score": avg_pos,
            "neg_score": avg_neg
        }
        
        return loss, metrics
