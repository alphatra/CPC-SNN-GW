import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.encoders import DeltaModulationEncoder, FastDeltaEncoder
from src.models.architectures import SpikingCNN, RSNN
from src.models.layers import DAIN_Layer

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
                 beta=0.9,
                 learnable_encoding=False):
        super().__init__()
        
        self.prediction_steps = prediction_steps
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        # 0. DAIN - Adaptive Input Normalization
        self.dain = DAIN_Layer(input_dim=in_channels)
        
        # 1. Encoder 
        # If learnable_encoding=True, use nn.Identity() to pass continuous signal 
        # directly to the first Conv1d layer of SpikingCNN.
        if learnable_encoding:
            self.encoder = nn.Identity()
            print("Using Learnable Encoding (Direct Continuous Input to SpikingCNN)")
        else:
            # Use Fast Vectorized Delta Mod
            self.encoder = FastDeltaEncoder(threshold=delta_threshold)
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
            x = self.dain(x)
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

    def compute_cpc_loss(self, z, c, spikes=None, mask=None):
        """
        Computes the InfoNCE loss for CPC.
        
        Args:
            z (torch.Tensor): Latent features (Batch, Time, Hidden)
            c (torch.Tensor): Context vectors (Batch, Time, Context)
            mask (torch.Tensor, optional): Boolean mask (Batch, Time) where True is valid data.
            
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
            
            # Prepare Mask for this step
            # mask_t corresponds to c_t (time t)
            # mask_tk corresponds to z_tk (time t+k)
            # Both must be valid for a valid prediction pair
            step_mask = None
            if mask is not None:
                mask_t = mask[:, :-k]
                mask_tk = mask[:, k:]
                step_mask = mask_t & mask_tk # (B, T-k)
                
                if step_mask.sum() == 0:
                    continue
            
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
            
            # Apply Mask if present
            if step_mask is not None:
                # step_mask: (B, T-k)
                mask_flat = step_mask.reshape(-1) # (B*(T-k))
                
                # Filter valid samples
                if mask_flat.sum() > 0:
                    logits_flat = logits_flat[mask_flat]
                    labels_flat = labels_flat[mask_flat]
                else:
                    continue
            
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
            # Note: labels_flat are indices 0..B-1. But logits_flat is (N_valid, B).
            # So labels_flat tells us which column is the positive.
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
            
            # Apply mask to regularization if needed?
            # Spikes are (B, C, T_original) or (B, C, T_pooled)?
            # Spikes passed here are z (from forward output), which is (B, H, T_pooled).
            # Wait, forward returns z, c, spikes.
            # spikes is (B, C, T_original).
            # z is (B, H, T_pooled).
            # The argument name is 'spikes', but in train_cpc.py we pass 'spikes=z'.
            # line 412: loss, metrics = self.model.compute_cpc_loss(z, c, spikes=z)
            # So 'spikes' argument is actually 'z' (latent features).
            # This seems to be a legacy naming or intent to regularize z activity.
            # If it is z, shape is (B, H, T_pooled).
            # We should mask it too.
            
            if mask is not None:
                # mask is (B, T_pooled)
                # z is (B, H, T_pooled) -> (B, T_pooled, H)
                # We want to mask time steps.
                # mask_expanded: (B, T_pooled, 1)
                mask_expanded = mask.unsqueeze(-1)
                # z_masked = z.permute(0, 2, 1) * mask_expanded # (B, T, H)
                # Actually we just want mean over valid elements.
                
                # z is (B, T, H) in compute_cpc_loss (permuted in forward before RSNN, but forward returns z as (B, H, T_pooled)?
                # Let's check forward.
                # forward: z = self.feature_extractor(spikes) -> (B, H, T_pooled)
                # z = z.permute(0, 2, 1) -> (B, T_pooled, H)
                # returns z, c, spikes.
                # So z is (B, T, H).
                
                # So spikes=z argument is (B, T, H).
                # mask is (B, T).
                
                mask_bool = mask.bool()
                z_valid = z[mask_bool] # (N_valid, H)
                mean_activity = torch.mean(z_valid)
            else:
                mean_activity = torch.mean(spikes) # spikes is z here
                
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

    def compute_barlow_twins_loss(self, z1, z2, lambda_coeff=5e-3):
        """
        Computes the Barlow Twins loss (Redundancy Reduction).
        
        Args:
            z1 (torch.Tensor): Embeddings from View 1 (Batch, Time, Hidden) or (Batch, Hidden)
            z2 (torch.Tensor): Embeddings from View 2 (Batch, Time, Hidden) or (Batch, Hidden)
            lambda_coeff (float): Weight for the off-diagonal terms.
            
        Returns:
            loss (torch.Tensor): Scalar loss.
            metrics (dict): Dictionary of metrics.
        """
        # If inputs are temporal (Batch, Time, Hidden), we can either:
        # 1. Pool over time (Mean/Max) to get (Batch, Hidden)
        # 2. Treat each time step as a sample (Batch * Time, Hidden)
        # The paper suggests "understanding signal identity", which is a global property.
        # However, GW signals are transient.
        # Let's try pooling over time (Mean) to get a single vector per event.
        
        if z1.dim() == 3:
            z1 = z1.mean(dim=1) # (Batch, Hidden)
            z2 = z2.mean(dim=1) # (Batch, Hidden)
            
        # Normalize representations along the batch dimension
        # z: (N, D)
        N, D = z1.shape
        
        # BatchNorm-style normalization (zero mean, unit std)
        z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-6)
        z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-6)
        
        # Cross-Correlation Matrix
        # c: (D, D)
        c = torch.mm(z1_norm.T, z2_norm) / N
        
        # Loss
        # 1. Invariance term (diagonal should be 1)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        
        # 2. Redundancy reduction term (off-diagonal should be 0)
        off_diag = c.flatten()[:-1].view(D-1, D+1)[:, 1:].flatten()
        off_diag = off_diag.pow(2).sum()
        
        loss = on_diag + lambda_coeff * off_diag
        
        metrics = {
            "on_diag": on_diag.item(),
            "off_diag": off_diag.item()
        }
        
        return loss, metrics
