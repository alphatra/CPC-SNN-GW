import torch
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
import wandb
import time
from src.models.cpc_snn import CPCSNN
from src.data_handling.torch_dataset import HDF5SFTPairDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from src.evaluation.metrics import compute_tpr_at_fpr, compute_ece, compute_brier_score

def compute_effective_rank(z):
    """
    Compute effective rank of features.
    z: (B, T, H)
    """
    if z.dim() == 3:
        B, T, H = z.shape
        z_flat = z.reshape(-1, H)
    else:
        z_flat = z
        
    z_flat = z_flat - z_flat.mean(dim=0)
    cov = (z_flat.T @ z_flat).float()
    
    # Try/except for eigvalsh stability
    try:
        eigs = torch.linalg.eigvalsh(cov.cpu()).to(cov.device)
    except:
        return 1.0
        
    eigs = eigs[eigs > 1e-6]
    if len(eigs) == 0:
        return 1.0
        
    p = eigs / eigs.sum()
    entropy = -torch.sum(p * torch.log(p + 1e-9))
    return torch.exp(entropy).item()

def compute_infonce_loss(preds, z_targets, prediction_steps, device, temperature=0.1):
    """
    Computes InfoNCE Loss using Batch Negatives.
    preds: List of K tensors, each (B, T, H) - Predictions W_k(c_t)
    z_targets: (B, T, H) - True latents
    """
    total_loss = 0.0
    B, T, H = z_targets.shape
    
    # Normalize targets once
    z_targets_norm = torch.nn.functional.normalize(z_targets, dim=-1) # (B, T, H)
    
    valid_steps = 0
    for k in range(1, prediction_steps + 1):
        # We want P_{t}^{k} to match Z_{t+k}
        
        if T - k <= 0: continue
            
        # Slice valid time windows
        # P: pred at t, uses c_t. Valid t: 0 to T-1-k
        p_slice = preds[k-1][:, :-k, :] # (B, T-k, H)
        
        # Z: target at t+k. Valid t: k to T-1
        z_slice = z_targets_norm[:, k:, :] # (B, T-k, H)
        
        # Flatten (B * (T-k), H) -> N samples
        # Each sample is a positive pair (P_i, Z_i)
        # All other Z_j are negatives for P_i
        N = p_slice.shape[0] * p_slice.shape[1]
        
        queries = torch.nn.functional.normalize(p_slice, dim=-1).reshape(N, H)
        keys = z_slice.reshape(N, H)
        
        # Matmul for similarity: (N, N)
        logits = torch.matmul(queries, keys.T) / temperature
        
        # Targets are diagonal elements (0..N-1)
        labels = torch.arange(N).to(device)
        
        loss_k = torch.nn.functional.cross_entropy(logits, labels)
        total_loss += loss_k
        valid_steps += 1
        
    return total_loss / max(valid_steps, 1)

def compute_focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """
    Focal Loss for binary classification (Gamma focusing + Alpha balancing).
    Alpha = Weight for Class 1. If alpha=0.25, Cls 1 has 0.25, Cls 0 has 0.75.
    Helps when FP are expensive (we want to penalize noise/background errors more).
    """
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce_loss) # Proba of true class
    
    # Alpha weighting
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    
    focal_loss = alpha_t * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()

def compute_tail_penalty(logits, targets, threshold=0.95):
    """
    Penalizes noise samples (target=0) that have high probability (> threshold).
    """
    probs = torch.sigmoid(logits)
    noise_mask = (targets == 0).float()
    # Penalty only for noise > threshold
    # Squared ReLU to aggressively penalize large violations
    penalty = (torch.relu(probs - threshold) ** 2) * noise_mask
    return penalty.mean()

def train_model(args, train_loader, val_loader, device, use_spikes, use_timeseries, profiler=None):
    
    # 1. W&B Init
    if not args.no_wandb:
        wandb_id = args.resume_id if args.resume_id else wandb.util.generate_id()
        wandb.init(project="cpc-snn-gw", config=args, name=args.run_name, id=wandb_id, resume="allow")
        
    # 2. Model Init
    in_channels = 1 if args.channel else 2
    
    # TF2D Setup
    if args.use_tf2d:
        print("TF2D Mode: Using 2D Conv Encoder on STFT.")
        comps = 3 if args.no_mask else 4 # mag, cos, sin, [MASK]
        in_channels = comps * (1 if args.channel else 2) 
        print(f"TF2D: In_channels={in_channels} (Mask={'OFF' if args.no_mask else 'ON'})")
    
    # SFT Baseline Setup
    elif args.use_sft:
        # Determine F_bins from one batch
        print("SFT Baseline: Determining input shape...")
        temp_batch = next(iter(train_loader))
        f_bins = temp_batch['f'].shape[1]
        # (mag, cos, sin) * (H1, L1) -> 3 * 2 = 6 components per freq
        # Total channels = 6 * F_bins
        # Or if single channel, 3 * F_bins
        comps = 3 # mag, cos, sin
        in_channels = comps * f_bins * (1 if args.channel else 2)
        print(f"SFT Baseline: Using raw SFT input. In_channels={in_channels} (F_bins={f_bins})")
    
    model = CPCSNN(
        in_channels=in_channels,
        hidden_dim=args.hidden_dim,
        context_dim=args.context_dim,
        prediction_steps=args.prediction_steps,
        delta_threshold=args.delta_threshold, 
        temperature=args.temperature,
        beta=args.beta,
        use_checkpointing=args.checkpointing,
        use_metal=not args.no_metal,
        use_continuous_input=not args.use_discrete_input,
        no_dain=args.no_dain,
        use_tf2d=args.use_tf2d
    ).to(device) # Removed memory_format=torch.channels_last (Fix #4)

    # Fix for LazyLinear (TF2D): Initialize parameters before wandb.watch
    if args.use_tf2d:
        print("Initializing LazyLinear layers with dummy forward pass...")
        # (B, C, T, F) -> T=32, F=256 is safe enough for init, exact F doesn't matter for init logic as long as it runs
        # But we want to be close to real F.
        # Let's use 256 as generic.
        was_training = model.training
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(2, in_channels, 32, 256).to(device)
            _ = model(dummy_input)
        model.train(was_training)

    # Freezing Logic for Finetuning
    if args.train_mode == 'finetune' and args.freeze_encoder:
        print("Freezing Encoder and Feature Extractor...")
        for p in model.encoder.parameters(): p.requires_grad = False
        for p in model.feature_extractor.parameters(): p.requires_grad = False
        if hasattr(model, 'tf_encoder'):
             for p in model.tf_encoder.parameters(): p.requires_grad = False
        if hasattr(model, 'adapter'):
             for p in model.adapter.parameters(): p.requires_grad = False
        if hasattr(model, 'dain'):
             for p in model.dain.parameters(): p.requires_grad = False
        
        # Verify
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters after freezing: {trainable_params}")

    if not args.no_wandb:
        # Reduced overhead: log gradients only, every 100 steps
        wandb.watch(model, log="gradients", log_freq=100)

    # 3. Optimizer & Scheduler
    learning_rate = args.lr
    rsnn_params = list(map(id, model.context_network.parameters())) if hasattr(model, 'context_network') else []
    base_params = filter(lambda p: id(p) not in rsnn_params, model.parameters())
    
    params = [
        {'params': base_params},
        {'params': model.context_network.parameters(), 'lr': learning_rate * 0.1} # 10x smaller
    ]
    
    if args.weight_decay > 0:
        optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(params, lr=learning_rate)
        
    # Scheduler Setup
    max_lr_base = learning_rate * 10
    max_lr_rsnn = max_lr_base * 0.1
    
    if args.stability_test:
        print("Using Constant LR (no OneCycleLR)")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs, gamma=1.0)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[max_lr_base, max_lr_rsnn],
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4
        )

    # 4. Checkpoint Loading
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Fix #6: Resume Scheduler
            if 'scheduler_state_dict' in checkpoint:
                 try:
                     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                     # For OneCycleLR, step count matters. 
                     # If we are strictly resuming, we might need to verify steps.
                 except Exception as e:
                     print(f"Warning: Failed to load scheduler state: {e}")
            
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
            
    # 5. AMP Scaler (Fix #2)
    use_amp = args.amp
    scaler = None
    
    if use_amp:
        if device.type == "cuda":
            # CUDA: Autocast + GradScaler
            scaler = torch.cuda.amp.GradScaler()
        elif device.type == "mps":
            # MPS: Autocast allowed, but GradScaler often unstable/no-op on MPS.
            # Disable Scaler for MPS (pseudo-AMP or bf16-only)
            print("AMP on MPS: Disabling GradScaler (running safe bf16/fp32 mixed).")
            scaler = None
        else:
             # CPU or other: Disable AMP generally or use bf16 without scaler
             pass
        
    print(f"Starting training for {args.epochs} epochs (starting from {start_epoch})...")
    print(f"AMP Enabled: {use_amp}")
    
    best_val_loss = float('inf')
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("checkpoints", args.run_name or timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # --- TRAINING LOOP ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            t0 = time.time()
            
            # --- Label Integrity Diagnostic ---
            if batch_idx == 0:
                if 'id' in batch:
                     ids = batch['id']
                     # Labels are usually in batch['label']
                     lbls = batch['label'] if 'label' in batch else torch.zeros(len(ids))
                     print(f"\n[Integrity] Ep {epoch+1} Batch 0 Sample:")
                     for i in range(min(3, len(ids))):
                         print(f"  ID: {ids[i]} | Label: {lbls[i].item()}")
                else:
                     print("\n[Integrity] 'id' key missing in batch.")
            # ----------------------------------
            
            # Data Prep
            if use_spikes or use_timeseries:
                x = batch['x'].to(device, non_blocking=True)
                if not use_spikes:
                     x = x.to(memory_format=torch.channels_last)
            
            elif args.use_tf2d:
                # TF2D Input: Stack Mag, Cos, Sin -> (B, 6, T, F)
                feat_list = []
                def get_tf_feats(ifo):
                     d = batch[ifo].to(device) # (B, 4, T, F)
                     slice_idx = 3 if args.no_mask else 4
                     return d[:, 0:slice_idx, :, :]
                
                if args.channel != "L1": feat_list.append(get_tf_feats("H1"))
                if args.channel != "H1": feat_list.append(get_tf_feats("L1"))
                
                x = torch.cat(feat_list, dim=1) # (B, 6, T, F)
                
                # Normalize (Instance Norm per channel over T,F)
                mean = x.mean(dim=(2,3), keepdim=True)
                std = x.std(dim=(2,3), keepdim=True)
                x = (x - mean) / (std + 1e-8)

            elif args.use_sft:
                # SFT Baseline: Stack Mag, Cos, Sin
                # H1: (B, 4, T, F) -> Extract 0,1,2
                feat_list = []
                
                # Helper to process IFO
                def process_ifo(ifo_name):
                    d = batch[ifo_name].to(device) # (B, 4, T, F)
                    # Take mag, cos, sin, [mask]
                    slice_idx = 3 if args.no_mask else 4
                    feats = d[:, 0:slice_idx, :, :]
                    # Permute to (B, 3, F, T) -> Flatten (B, 3*F, T)
                    feats = feats.permute(0, 1, 3, 2).reshape(d.shape[0], -1, d.shape[2])
                    return feats

                if args.channel != "L1": feat_list.append(process_ifo("H1"))
                if args.channel != "H1": feat_list.append(process_ifo("L1"))
                
                x = torch.cat(feat_list, dim=1) # (B, C_flat, T)
                
                # Normalize (Instance Norm per channel)
                mean = x.mean(dim=2, keepdim=True)
                std = x.std(dim=2, keepdim=True)
                x = (x - mean) / (std + 1e-8)
                
            else:
                # GPU Reconstruction
                x = HDF5SFTPairDataset.batch_reconstruct_torch(batch, device=device)
                
                # Validation Logging (First batch of epoch)
                if batch_idx == 0:
                     print(f"\n[Epoch {epoch+1} Batch 0 Stats (Pre-Norm)]")
                     # H1
                     h1 = x[:, 0, :]
                     print(f"H1: Mean={h1.mean().item():.4f}, Std={h1.std().item():.4f}, Max={h1.abs().max().item():.4f}")
                     if x.shape[1] > 1:
                         l1 = x[:, 1, :]
                         print(f"L1: Mean={l1.mean().item():.4f}, Std={l1.std().item():.4f}, Max={l1.abs().max().item():.4f}")
                     print("-" * 30)

                # Normalize (Instance Norm)
                mean = x.mean(dim=2, keepdim=True)
                std = x.std(dim=2, keepdim=True)
                x = (x - mean) / (std + 1e-8)
            
            if not args.use_tf2d:
                if args.channel == "H1" and not args.use_sft: x = x[:, 0:1, :]
                elif args.channel == "L1" and not args.use_sft: x = x[:, 1:2, :]
            
            t_data = time.time() - t0
            t1 = time.time()
            
            # Augmentation
            if model.training and not args.no_aug:
                # 1. Time Shift (Always active unless no_aug)
                if args.use_tf2d:
                    # x is (B, C, T, F). Shift along T (dim=2)
                    max_shift = int(0.1 * x.shape[2]) 
                    shift = int(torch.randint(-max_shift, max_shift, (1,)).item())
                    x = torch.roll(x, shifts=shift, dims=2)
                    
                    # 2. SpecAugment - Frequency Masking
                    if args.aug_freq_mask > 0 and torch.rand(1).item() < args.aug_prob:
                        # x: (B, C, T, F)
                        F = x.shape[3]
                        f_width = torch.randint(1, args.aug_freq_mask + 1, (1,)).item()
                        f0 = torch.randint(0, F - f_width + 1, (1,)).item()
                        x[:, :, :, f0:f0+f_width] = 0.0

                else:
                    # x is (B, C, T). Shift along T (dim=-1)
                    max_shift = int(0.1 * x.shape[-1])
                    shift = int(torch.randint(-max_shift, max_shift, (1,)).item())
                    x = torch.roll(x, shifts=shift, dims=-1)
                
                # 3. Amplitude Jitter (TF2D or 1D)
                if args.aug_amp > 0.0 and torch.rand(1).item() < args.aug_prob:
                    # Scale factor: 1.0 + uniform(-amp, amp)
                    scale = 1.0 + (torch.rand(x.size(0), 1, 1, 1, device=device) * 2 - 1) * args.aug_amp
                    if not args.use_tf2d: scale = scale.squeeze(-1) # Handle 1D
                    x = x * scale
            
            # Label
            y = batch['label'].to(device).float().view(-1, 1)
            
            # Sanity Check: Permute Labels in Batch
            if args.sanity_permute_labels:
                y = y[torch.randperm(y.size(0))]
            
            optimizer.zero_grad()
            
            # Forward & Loss
            if use_amp:
                # MPS supports bfloat16 (mostly) or float16
                dtype = torch.bfloat16 if (device.type == 'mps' or device.type == 'cpu') else torch.float16
                with torch.autocast(device_type=device.type, dtype=dtype):
                    logits, c, z = model(x, is_encoded=use_spikes)
                    
                    if args.train_mode == 'pretrain_cpc':
                        cpc_preds = model.predict_future(c)
                        loss = compute_infonce_loss(cpc_preds, z, args.prediction_steps, device, args.temperature)
                        acc = 0.0
                    else:
                        if args.loss_type == 'focal':
                            loss = compute_focal_loss(logits, y, args.focal_gamma)
                        else:
                            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

                        # Tail Penalty
                        if args.tail_penalty > 0:
                            loss = loss + args.tail_penalty * compute_tail_penalty(logits, y, args.tail_threshold)
                        
                        # Multi-Task CPC (Option B)
                        if args.lambda_infonce > 0:
                            cpc_preds = model.predict_future(c)
                            loss_cpc = compute_infonce_loss(cpc_preds, z, args.prediction_steps, device, args.temperature)
                            loss = loss + args.lambda_infonce * loss_cpc
                        
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).float()
                        acc = (preds == y).float().mean().item()
            
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if args.grad_clip > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    else:
                        grad_norm = torch.tensor(0.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # AMP without Scaler (e.g. BF16 on MPS)
                    loss.backward()
                    if args.grad_clip > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    else:
                        grad_norm = torch.tensor(0.0)
                    optimizer.step()
                
                scheduler.step()
            else:
                logits, c, z = model(x, is_encoded=use_spikes)
                
                if args.train_mode == 'pretrain_cpc':
                    cpc_preds = model.predict_future(c)
                    loss = compute_infonce_loss(cpc_preds, z, args.prediction_steps, device, args.temperature)
                    acc = 0.0
                else: 
                    if args.loss_type == 'focal':
                        loss = compute_focal_loss(logits, y, args.focal_gamma)
                    else:
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

                    # Tail Penalty
                    if args.tail_penalty > 0:
                        loss = loss + args.tail_penalty * compute_tail_penalty(logits, y, args.tail_threshold)
                    
                    # Multi-Task CPC (Option B)
                    if args.lambda_infonce > 0:
                        cpc_preds = model.predict_future(c)
                        loss_cpc = compute_infonce_loss(cpc_preds, z, args.prediction_steps, device, args.temperature)
                        loss = loss + args.lambda_infonce * loss_cpc
                        
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    acc = (preds == y).float().mean().item()
                
                loss.backward()
                if args.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                else:
                    grad_norm = torch.tensor(0.0)
                optimizer.step()
                scheduler.step()
                
            t_model = time.time() - t1
            train_loss += loss.item()
            train_acc += acc
            
            pbar.set_postfix({'loss': loss.item(), 'acc': acc})
            
            if not args.no_wandb:
                wandb.log({
                    "batch_loss": loss.item(), 
                    "batch_acc": acc,
                    "grad_norm": grad_norm,
                    "lr": optimizer.param_groups[0]['lr'],
                    "rsnn_context_mean": c.mean().item(),
                    "rsnn_context_std": c.std().item(),
                    "input_rms": x.pow(2).mean().sqrt().item(),
                    "data_load_time": t_data,
                    "effective_rank": compute_effective_rank(z)
                })
            
            if profiler: profiler.step()
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        val_acc = 0
        all_val_logits = []
        all_val_labels = []
        all_val_z = []
        all_val_c = []
        
        with torch.no_grad():
            for batch in val_loader:
                if use_spikes or use_timeseries:
                    x = batch['x'].to(device)
                elif args.use_tf2d:
                     # TF2D Input: Stack Mag, Cos, Sin -> (B, 6, T, F)
                     feat_list = []
                     def get_tf_feats(ifo):
                          d = batch[ifo].to(device) # (B, 4, T, F)
                          slice_idx = 3 if args.no_mask else 4
                          return d[:, 0:slice_idx, :, :]
                     
                     if args.channel != "L1": feat_list.append(get_tf_feats("H1"))
                     if args.channel != "H1": feat_list.append(get_tf_feats("L1"))
                     
                     x = torch.cat(feat_list, dim=1) # (B, 6, T, F)
                     
                     # Normalize (Instance Norm per channel over T,F)
                     mean = x.mean(dim=(2,3), keepdim=True)
                     std = x.std(dim=(2,3), keepdim=True)
                     x = (x - mean) / (std + 1e-8)

                elif args.use_sft:
                     # SFT Baseline: Stack Mag, Cos, Sin
                     # H1: (B, 4, T, F) -> Extract 0,1,2
                     feat_list = []
                     
                     # Helper to process IFO
                     def process_ifo(ifo_name):
                         d = batch[ifo_name].to(device) 
                         feats = d[:, 0:3, :, :]
                         feats = feats.permute(0, 1, 3, 2).reshape(d.shape[0], -1, d.shape[2])
                         return feats
 
                     if args.channel != "L1": feat_list.append(process_ifo("H1"))
                     if args.channel != "H1": feat_list.append(process_ifo("L1"))
                     x = torch.cat(feat_list, dim=1)
                     # Norm
                     mean = x.mean(dim=2, keepdim=True)
                     std = x.std(dim=2, keepdim=True)
                     x = (x - mean) / (std + 1e-8)
                else:
                    x = HDF5SFTPairDataset.batch_reconstruct_torch(batch, device=device)
                    # Normalize
                    mean = x.mean(dim=2, keepdim=True)
                    std = x.std(dim=2, keepdim=True)
                    x = (x - mean) / (std + 1e-8)
                    
                if not args.use_tf2d:
                    if args.channel == "H1" and not args.use_sft: x = x[:, 0:1, :]
                    elif args.channel == "L1" and not args.use_sft: x = x[:, 1:2, :]
                
                y = batch['label'].to(device).float().view(-1, 1)
                logits, c, z = model(x, is_encoded=use_spikes)
                
                if args.train_mode == 'pretrain_cpc':
                     cpc_preds = model.predict_future(c)
                     loss = compute_infonce_loss(cpc_preds, z, args.prediction_steps, device, args.temperature)
                     val_loss += loss.item()
                     # acc is 0
                     preds = torch.zeros_like(y)
                else:
                    if args.loss_type == 'focal':
                        loss = compute_focal_loss(logits, y, args.focal_gamma)
                    else:
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

                    # Tail Penalty
                    if args.tail_penalty > 0:
                        loss = loss + args.tail_penalty * compute_tail_penalty(logits, y, args.tail_threshold)
                    
                    if args.lambda_infonce > 0:
                        cpc_preds = model.predict_future(c)
                        loss_cpc = compute_infonce_loss(cpc_preds, z, args.prediction_steps, device, args.temperature)
                        loss = loss + args.lambda_infonce * loss_cpc
                        
                    val_loss += loss.item()
                    
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    val_acc += (preds == y).float().mean().item()
                
                all_val_logits.append(logits.detach().cpu())
                all_val_labels.append(y.cpu())
                all_val_z.append(z.detach().cpu())
                all_val_c.append(c[:, -1, :].detach().cpu()) # Store final context
                
                # Visualize Input Waveforms (Once per epoch, first batch)
                if not args.no_wandb and len(all_val_logits) == 1 and not args.use_tf2d:
                    # Pick 1 signal and 1 noise if available
                    sig_idx = (y == 1).nonzero(as_tuple=True)[0]
                    noise_idx = (y == 0).nonzero(as_tuple=True)[0]
                    
                    import matplotlib.pyplot as plt
                    
                    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
                    # Plot Signal
                    if len(sig_idx) > 0:
                        idx = sig_idx[0].item()
                        t_axis = np.arange(x.shape[-1])
                        axes[0].plot(t_axis, x[idx, 0].cpu().numpy(), label="H1", alpha=0.8)
                        if x.shape[1] > 1:
                            axes[0].plot(t_axis, x[idx, 1].cpu().numpy(), label="L1", alpha=0.8)
                        axes[0].set_title(f"Input Strain: SIGNAL (Ep {epoch+1})")
                        axes[0].legend()
                        
                    # Plot Noise
                    if len(noise_idx) > 0:
                        idx = noise_idx[0].item()
                        t_axis = np.arange(x.shape[-1])
                        axes[1].plot(t_axis, x[idx, 0].cpu().numpy(), label="H1", alpha=0.8)
                        if x.shape[1] > 1:
                            axes[1].plot(t_axis, x[idx, 1].cpu().numpy(), label="L1", alpha=0.8)
                        axes[1].set_title(f"Input Strain: NOISE (Ep {epoch+1})")
                        axes[1].legend()
                        
                    plt.tight_layout()
                    wandb.log({"val_input_waveforms": wandb.Image(fig)})
                    plt.close(fig)
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        
        # Metrics
        all_val_logits = torch.cat(all_val_logits, dim=0)
        all_val_labels = torch.cat(all_val_labels, dim=0)
        all_val_z = torch.cat(all_val_z, dim=0)
        all_val_c = torch.cat(all_val_c, dim=0)
        
        val_probs = torch.sigmoid(all_val_logits).numpy().flatten()
        val_labels_np = all_val_labels.numpy().flatten()
        
        # --- DIAGNOSTICS & METRICS ---
        from sklearn.metrics import (
            confusion_matrix, accuracy_score, matthews_corrcoef, f1_score, 
            precision_score, recall_score, balanced_accuracy_score, log_loss
        )
        # Import custom metrics
        from src.evaluation.metrics import compute_tpr_at_fpr, compute_stability_metrics
        
        # 1. Ranking Metrics
        if args.train_mode != 'pretrain_cpc':
            try:
                val_roc_auc = roc_auc_score(val_labels_np, val_probs)
                val_pr_auc = average_precision_score(val_labels_np, val_probs)
                
                # Partial AUC (fpr <= 1e-4) - tighter focus
                try:
                    val_pauc_1e4 = roc_auc_score(val_labels_np, val_probs, max_fpr=1e-4)
                except:
                    val_pauc_1e4 = 0.0 # Fallback
                
                # TPR @ various FPRs
                current_fprs = [1e-3, 1e-4, 1e-5, 1e-6]
                t_metrics = compute_tpr_at_fpr(val_labels_np, val_probs, fpr_thresholds=current_fprs)
                
                val_tpr_1e3 = t_metrics[1e-3]['tpr']
                val_tpr_1e4 = t_metrics[1e-4]['tpr']
                val_fnr_1e4 = t_metrics[1e-4]['fnr']
                val_thr_1e4 = t_metrics[1e-4]['threshold']
                val_tpr_1e5 = t_metrics[1e-5]['tpr']
                val_tpr_1e6 = t_metrics[1e-6]['tpr']

            except Exception as e:
                print(f"Ranking Metrics Error: {e}")
                val_roc_auc = val_pr_auc = val_pauc_1e4 = 0.0
                val_tpr_1e3 = val_tpr_1e4 = val_fnr_1e4 = 0.0
                val_tpr_1e5 = val_tpr_1e6 = val_thr_1e4 = 0.0

            # 2. Stability Metrics (Noise Tails)
            try:
                stab_metrics = compute_stability_metrics(val_labels_np, val_probs)
                val_noise_max = stab_metrics['noise_max']
                val_noise_q999 = stab_metrics['noise_q99.9']
                val_d_prime = stab_metrics['d_prime']
                val_ks_stat = stab_metrics['ks_stat']
            except Exception as e:
                print(f"Stability Metrics Error: {e}")
                val_noise_max = val_noise_q999 = val_d_prime = val_ks_stat = 0.0
        else:
             # Pretrain Defaults
             val_roc_auc = val_pr_auc = val_pauc_1e4 = 0.0
             val_tpr_1e3 = val_tpr_1e4 = val_fnr_1e4 = 0.0
             val_tpr_1e5 = val_tpr_1e6 = val_thr_1e4 = 0.0
             val_noise_max = val_noise_q999 = val_d_prime = val_ks_stat = 0.0

        # 3. Calibration Metrics
        # 3. Calibration Metrics
        if args.train_mode != 'pretrain_cpc':
            try:
                val_ece = compute_ece(val_labels_np, val_probs)
                val_brier = compute_brier_score(val_labels_np, val_probs)
                val_nll = log_loss(val_labels_np, val_probs)
                
                # MCE (Max Calibration Error)
                def compute_mce_simple(y_true, y_prob, n_bins=10):
                    bins = np.linspace(0.0, 1.0, n_bins + 1)
                    binids = np.digitize(y_prob, bins) - 1
                    max_err = 0.0
                    for i in range(n_bins):
                        mask = binids == i
                        if np.any(mask):
                            prob_mean = y_prob[mask].mean()
                            true_mean = y_true[mask].mean()
                            err = np.abs(prob_mean - true_mean)
                            if err > max_err: max_err = err
                    return max_err
                val_mce = compute_mce_simple(val_labels_np, val_probs)
                
            except Exception as e:
                print(f"Calibration Metrics Error: {e}")
                val_ece = val_brier = val_nll = val_mce = 0.0

            # 4. Threshold Metrics (@0.5 and @FPR=1e-4)
            pred05 = (val_probs > 0.5).astype(int)
            cm05 = confusion_matrix(val_labels_np, pred05, labels=[0, 1])
            
            # Metrics @ 0.5
            acc05 = accuracy_score(val_labels_np, pred05)
            mcc05 = matthews_corrcoef(val_labels_np, pred05)
            f1_05 = f1_score(val_labels_np, pred05)
            bal_acc05 = balanced_accuracy_score(val_labels_np, pred05)
            
            # Metrics @ FPR=1e-4 Threshold
            thr_target = val_thr_1e4 if val_thr_1e4 > 0 else 0.9 
            pred_target = (val_probs > thr_target).astype(int)
            mcc_target = matthews_corrcoef(val_labels_np, pred_target)
            f1_target = f1_score(val_labels_np, pred_target)
            cm_target = confusion_matrix(val_labels_np, pred_target, labels=[0, 1])
        else:
            val_ece = val_brier = val_nll = val_mce = 0.0
            acc05 = mcc05 = f1_05 = bal_acc05 = 0.0
            mcc_target = f1_target = 0.0
            cm05 = cm_target = np.zeros((2,2))
            val_thr_1e4 = 0.9
            thr_target = 0.9

        # 5. Representation Metrics
        val_eff_rank_z = compute_effective_rank(all_val_z)
        val_eff_rank_c = compute_effective_rank(all_val_c)
        
        norm_z = torch.norm(all_val_z, dim=-1)
        norm_c = torch.norm(all_val_c, dim=-1)
        z_norm_mean, z_norm_std = norm_z.mean().item(), norm_z.std().item()
        c_norm_mean, c_norm_std = norm_c.mean().item(), norm_c.std().item()

        # Logits & Probs Stats
        logits_mean, logits_std = all_val_logits.mean().item(), all_val_logits.std().item()
        probs_mean, probs_std = val_probs.mean(), val_probs.std()
        probs_min, probs_max = val_probs.min(), val_probs.max()

        # Print Diagnostics
        # Print Diagnostics
        print("\n--- ADVANCED DIAGNOSTICS ---")
        if args.train_mode == 'pretrain_cpc':
             print(f"Mode: CPC PRETRAIN (InfoNCE Loss)")
             print(f"Val Loss (InfoNCE): {avg_val_loss:.4f}")
             print(f"Rank Z: {val_eff_rank_z:.2f} (Norm: {z_norm_mean:.2f})")
             print(f"Rank C: {val_eff_rank_c:.2f} (Norm: {c_norm_mean:.2f})")
        else:
            print(f"AUC: {val_roc_auc:.5f} | pAUC(1e-4): {val_pauc_1e4:.5f}")
            print(f"TPR @1e-4: {val_tpr_1e4:.4f} | FNR: {val_fnr_1e4:.4f}")
            print(f"Efficiency @1e-6: {val_tpr_1e6:.4f}")
            print(f"Stability: Noise Max={val_noise_max:.4f}, KS={val_ks_stat:.4f}, d'={val_d_prime:.2f}")
            print(f"Distributions: Probs Mean={probs_mean:.4f} (Min={probs_min:.4f}, Max={probs_max:.4f}) | Logits Mean={logits_mean:.4f}")
            
            print(f"CM @0.5:\n{cm05}")
            print(f"ACC @0.5: {acc05:.4f} | MCC: {mcc05:.4f}")
            
            print(f"CM @FPR=1e-4 (Thr={thr_target:.6f}):\n{cm_target}")
            print(f"MCC @Target: {mcc_target:.4f}")
            
            print(f"Rank Z: {val_eff_rank_z:.2f} (Norm: {z_norm_mean:.2f})")
            print(f"Rank C: {val_eff_rank_c:.2f} (Norm: {c_norm_mean:.2f})")
            
        print("----------------------------\n")
        
        print(f"Epoch {epoch+1}: Train Los={avg_train_loss:.4f} | Val Los={avg_val_loss:.4f} | AUC={val_roc_auc:.4f} | RankZ={val_eff_rank_z:.2f}")
        
        if not args.no_wandb:
            wandb.log({
                "epoch": epoch + 1,
                
                # Main
                "train_loss": avg_train_loss,
                "train_acc": avg_train_acc,
                "val_loss": avg_val_loss,
                
                # Ranking (Low FPR focus)
                "val_auc": val_roc_auc,
                "val_pr_auc": val_pr_auc,
                "val_pauc_1e4": val_pauc_1e4,
                
                "val_tpr@fpr=1e-3": val_tpr_1e3,
                "val_tpr@fpr=1e-4": val_tpr_1e4,
                "val_tpr@fpr=1e-5": val_tpr_1e5,
                "val_tpr@fpr=1e-6": val_tpr_1e6,
                "val_fnr@fpr=1e-4": val_fnr_1e4,
                "val_thr@fpr=1e-4": val_thr_1e4,

                # Stability & Tails
                "val_noise_max": val_noise_max,
                "val_noise_q99.9": val_noise_q999,
                "val_d_prime": val_d_prime,
                "val_ks_stat": val_ks_stat,

                # Calibration
                "val_ece": val_ece,
                "val_mce": val_mce,
                "val_nll": val_nll,
                "val_brier": val_brier,
                
                # Threshold @ 0.5
                "val_acc_05": acc05,
                "val_mcc_05": mcc05,
                "val_f1_05": f1_05,
                "val_bal_acc_05": bal_acc05,
                
                # Threshold @ FPR=1e-4
                "val_mcc_tgt": mcc_target,
                "val_f1_tgt": f1_target,
                
                # Representation
                "val_rank_z": val_eff_rank_z,
                "val_rank_c": val_eff_rank_c,
                "val_norm_z": z_norm_mean,
                "val_norm_c": c_norm_mean,
                "val_std_c": c_norm_std,
                
                # Distribution Stats
                "val_logits_mean": logits_mean,
                "val_logits_std": logits_std,
                "val_probs_mean": probs_mean,
                "val_probs_std": probs_std,
                
                # GW Specific Plots
                "val_scores_signal": wandb.Histogram(val_probs[val_labels_np == 1]),
                "val_scores_noise": wandb.Histogram(val_probs[val_labels_np == 0]),
                
                # Confusion Matrix at 0.5
                "val_conf_mat": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=val_labels_np,
                    preds=pred05,
                    class_names=["Noise", "Signal"]
                )
            })
            
        # Checkpointing
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), # Fix #6
            'loss': avg_val_loss,
            'config': vars(args)
        }
        torch.save(checkpoint, os.path.join(save_dir, "latest.pt"))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(save_dir, "best.pt"))
            print(f"New best model saved to {save_dir}/best.pt")

    print("Training complete.")
    if not args.no_wandb:
        wandb.finish()
