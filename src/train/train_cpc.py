import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import argparse
import numpy as np
from tqdm import tqdm
import wandb
import time

from src.models.cpc_snn import CPCSNN
from src.data_handling.torch_dataset import HDF5SFTPairDataset

def train(args):
    # 1. W&B Init
    # 1. W&B Init
    if not args.no_wandb:
        wandb_id = args.resume_id if args.resume_id else wandb.util.generate_id()
        wandb.init(project="cpc-snn-gw", config=args, name=args.run_name, id=wandb_id, resume="allow")
    
    # 2. Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 3. Load Data
    with open(args.noise_indices, 'r') as f:
        noise_indices = json.load(f)
    
    if args.fast:
        print("FAST MODE: Using subset of data.")
        noise_indices = noise_indices[:100]
        args.epochs = 1
        
    # Check for pre-processed data (Spikes > TimeSeries > On-the-fly)
    spikes_path = args.h5_path.replace(".h5", "_spikes.h5")
    ts_path = args.h5_path.replace(".h5", "_timeseries.h5")
    
    use_spikes = os.path.exists(spikes_path)
    use_timeseries = os.path.exists(ts_path)
    
    if use_spikes:
        print(f"Using pre-encoded spikes from {spikes_path}")
        from src.data_handling.torch_dataset import HDF5TimeSeriesDataset
        dataset = HDF5TimeSeriesDataset(
            h5_path=spikes_path,
            index_list=noise_indices
        )
    elif use_timeseries:
        print(f"Using pre-processed time series data from {ts_path}")
        from src.data_handling.torch_dataset import HDF5TimeSeriesDataset
        dataset = HDF5TimeSeriesDataset(
            h5_path=ts_path,
            index_list=noise_indices
        )
    else:
        print("Using on-the-fly reconstruction (Slower)")
        dataset = HDF5SFTPairDataset(
            h5_path=args.h5_path,
            index_list=noise_indices,
            return_time_series=False
        )
    
    # Split
    # NOWE (DOBRE - Time Series Split):
    # Dataset indeksuje pliki po kolei, więc wystarczy podzielić indeksy
    indices = list(range(len(dataset)))
    split_idx = int(len(dataset) * 0.8) # 80% trening, 20% walidacja

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)

    print(f"Time Series Split: Train [{len(train_set)}] samples, Val [{len(val_set)}] samples")
    
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": False,
    }
    
    if args.workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    
    # 4. Model
    in_channels = 1 if args.channel else 2
    model = CPCSNN(
        in_channels=in_channels,
        hidden_dim=args.hidden_dim,
        context_dim=args.context_dim,
        prediction_steps=args.prediction_steps,
        delta_threshold=args.delta_threshold,
        temperature=args.temperature,
        beta=args.beta
    ).to(device)
    
    if not args.no_wandb:
        wandb.watch(model, log="all")
        
    # Optimizer
    # Reduced LR to 1e-4 as requested
    learning_rate = args.lr
    
    if args.weight_decay > 0:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler
    # Scheduler
    # Use OneCycleLR for better stability
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,           # Higher peak
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,         # 30% warmup
        div_factor=25,
        final_div_factor=1e4
    )
    
    
    # Resume Logic
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Note: OneCycleLR scheduler state might be tricky to resume perfectly if not saved/loaded carefully,
            # but usually loading state_dict works if the total steps match.
            # However, since we re-create scheduler based on args.epochs, we might need to adjust if resuming mid-run.
            # For now, we'll try to load it if present.
            if 'scheduler_state_dict' in checkpoint: # We didn't save it before, but good to have for future
                 # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                 pass
            
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    # 5. Training Loop
    best_val_loss = float('inf')
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("checkpoints", args.run_name or timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # AMP Scaler for MPS
    # Check if torch.amp.GradScaler exists (PyTorch 2.3+)
    use_amp = args.amp and hasattr(torch, 'amp')
    if use_amp and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('mps')
    elif use_amp:
        # Fallback for older PyTorch (might not work on MPS)
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    print(f"Starting training for {args.epochs} epochs (starting from {start_epoch})...")
    print(f"AMP Enabled: {use_amp}")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            t0 = time.time()
            
            if use_spikes or use_timeseries:
                x = batch['x'].to(device)
            else:
                # GPU Reconstruction
                x = HDF5SFTPairDataset.batch_reconstruct_torch(batch, device=device)
                
                # Normalize (Instance Norm)
                mean = x.mean(dim=2, keepdim=True)
                std = x.std(dim=2, keepdim=True)
                x = (x - mean) / (std + 1e-8)
            
            # Select channel if needed
            if args.channel == "H1":
                x = x[:, 0:1, :] # Keep dim: (B, 1, T)
            elif args.channel == "L1":
                x = x[:, 1:2, :] # Keep dim: (B, 1, T)
            
            # --- Time Jittering (Data Augmentation) ---
            if model.training:
                # Shift by +/- 5% of window length
                max_shift = int(x.shape[-1] * 0.05)
                if max_shift > 0:
                    shift = np.random.randint(-max_shift, max_shift)
                    x = torch.roll(x, shifts=shift, dims=-1)
            # ------------------------------------------
            
            t_data = time.time() - t0
            t1 = time.time()
            
            optimizer.zero_grad()
            
            # Mixed Precision Forward
            if use_amp:
                with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                    z, c, spikes = model(x, is_encoded=use_spikes)
                    # Regularize z (SNN output) instead of encoder spikes
                    loss, metrics = model.compute_cpc_loss(z, c, spikes=z)
                    acc = metrics["acc1"]
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                z, c, spikes = model(x, is_encoded=use_spikes)
                # Regularize z (SNN output) instead of encoder spikes
                loss, metrics = model.compute_cpc_loss(z, c, spikes=z)
                acc = metrics["acc1"]
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
            
            t_model = time.time() - t1
            batch_time = time.time() - t0
            
            train_loss += loss.item()
            train_acc += acc
            
            pbar.set_postfix({'loss': loss.item(), 'acc': acc, 't_data': f"{t_data:.2f}s", 't_model': f"{t_model:.2f}s"})
            
            if not args.no_wandb:
                wandb.log({
                    "batch_loss": loss.item(), 
                    "batch_acc": acc,
                    "train_acc_top5": metrics["acc5"],
                    "cpc_pos_score": metrics["pos_score"],
                    "cpc_neg_score": metrics["neg_score"],
                    "cpc_margin": metrics["pos_score"] - metrics["neg_score"],
                    "latent_diversity": z.std(dim=0).mean().item(),
                    "grad_norm": grad_norm,
                    "lr": optimizer.param_groups[0]['lr'],
                    "batch_time": batch_time,
                    "time_data_load": t_data,
                    "time_model_fwd_bwd": t_model,
                    "snn_spike_density": z.mean().item(),
                    "rsnn_context_mean": c.mean().item(),
                    "rsnn_context_std": c.std().item(),
                    "input_rms": x.std().item()
                })
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for batch in val_loader:
                if use_spikes or use_timeseries:
                    x = batch['x'].to(device)
                else:
                    x = HDF5SFTPairDataset.batch_reconstruct_torch(batch, device=device)
                    # Normalize
                    mean = x.mean(dim=2, keepdim=True)
                    std = x.std(dim=2, keepdim=True)
                    x = (x - mean) / (std + 1e-8)
                    
                if args.channel == "H1":
                    x = x[:, 0:1, :]
                elif args.channel == "L1":
                    x = x[:, 1:2, :]
                    
                z, c, spikes = model(x, is_encoded=use_spikes)
                loss, metrics = model.compute_cpc_loss(z, c, spikes=z)
                acc = metrics["acc1"]
                val_loss += loss.item()
                val_acc += acc
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")
        
        if not args.no_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_acc": avg_train_acc,
                "val_loss": avg_val_loss,
                "val_acc": avg_val_acc
            })
            
        # Save Checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
            'config': vars(args)
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(save_dir, "latest.pt"))
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(save_dir, "best.pt"))
            print(f"New best model saved to {save_dir}/best.pt")
        
        # Step Scheduler
        # scheduler.step() # Moved to batch loop for OneCycleLR

    print("Training complete.")
    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CPC-SNN on Background Noise")
    
    # Resolve paths relative to project root
    # script_dir = src/train
    # project_root = src/train/../..
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    default_h5 = os.path.join(project_root, "data/cpc_snn_train.h5")
    default_noise = os.path.join(project_root, "data/indices_noise.json")
    
    # Data
    parser.add_argument("--h5_path", type=str, default=default_h5)
    parser.add_argument("--noise_indices", type=str, default=default_noise)
    parser.add_argument("--channel", type=str, default=None, choices=["H1", "L1"], help="Train on specific channel only")
    
    # Model
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--context_dim", type=int, default=64)
    parser.add_argument("--prediction_steps", type=int, default=6)
    parser.add_argument("--delta_threshold", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
    parser.add_argument("--beta", type=float, default=0.85, help="LIF decay rate (beta)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4, help="Default changed to 3e-4")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    
    # Misc
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--fast", action="store_true", help="Run fast verification mode")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable Mixed Precision (AMP)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--resume_id", type=str, default=None, help="WandB Run ID to resume")
    
    args = parser.parse_args()
    train(args)
