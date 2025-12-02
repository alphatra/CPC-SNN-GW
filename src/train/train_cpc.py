import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import argparse
import numpy as np
import wandb
import time

from src.models.cpc_snn import CPCSNN
from src.data_handling.torch_dataset import HDF5SFTPairDataset, HDF5TimeSeriesDataset, InMemoryHDF5Dataset, InMemoryGPU_Dataset
from src.utils.scheduler import SNRScheduler
from src.train.cpc_trainer import CPCTrainer, generate_signal_bank
from src.data_handling.generate_balanced_bank import generate_balanced_bank

def train(args):
    # 1. W&B Init
    if not args.no_wandb:
        mode = "offline" if args.offline else "online"
        wandb_id = args.resume_id if args.resume_id else wandb.util.generate_id()
        wandb.init(project="cpc-snn-gw", config=args, name=args.run_name, id=wandb_id, resume="allow", mode=mode)
    
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
    
    # Force on-the-fly if SNR Scheduler is active
    if args.snr_scheduler:
        if use_spikes or use_timeseries:
            print("WARNING: Dynamic SNR Scheduling enabled. Ignoring pre-computed spikes/timeseries to allow on-the-fly injection.")
        use_spikes = False
        use_timeseries = False
    
    # Set flags in args for Trainer
    args.use_spikes = use_spikes
    args.use_timeseries = use_timeseries
    
    # Signal Bank for Injection (GPU Tensor)
    # Load if we are using on-the-fly dataset (HDF5SFTPairDataset/InMemory)
    # Even if snr_scheduler is False, we might want fixed SNR injection.
    signal_bank = None
    if not (use_spikes or use_timeseries):
         if args.balanced_bank:
             bank_path = args.balanced_bank_path
             if os.path.exists(bank_path):
                 print(f"Loading Balanced Signal Bank from {bank_path}...")
                 signal_bank = torch.load(bank_path, map_location=device)
                 print(f"Balanced Bank Loaded: {len(signal_bank)} bins")
             else:
                 print(f"Balanced Bank not found at {bank_path}. Generating...")
                 generate_balanced_bank(bank_path, n_per_bin=100)
                 signal_bank = torch.load(bank_path, map_location=device)
                 print(f"Balanced Bank Generated and Loaded: {len(signal_bank)} bins")
         else:
             # Legacy Random Generation
             signal_bank = generate_signal_bank(n_signals=200 if not args.fast else 10).to(device)
             print(f"Signal Bank Loaded on GPU: {signal_bank.shape}")
    
    # Split indices for train/val
    indices = list(range(len(noise_indices))) # Use noise_indices length for splitting
    split_idx = int(len(indices) * 0.8) # 80% training, 20% validation

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Dataset loading logic
    if use_spikes:
        print(f"Using pre-encoded spikes from {spikes_path}")
        train_dataset = HDF5TimeSeriesDataset(h5_path=spikes_path, index_list=[noise_indices[i] for i in train_indices])
        val_dataset = HDF5TimeSeriesDataset(h5_path=spikes_path, index_list=[noise_indices[i] for i in val_indices])
    elif use_timeseries:
        print(f"Using pre-processed time series data from {ts_path}")
        train_dataset = HDF5TimeSeriesDataset(h5_path=ts_path, index_list=[noise_indices[i] for i in train_indices])
        val_dataset = HDF5TimeSeriesDataset(h5_path=ts_path, index_list=[noise_indices[i] for i in val_indices])
    else:
        # Use InMemoryGPU_Dataset for maximum speed (Total Cache)
        print("Using InMemoryGPU_Dataset (Total Cache) for maximum speed...")
        train_dataset = InMemoryGPU_Dataset(
            args.h5_path, 
            [noise_indices[i] for i in train_indices], 
            device=device
        )
        val_dataset = InMemoryGPU_Dataset(
            args.h5_path, 
            [noise_indices[i] for i in val_indices], 
            device=device
        )
    
    print(f"Time Series Split: Train [{len(train_dataset)}] samples, Val [{len(val_dataset)}] samples")
    
    loader_kwargs = {
        "batch_size": args.batch_size,
        "pin_memory": False, # Data already on GPU
        "num_workers": 0, # No workers needed
    }
    
    if use_spikes or use_timeseries:
        # For legacy datasets, we might want workers
        loader_kwargs["num_workers"] = args.workers
        loader_kwargs["pin_memory"] = True
        if args.workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    
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
    
    # AMP Scaler
    use_amp = args.amp and hasattr(torch, 'amp')
    if use_amp and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('mps')
    elif use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Optimize with torch.compile (PyTorch 2.0+)
    if args.compile and hasattr(torch, "compile"):
        print(f"Compiling model with torch.compile (backend={args.compile_backend})...")
        print("Setting TORCH_LOGS='+dynamo' for verbose output.")
        os.environ["TORCH_LOGS"] = "+dynamo"
        torch._dynamo.config.suppress_errors = True
        
        # Clean up memory before compilation to avoid OOM
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            t0_comp = time.time()
            # Use specified backend (default 'inductor' often fails on MPS, 'aot_eager' is safer)
            model = torch.compile(model, backend=args.compile_backend)
            
            # Trigger compilation with a dummy forward pass
            print("Triggering compilation with dummy input...")
            # Use a smaller batch for compilation trigger to save memory? 
            # No, graph should match runtime.
            dummy_input = torch.randn(args.batch_size, in_channels, 16384).to(device)
            with torch.no_grad():
                if use_amp:
                    with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                        _ = model(dummy_input, is_encoded=False)
                else:
                    _ = model(dummy_input, is_encoded=False)
            
            print(f"Model compiled successfully in {time.time() - t0_comp:.1f}s!")
        except Exception as e:
            print(f"WARNING: torch.compile failed: {e}")
            print("Continuing without compilation...")
            # Fallback to uncompiled model if compilation failed (and didn't crash process)
            # Note: If it crashed (SIGKILL), we can't catch it here.
            if hasattr(model, "_orig_mod"):
                model = model._orig_mod
    
    if not args.no_wandb:
        wandb.watch(model, log="all")
        
    # Optimizer
    # Use fused=True if supported (PyTorch 2.0+)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
        print("Using Fused AdamW optimizer.")
    except:
        print("Fused AdamW not supported, falling back to standard.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler (LR)
    scheduler_lr = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1e4
    )
    
    # SNR Scheduler
    snr_scheduler = None
    if args.snr_scheduler:
        snr_scheduler = SNRScheduler(
            start_snr=args.start_snr,
            min_snr=args.min_snr,
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs
        )
        print(f"SNR Scheduler Active: Start={args.start_snr}, Min={args.min_snr}, Warmup={args.warmup_epochs}")
    
    # Resume Logic
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                 # scheduler_lr.load_state_dict(checkpoint['scheduler_state_dict'])
                 pass
            
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    # 5. Initialize Trainer
    trainer = CPCTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        device=device,
        scaler=scaler,
        scheduler_lr=scheduler_lr,
        snr_scheduler=snr_scheduler,
        signal_bank=signal_bank
    )

    # 6. Training Loop
    best_val_loss = float('inf')
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("checkpoints", args.run_name or timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting training for {args.epochs} epochs (starting from {start_epoch})...")
    print(f"AMP Enabled: {use_amp}")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        avg_train_loss, avg_train_acc = trainer.train_epoch(epoch)
        
        # Validate
        avg_val_loss, avg_val_acc = trainer.validate()
        
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
        
        # Save periodic
        if (epoch + 1) % args.save_interval == 0:
            torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"))
            print(f"Saved periodic checkpoint to {save_dir}/checkpoint_epoch_{epoch+1}.pt")
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(save_dir, "best.pt"))
            print(f"New best model saved to {save_dir}/best.pt")

    print("Training complete.")
    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CPC-SNN on Background Noise")
    
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
    parser.add_argument("--delta_threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
    parser.add_argument("--beta", type=float, default=0.85, help="LIF decay rate (beta)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4, help="Default changed to 3e-4")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every N epochs")
    
    # SNR Scheduling
    parser.add_argument("--snr_scheduler", action="store_true", default=False, help="Enable Dynamic SNR Scheduling")
    parser.add_argument("--start_snr", type=float, default=20.0, help="Starting SNR for Warmup")
    parser.add_argument("--min_snr", type=float, default=4.0, help="Minimum SNR after decay")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--balanced_bank", action="store_true", default=False, help="Use pre-computed Balanced SNR Bank")
    parser.add_argument("--balanced_bank_path", type=str, default="data/balanced_signal_bank.pt", help="Path to balanced bank file")
    
    # Misc
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--fast", action="store_true", help="Run fast verification mode")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--offline", action="store_true", default=False, help="Run W&B in offline mode (sync later)")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable Mixed Precision (AMP)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--resume_id", type=str, default=None, help="WandB Run ID to resume")
    parser.add_argument("--compile", action="store_true", default=False, help="Enable torch.compile optimization")
    parser.add_argument("--compile_backend", type=str, default="aot_eager", help="Backend for torch.compile (e.g., inductor, aot_eager)")
    
    args = parser.parse_args()
    
    # Prevent Sleep (macOS)
    if os.uname().sysname == 'Darwin':
        import subprocess
        print("Preventing system sleep (caffeinate)...")
        # -i: prevent idle sleep, -s: prevent system sleep
        subprocess.Popen(['caffeinate', '-i'])
        
    train(args)
