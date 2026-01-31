import torch
import torch.profiler
from src.train.arguments import parse_args
from src.train.data_setup import setup_dataloaders
from src.train.trainer import train_model

def main():
    # 1. Parse Arguments
    args = parse_args()
    
    # 2. Device Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 3. Setup Data
    train_loader, val_loader, use_spikes, use_timeseries = setup_dataloaders(args, device)
    
    # 4. Profiler Context (Optional)
    profiler = None
    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            
        with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            train_model(args, train_loader, val_loader, device, use_spikes, use_timeseries, profiler=prof)\

    else:
        # 5. Run Training
        train_model(args, train_loader, val_loader, device, use_spikes, use_timeseries)

if __name__ == "__main__":
    main()
