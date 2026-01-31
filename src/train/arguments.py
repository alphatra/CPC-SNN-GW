import argparse
import os

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Train CPC-SNN on Background Noise")
    
    # Resolve paths relative to project root
    # script_dir = src/train
    # project_root = src/train/../..
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    default_h5 = os.path.join(project_root, "data/cpc_snn_train.h5")
    # Updated to use existing _OLD files as defaults
    default_noise = os.path.join(project_root, "data/indices_noise.json")
    default_signal = os.path.join(project_root, "data/indices_signal.json")
    
    # Data
    parser.add_argument("--h5_path", type=str, default=default_h5)
    parser.add_argument("--noise_indices", type=str, default=default_noise)
    parser.add_argument("--signal_indices", type=str, default=default_signal)
    parser.add_argument("--channel", type=str, default=None, choices=["H1", "L1"], help="Train on specific channel only")
    
    # Model
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--context_dim", type=int, default=64)
    parser.add_argument("--prediction_steps", type=int, default=6)
    parser.add_argument("--delta_threshold", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
    parser.add_argument("--beta", type=float, default=0.85, help="LIF decay rate (beta)")
    parser.add_argument("--no_dain", action="store_true", default=False, help="Disable DAIN normalization")
    # Basic
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm")
    
    # Misc
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--fast", action="store_true", help="Run fast verification mode")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--amp", action="store_true", default=False, help="Enable Mixed Precision (AMP)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--resume_id", type=str, default=None, help="WandB Run ID to resume")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch Profiler for one epoch")
    parser.add_argument("--offline", action="store_true", help="Run W&B in offline mode")
    parser.add_argument("--checkpointing", action="store_true", help="Enable Gradient Checkpointing (slower but saves memory)")
    parser.add_argument("--no_metal", action="store_true", default=False, help="Disable Native Metal (MSL) Kernel for LIF updates")
    parser.add_argument("--use_discrete_input", action="store_true", default=False, help="Use Legacy FastDeltaEncoder (Discrete Spikes) instead of learnable raw input")
    parser.add_argument("--no_aug", action="store_true", default=False, help="Disable data augmentation")
    parser.add_argument("--aug_amp", type=float, default=0.1, help="Amplitude Jitter Strength (portion, e.g. 0.1)")
    parser.add_argument("--aug_freq_mask", type=int, default=8, help="Frequency Masking (SpecAugment) - Max Bands")
    parser.add_argument("--aug_prob", type=float, default=0.5, help="Probability of applying augmentations")
    parser.add_argument("--force_reconstruct", type=str2bool, default=False, help="Force on-the-fly reconstruction")
    parser.add_argument("--use_sft", action="store_true", default=False, help="Use raw SFT tensors input")
    parser.add_argument("--use_tf2d", action="store_true", default=True, help="Use 2D Conv Encoder on STFT (B, 6, T, F)")
    parser.add_argument("--sft_channels", type=int, default=2, help="Number of SFT channels to use (1=H1, 2=H1+L1)")
    parser.add_argument("--no_mask", action="store_true", default=False, help="Disable Mask Channel (Use only Mag/Cos/Sin) for leakage check")
    
    # Robust Validation Args
    parser.add_argument("--split_strategy", type=str, default="time", choices=["random", "time"], help="Data split strategy")
    parser.add_argument("--sanity_permute_labels", action="store_true", default=False, help="Sanity check: Permute labels")
    parser.add_argument("--sanity_noise_only", action="store_true", default=False, help="Sanity check: Train on noise vs noise")
    
    parser.add_argument("--stability_test", action="store_true", default=False, help="Run stability test: 512 samples, 50 epochs, constant LR")
    
    # CPC / InfoNCE Args
    parser.add_argument("--train_mode", type=str, default="supervised", choices=["supervised", "pretrain_cpc", "finetune"], help="Training mode")
    parser.add_argument("--freeze_encoder", action="store_true", default=False, help="Freeze encoder during finetuning")
    parser.add_argument("--lambda_infonce", type=float, default=0.1, help="Weight for InfoNCE loss in multi-task mode (if applicable)")
    
    # Advanced Loss Args (Tail Suppression)
    parser.add_argument("--loss_type", type=str, default="focal", choices=["bce", "focal"], help="Loss function type")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma for Focal Loss")
    parser.add_argument("--tail_penalty", type=float, default=2.0, help="Weight for Noise Tail Penalty (ReLU(p_noise - thr))")
    parser.add_argument("--tail_threshold", type=float, default=0.5, help="Threshold for Noise Tail Penalty")
    
    args = parser.parse_args()
    
    if args.profile:
        print("Profiling enabled for 1 epoch...")
        args.epochs = 1
        
    return args
