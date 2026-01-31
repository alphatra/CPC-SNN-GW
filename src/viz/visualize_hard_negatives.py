import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.models.cpc_snn import CPCSNN
from src.data_handling.torch_dataset import HDF5SFTPairDataset

# Configuration
CHECKPOINT_PATH = "checkpoints/cpc_fix_v5_atan/best.pt"
DATA_PATH = "data/cpc_snn_train.h5"
OUTPUT_DIR = "figures/hard_negatives"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
THRESHOLD = 0.5  # Threshold to consider it a "detection"
MAX_SAMPLES = 20 # Max images to save

def load_model(path, device):
    print(f"Loading model from {path}...")
    # Initialize with default params matching your architecture
    # Ideally these should come from a config file saved with the model
    model = CPCSNN(
        in_channels=2,
        hidden_dim=256,   # Adjust based on training config
        context_dim=256,
        prediction_steps=12,
        use_checkpointing=False,
        use_continuous_input=True,
        no_dain=False
    )
    
    # Load state dict
    try:
        checkpoint = torch.load(path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Handle prefix issues if saved with DDP
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict, strict=False) # strict=False to be safe with slight version diffs
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def visualize_sample(sample_idx, signal, mask_h1, pred_score, label, save_path):
    """
    Visualizes H1 and L1 strain + Mask
    Signal shape: (2, Time)
    """
    t = np.arange(signal.shape[1])
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # H1
    axes[0].plot(t, signal[0], color='#1f77b4', label='H1 Strain')
    axes[0].set_title(f"H1 Detector (Sample {sample_idx})")
    axes[0].set_ylabel("Strain (Whitened)")
    axes[0].grid(True, alpha=0.3)
    
    # L1
    axes[1].plot(t, signal[1], color='#ff7f0e', label='L1 Strain')
    axes[1].set_title(f"L1 Detector")
    axes[1].set_ylabel("Strain (Whitened)")
    axes[1].grid(True, alpha=0.3)
    
    # Mask (Ground Truth Injection)
    if mask_h1 is not None:
        axes[2].plot(t, mask_h1, color='green', label='Injection Mask')
    axes[2].set_title(f"Ground Truth Mask (Label: {label})")
    axes[2].set_ylabel("Mask")
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(f"Hard Negative? | Label: {label} | Pred: {pred_score:.4f} | Threshold: {THRESHOLD}", fontsize=14, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 1. Load Dataset
    print(f"Loading dataset from {DATA_PATH}...")
    try:
        import h5py
        with h5py.File(DATA_PATH, "r") as f:
            keys = list(f.keys())
            
        dataset = HDF5SFTPairDataset(
            h5_path=DATA_PATH,
            index_list=keys,
            # Configs must match training
            use_phase=False, # Assuming raw strain input based on use_continuous_input=True
            return_mask=True
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # 2. Load Model
    model = load_model(CHECKPOINT_PATH, DEVICE)
    if model is None: return

    print(f"Scanning for Hard Negatives (Label=0, Pred > {THRESHOLD})...")
    
    fp_count = 0
    
    # Iterate
    # Note: Batch size 1 for simple analysis, can be batched for speed
    for i in tqdm(range(len(dataset))):
        if fp_count >= MAX_SAMPLES:
            break
            
        sample = dataset[i]
        
        # Prepare input
        # Dataset returns: {'H1': ..., 'L1': ..., 'y': ..., 'mask_H1': ...}
        # Model expects raw strain if use_continuous_input=True
        # Assuming dataset returns dict with 'H1' shape (Time,) or (1, Time)
        
        # We need to stack H1 and L1 -> (1, 2, Time)
        h1 = sample['H1']
        l1 = sample['L1']
        
        # Check shapes - dataset output varies by config
        if isinstance(h1, torch.Tensor):
            h1 = h1.float()
            l1 = l1.float()
        else:
            h1 = torch.tensor(h1).float()
            l1 = torch.tensor(l1).float()
            
        # Ensure (Time,)
        if h1.ndim > 1: h1 = h1.flatten()
        if l1.ndim > 1: l1 = l1.flatten()
            
        x = torch.stack([h1, l1], dim=0).unsqueeze(0).to(DEVICE) # (1, 2, Time)
        label = sample['y'].item()
        
        # Skip if it's actually an injection (we want False Positives only)
        if label == 1:
            continue
            
        # Inference
        with torch.no_grad():
            logits, _, _ = model(x)
            prob = torch.sigmoid(logits).item()
            
        # Check Condition: Noise (0) classified as Signal (>Threshold)
        if prob > THRESHOLD:
            fp_count += 1
            print(f"Found FP! Sample {keys[i]}, Prob: {prob:.4f}")
            
            # Save Plot
            mask = sample['mask_H1'] if 'mask_H1' in sample else None
            if isinstance(mask, torch.Tensor): mask = mask.numpy().flatten()
            
            save_path = f"{OUTPUT_DIR}/FP_sample_{keys[i]}_prob_{prob:.2f}.png"
            visualize_sample(keys[i], x.cpu().numpy()[0], mask, prob, label, save_path)
            
    print(f"Done. Found {fp_count} hard negatives. Saved in {OUTPUT_DIR}.")

if __name__ == "__main__":
    main()
