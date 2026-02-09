import torch
from torch.utils.data import DataLoader
import h5py
from tqdm import tqdm

from src.data_handling.torch_dataset import HDF5SFTPairDataset
from src.models.cpc_snn import CPCSNN

def smoke_test():
    print("=== Starting Smoke Test ===")
    
    # 1. Load Data
    h5_path = "data/cpc_snn_train.h5"
    print(f"Loading data from {h5_path}...")
    
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        # Use a small subset
        keys = keys[:32] 
        # Convert keys to int if needed by Dataset, but Dataset expects list of IDs
        # The dataset class converts them to str internally.
        
    dataset = HDF5SFTPairDataset(h5_path, keys, return_time_series=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(f"Dataset loaded. Samples: {len(dataset)}")
    
    # 2. Initialize Model
    print("Initializing CPCSNN model...")
    model = CPCSNN(
        in_channels=2, 
        hidden_dim=32, 
        context_dim=32, 
        prediction_steps=4,
        delta_threshold=0.1,
        use_metal=False
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 3. Training Loop (1 Epoch, few batches)
    print("Running training loop...")
    model.train()
    
    for i, batch in enumerate(dataloader):
        # Batch is a dict
        # 'x': (Batch, Channels, Time)
        
        x = batch['x']
        # Slice to 1000 steps for speed in smoke test
        x = x[:, :, :1000]


        
        # Forward
        logits, c, z = model(x)
        y = batch["label"].float().view(-1, 1)
        
        # Loss (binary classification smoke check)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y).float().mean().item()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Batch {i}: Loss={loss.item():.4f}, Acc={acc:.4f}")
        
    print("=== Smoke Test Passed! ===")

if __name__ == "__main__":
    smoke_test()
