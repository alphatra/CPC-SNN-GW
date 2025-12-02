import h5py
import numpy as np

h5_path = "data/cpc_snn_train.h5"

noise_ids = []
signal_ids = []

with h5py.File(h5_path, 'r') as f:
    keys = [k for k in f.keys() if k.isdigit()]
    print(f"Total samples: {len(keys)}")
    
    for k in keys:
        if "label" in f[k]:
            label = f[k]["label"][()]
            if label == 0:
                noise_ids.append(k)
            elif label == 1:
                signal_ids.append(k)
            else:
                print(f"Unknown label {label} for {k}")
        else:
            # Fallback to metadata if label dataset missing
            if "meta" in f[k] and "is_signal" in f[k]["meta"].attrs:
                 is_signal = f[k]["meta"].attrs["is_signal"]
                 if is_signal == 0:
                     noise_ids.append(k)
                 else:
                     signal_ids.append(k)
            else:
                print(f"No label or metadata for {k}")

print(f"Noise samples: {len(noise_ids)}")
print(f"Signal samples: {len(signal_ids)}")

# Save indices for later use
import json
with open("data/indices_noise.json", "w") as f:
    json.dump(noise_ids, f)
with open("data/indices_signal.json", "w") as f:
    json.dump(signal_ids, f)
