import h5py
import json
import numpy as np
import os
import sys

def verify_data():
    project_root = "/Users/gracjanziemianski/Documents/CPC-SNN-GravitationalWavesDetection"
    h5_path = os.path.join(project_root, "data/cpc_snn_train.h5")
    noise_path = os.path.join(project_root, "data/indices_noise.json")
    signal_path = os.path.join(project_root, "data/indices_signal.json")
    
    print(f"Loading indices...")
    with open(noise_path, 'r') as f:
        noise_ids = json.load(f)
    with open(signal_path, 'r') as f:
        signal_ids = json.load(f)
        
    print(f"Noise IDs: {len(noise_ids)}")
    print(f"Signal IDs: {len(signal_ids)}")
    
    with h5py.File(h5_path, 'r') as h5:
        print("\n--- Verifying SIGNAL samples ---")
        for i in range(5):
            gid = str(signal_ids[i])
            print(f"DEBUG: gid='{gid}', type={type(gid)}")
            
            if gid not in h5:
                print(f"MISSING: {gid}")
                continue
            
            grp = h5[gid]
            print(f"DEBUG: grp type={type(grp)}")
            print(f"DEBUG: grp keys={list(grp.keys())}")
            
            try:
                y = grp["label"][()] if "label" in grp else "N/A"
                print(f"DEBUG: Label type={type(y)} val={y}")
                
                # Check Key explicitly
                if "H1" in grp:
                    dset = grp["H1"]
                    # H1 is a group with keys ['mag', 'cos', 'sin', 'mask']
                    if "mag" in dset:
                        mag = dset["mag"][()]
                        h1 = mag # Use magnitude for stats
                    else:
                        print("DEBUG: mag missing in H1")
                        continue
                else:
                    print("DEBUG: H1 missing")
                    continue
            except Exception as e:
                print(f"DEBUG CRASH: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Check if stored as STFT (complex) or TimeSeries
            msg = f"ID: {gid} | Label: {y} (Expect 1)"
            if int(y) == 1:
                msg += " [OK]"
            else:
                msg += " [FAIL]"
            
            msg += f" | H1 shape: {h1.shape} | Mean: {np.mean(h1):.4f} | Std: {np.std(h1):.4f}"
            print(msg)
            
        print("\n--- Verifying NOISE samples ---")
        for i in range(5):
            gid = str(noise_ids[i])
            if gid not in h5:
                print(f"MISSING: {gid}")
                continue
                
            grp = h5[gid]
            y = grp["label"][()] if "label" in grp else "N/A"
            
            if "H1" in grp:
                dset = grp["H1"]
                # H1 is a group with keys ['mag', 'cos', 'sin', 'mask']
                if "mag" in dset:
                    mag = dset["mag"][()]
                    h1 = mag # Use magnitude for stats
                else:
                    print("DEBUG: mag missing in H1")
                    continue
            else:
                print("DEBUG: H1 missing")
                continue
            
            msg = f"ID: {gid} | Label: {y} (Expect 0)"
            if int(y) == 0:
                msg += " [OK]"
            else:
                msg += " [FAIL]"
            
            msg += f" | H1 shape: {h1.shape} | Mean: {np.mean(h1):.4f} | Std: {np.std(h1):.4f}"
            print(msg)

if __name__ == "__main__":
    verify_data()
