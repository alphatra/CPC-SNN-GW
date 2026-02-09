import h5py

def check_structure():
    path = "data/cpc_snn_train_spikes.h5"
    print(f"Inspecting {path}...")
    try:
        with h5py.File(path, 'r') as h5:
            print(f"Total keys: {len(h5.keys())}")
            # Show first 5 keys
            keys = list(h5.keys())[:5]
            print(f"Sample keys: {keys}")
            
            if keys:
                gid = keys[0]
                grp = h5[gid]
                print(f"Structure of '{gid}':")
                for k in grp.keys():
                    print(f"  - {k} : {grp[k]}")
    except Exception as e:
        print(f"Error accessing file: {e}")

if __name__ == "__main__":
    check_structure()
