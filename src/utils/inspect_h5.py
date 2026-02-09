import h5py
import argparse
import numpy as np

def inspect(path):
    print(f"Inspecting {path}...")
    with h5py.File(path, 'r') as f:
        k = list(f.keys())[0]
        grp = f[k]
        
        if 'meta' in grp:
            obj = grp['meta']
            print(f"Type of 'meta': {type(obj)}")
            
            if isinstance(obj, h5py.Group):
                print(f"Meta is a Group. Keys: {list(obj.keys())}")
                print(f"Attributes: {dict(obj.attrs)}")
            elif isinstance(obj, h5py.Dataset):
                print(f"Meta is a Dataset.")
                print(f"Shape: {obj.shape}")
                print(f"Dtype: {obj.dtype}")
                try:
                    data = obj[()]
                    print(f"Data: {data}")
                except Exception as e:
                    print(f"Error reading data: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    inspect(args.path)
