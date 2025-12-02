import h5py
import os
from src.utils.paths import DATA_DIR

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

SRC_PATH = DATA_DIR / "gw_multi_events_sft.h5"
DST_PATH = DATA_DIR / "gw_multi_events_sft_clean.h5"


def is_valid_sample(grp):
    required_toplevel = ["H1", "L1", "f", "label"]
    for key in required_toplevel:
        if key not in grp:
            return False
    for ifo in ["H1", "L1"]:
        if ifo not in grp:
            return False
        for sub in ["cos", "sin", "mag"]:
            if sub not in grp[ifo]:
                return False
    return True


print("=== HDF5 CLONE & CLEAN ===")
print("Source:", SRC_PATH)
print("Destination:", DST_PATH)

with h5py.File(SRC_PATH, "r") as src, h5py.File(DST_PATH, "w") as dst:

    all_ids = sorted(src.keys())
    print(f"[INFO] Found {len(all_ids)} samples in source")

    valid = []
    invalid = []

    for gid in all_ids:
        grp = src[gid]
        if is_valid_sample(grp):
            valid.append(gid)
        else:
            invalid.append(gid)

    print(f"[INFO] Valid samples:   {len(valid)}")
    print(f"[INFO] Invalid samples: {len(invalid)}")

    print("\n[INFO] Copying valid samples to new file...\n")

    for gid in valid:
        src.copy(gid, dst)
        print(f"  [COPY] {gid}")

    print("\n[INFO] Clone complete.")
    print(f"[INFO] Final number of samples in cleaned file: {len(dst.keys())}")

print("\n[OK] All done.")
