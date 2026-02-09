"""
Deprecated legacy training script.

This file used to duplicate data-building and training logic that is now
implemented in:
- src/data_handling/gw_data.py
- src/data_handling/build_multi_hdf5.py
- src/train/train_cpc.py
"""


def main() -> None:
    raise SystemExit(
        "train_binary_gw.py is deprecated. Use: python -m src.train.train_cpc --help"
    )


if __name__ == "__main__":
    main()
