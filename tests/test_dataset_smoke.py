from pathlib import Path

import h5py
import pytest

from src.data_handling.torch_dataset import HDF5SFTPairDataset


def test_hdf5_dataset_single_sample_smoke():
    h5_path = Path("data/cpc_snn_train.h5")
    if not h5_path.exists():
        pytest.skip("Missing data/cpc_snn_train.h5")

    with h5py.File(h5_path, "r") as h5:
        ids = [k for k in h5.keys() if k.isdigit()]
        if not ids:
            pytest.skip("No numeric sample ids in HDF5")
        sample_id = ids[0]

    ds = HDF5SFTPairDataset(
        h5_path=str(h5_path),
        index_list=[sample_id],
        return_time_series=False,
    )
    sample = ds[0]
    assert "H1" in sample and "L1" in sample and "label" in sample
    assert sample["H1"].dim() == 3
    assert sample["L1"].dim() == 3
