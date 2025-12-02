# Walkthrough - Data Handling Refactoring

I have successfully refactored the data handling pipeline for the CPC-SNN Gravitational Waves Detection project. The changes enable Multi-Interferometer (Multi-IFO) support, integrate Hydra for configuration management, and ensure robust data generation and loading.

## 1. Key Changes

### `src/data_handling/generate_training_set.py`
- **Hydra Integration**: Now uses `hydra` to manage configuration via `configs/data_generation.yaml`.
- **Multi-IFO Support**: 
    - Searches for coincident segments across multiple detectors (e.g., H1, L1).
    - Fetches and whitens data for all specified interferometers.
    - Projects simulated waveforms onto each detector using antenna patterns and time delays.
- **Robustness**:
    - Implemented `scipy.signal.welch` for PSD estimation to bypass `PyCBC` issues in the current environment.
    - Fixed `LIGOTimeGPS` type handling and `crop` usage for correct data processing.
    - Added `setup_gwosc_cache()` to ensure consistent data caching.

### `src/data_handling/gw_utils.py`
- **Refactored Whitening**: Updated `get_whitened_strain` to use `scipy.signal.welch` for stable PSD calculation.
- **Waveform Projection**: Added `project_waveform_to_ifo` to handle geometric delays and antenna responses for Multi-IFO injections.
- **Cleaned Imports**: Organized imports and removed unused code.

### `src/data_handling/torch_dataset.py`
- **Multi-IFO Loading**: Updated `HDF5SFTPairDataset` to load paired data (H1/L1) from the new HDF5 structure.
- **Flexible Configuration**: Supports loading magnitude, phase (cos/sin), and masks.

### `src/utils/paths.py`
- **Centralized Paths**: Defined `DATA_DIR`, `CACHE_DIR`, and `CONFIG_DIR` for consistent path management across the project.

## 2. Verification Results

### Data Generation
- **Script**: `src/data_handling/generate_training_set.py`
- **Config**: `configs/mini_data.yaml` (created for testing)
- **Result**: Successfully generated `data/mini_test.h5` containing 200 samples (100 background, 100 injection).
- **Fixes**: Added `ifos` to sample metadata to support visualization notebooks.
- **Log**:
  ```
  [INFO] - Found 1 coincident segments > 64s.
  [INFO] - Processing segment 1262304018-1262304118 (100s)...
  [INFO] - Done! Saved 200 samples to .../data/mini_test.h5
  ```


### Dataset Loading
- **Script**: `test_dataset_loading.py` (temporary)
- **Result**: Successfully instantiated `HDF5SFTPairDataset` and iterated through a batch.
- **Output**:
  ```
  Found 200 samples in data/mini_test.h5
  Dataset length: 200
  Batch keys: dict_keys(['H1', 'L1', 'mask_H1', 'mask_L1', 'f', 'y'])
  H1 shape: torch.Size([4, 4, 31, 246])
  ```

## 3. How to Run

### Generate Data
To generate a full training set, edit `configs/data_generation.yaml` and run:
```bash
python src/data_handling/generate_training_set.py
```

To run the mini test configuration:
```bash
python src/data_handling/generate_training_set.py --config-name mini_data
```

### Train Model
You can now proceed to train your model using the generated HDF5 file. Ensure your training script uses `src.data_handling.torch_dataset.HDF5SFTPairDataset`.
