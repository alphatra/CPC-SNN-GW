# ✅ Next Run Checklist (CPU sanity → GPU)

## CRITICAL UPDATE: Switch to MLGWSC-1 Dataset (HIGHEST PRIORITY)
- [ ] 🚨 **IMMEDIATE**: Generate MLGWSC-1 professional dataset (2778x more data than current)
  ```bash
  # Generate Dataset-4 (Real O3a background + PyCBC injections)
  mkdir -p /teamspace/studios/this_studio/data/dataset-4/v2
  cd /teamspace/studios/this_studio/ml-mock-data-challenge-1
  
  # Training data generation
  python3 generate_data.py -d 4 \
    -i /teamspace/studios/this_studio/data/dataset-4/v2/train_injections_s24w61w_1.hdf \
    -f /teamspace/studios/this_studio/data/dataset-4/v2/train_foreground_s24w61w_1.hdf \
    -b /teamspace/studios/this_studio/data/dataset-4/v2/train_background_s24w61w_1.hdf \
    --duration 600 --force
  
  # Validation data generation  
  python3 generate_data.py -d 4 \
    -i /teamspace/studios/this_studio/data/dataset-4/v2/val_injections_s24w6d1_1.hdf \
    -f /teamspace/studios/this_studio/data/dataset-4/v2/val_foreground_s24w6d1_1.hdf \
    -b /teamspace/studios/this_studio/data/dataset-4/v2/val_background_s24w6d1_1.hdf \
    --duration 600 --force
  
  # Waveform generation for AResGW-compatible training
  python3 /teamspace/studios/this_studio/gw-detection-deep-learning/scripts/generate_waveforms.py \
    --background-hdf /teamspace/studios/this_studio/data/dataset-4/v2/val_background_s24w6d1_1.hdf \
    --injections-hdf /teamspace/studios/this_studio/data/dataset-4/v2/val_injections_s24w6d1_1.hdf \
    --output-npy /teamspace/studios/this_studio/data/dataset-4/v2/val_injections_s24w6d1_1.25s.npy
  ```

## Secondary tasks (after MLGWSC-1 dataset ready)
- [ ] Fix pip in venv and install scikit-learn for full ROC/PR/ECE  
- [ ] Test with MLGWSC-1 data (expect 70%+ accuracy vs current 50%)

## Optional CPU safeguards
- [ ] Lower eval batch size (e.g., 16) to avoid LLVM OOM
- [ ] Cap quick-mode steps per epoch (e.g., 40) for faster feedback
- [ ] Consider class weighting/focal loss for collapse prevention on tiny sets

## Move to GPU (after CPU sanity OK)
- [ ] Switch `--device gpu` and remove quick caps
- [ ] Re-enable Orbax checkpoint managers (outside quick-mode)
- [ ] Increase batch size and/or sequence length for better learning


