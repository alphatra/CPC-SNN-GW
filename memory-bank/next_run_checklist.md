# ✅ Next Run Checklist (CPU sanity → GPU)

## Must-do before next run
- [ ] Fix pip in venv and install scikit-learn for full ROC/PR/ECE
  - `python -m ensurepip --upgrade`
  - `python -m pip install -U pip setuptools wheel`
  - `python -m pip install scikit-learn`
  - (fallback) `curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py && python -m pip install scikit-learn`
- [ ] Run fast synthetic sanity (2 epochs) on CPU:
  ```bash
  python cli.py train --mode standard --epochs 2 --batch-size 1 \
    --quick-mode --synthetic-quick --synthetic-samples 60 \
    --spike-time-steps 8 --snn-hidden 32 --cpc-layers 2 --cpc-heads 2 \
    --balanced-early-stop --opt-threshold \
    --output-dir outputs/sanity_2ep_cpu_synth --device cpu
  ```

## Optional CPU safeguards
- [ ] Lower eval batch size (e.g., 16) to avoid LLVM OOM
- [ ] Cap quick-mode steps per epoch (e.g., 40) for faster feedback
- [ ] Consider class weighting/focal loss for collapse prevention on tiny sets

## Move to GPU (after CPU sanity OK)
- [ ] Switch `--device gpu` and remove quick caps
- [ ] Re-enable Orbax checkpoint managers (outside quick-mode)
- [ ] Increase batch size and/or sequence length for better learning


