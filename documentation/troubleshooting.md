# Troubleshooting

This guide addresses common issues encountered when using the CPC-SNN-GW system and provides solutions to resolve them.

## Common Issues and Solutions

### 1. "Delay kernel timed out" or CUDA Initialization Errors

**Symptom**: The training process fails immediately with errors like "Delay kernel timed out" or other CUDA-related initialization failures.

**Cause**: This is typically caused by the GPU's CUDA kernels not being properly initialized before the main computational graph is compiled and executed. This is a common issue with JAX on certain GPU systems.
**Solution**: The system includes a built-in 6-stage GPU warmup procedure to prevent this. Ensure you are using the main entry points (`cli.py` or `enhanced_cli.py`), as they automatically apply this warmup. If the problem persists, verify that your JAX installation is correct for your GPU and CUDA version. You can test your JAX/GPU setup with:
```python
import jax
print(jax.devices())
```
This should list your GPU (e.g., `[CudaDevice(id=0)]`). If it lists a CPU, your JAX GPU backend is not configured correctly.

---

### 2. CPC Loss is Zero (0.000000)

**Symptom**: The training logs show the CPC loss value stuck at 0.000000, indicating that the contrastive learning is not functioning.
**Cause**: This was a critical bug in earlier versions of the system, caused by an incorrect implementation of the CPC loss function that failed when the batch size was 1.
**Solution**: This issue has been resolved in the current version. The `calculate_fixed_cpc_loss` function in `training/cpc_loss_fixes.py` uses a robust Temporal InfoNCE implementation that works for any batch size. Ensure you are using the latest code and that the `create_enhanced_loss_fn` is being used in your training script. The loss should now be a positive value (e.g., > 0.5).

---

### 3. "Fake Accuracy" or Suspiciously High Test Accuracy

**Symptom**: The test accuracy is reported as very high (e.g., >95%) or even 100%, but the model is not actually learning.
**Cause**: This is almost always due to a flawed test set. The most common cause is a **single-class test set**, where all test samples belong to the same class (e.g., all noise). The model can achieve high accuracy by simply predicting the majority class.
**Solution**: The system uses a stratified split via `create_stratified_split` in `utils/data_split.py` to prevent this. If you encounter this issue, verify that your data loading and splitting code is correct. The `validate_split_quality` function will raise a `ValueError` if a single-class test set is detected. Check your data labels and ensure the dataset has a proper mix of signal and noise examples.

---

### 4. Out-of-Memory (OOM) Errors on GPU

**Symptom**: The training process crashes with an out-of-memory error, often during the JIT compilation phase.
**Cause**: The model or data batch is too large for the available GPU memory. This is common with large batch sizes or very deep models.
**Solution**: The primary solution is to reduce the `batch_size` in the configuration. The system is designed to work with `batch_size=1` for memory-constrained environments. To maintain training stability with a small batch size, enable gradient accumulation by setting `gradient_accumulation_steps` to a value greater than 1 (e.g., 4 or 8). This accumulates gradients over multiple small batches before updating the weights. You can also cap JAX's memory usage with the environment variable `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`.

---

### 5. Model Collapse (Always Predicts the Same Class)


**Symptom**: The model's predictions on the test set are all 0s or all 1s, regardless of the input.
**Cause**: This indicates a fundamental failure in the training process. The model has found a trivial solution (always predict the majority class) and has stopped learning.
**Solution**: The `evaluate_on_test_set` function in `training/test_evaluation.py` includes a `model_collapse` detection flag. If this is triggered, the training has failed. Common causes include a learning rate that is too high or too low, poor data quality, or a bug in the loss function. Try reducing the learning rate, verifying the data labels, and ensuring the CPC loss is working (not zero). Re-running the training with different random seeds may also help.

---

### 6. Real LIGO Data Not Found

**Symptom**: The `download_gw150914_data` function fails and falls back to simulated data, or raises a `FileNotFoundError`.
**Cause**: The required HDF5 files (`H-H1_LOSC_4_V2-1126259446-32.hdf5` and `L-L1_LOSC_4_V2-1126259446-32.hdf5`) are not present in the expected directory (usually the project root).
**Solution**: Download the files from the LIGO Open Science Center (LOSC) and place them in the project's root directory. The download links are provided in the [Installation Guide](installation-guide.md). Ensure the filenames match exactly.

## General Debugging Tips

*   **Check the Logs**: The system generates detailed logs in the `outputs/complete_enhanced_training/logs/` directory. The `training.log` file contains a comprehensive record of the training process and is the first place to look for error messages.
*   **Use Verbose Mode**: Run the CLI with `--verbose` or `--debug` flags (if available) to get more detailed output.
*   **Verify Dependencies**: Ensure all required Python packages (JAX, Spyx, ReadLIGO) are installed and the correct versions.
*   **Consult the Memory Bank**: The `memory-bank/` directory contains detailed context files (e.g., `activeContext.md`, `techContext.md`) that document known issues and solutions.