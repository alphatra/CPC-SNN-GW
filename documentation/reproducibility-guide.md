# Reproducibility Guide

This guide provides detailed instructions for reproducing the experimental results of the CPC-SNN-GW system, ensuring the scientific validity and transparency of the research.

## Software and Environment

To reproduce the results, it is essential to use the exact software versions and environment configuration.

### Python Environment
Create a new conda environment with the specified versions:
```bash
conda create -n cpc-snn-repro python=3.10.10
conda activate cpc-snn-repro
```

### Python Package Dependencies
Install the following packages with their exact versions:
```bash
pip install jax==0.4.25 jaxlib==0.4.25
pip install spyx==0.1.0  # Or the specific version used
pip install gwpy==3.0.4 readligo.py==1.1.0
pip install wandb==0.15.12 tensorboard==2.13.0
pip install numpy==1.24.3 scipy==1.10.1
```

**Note on JAX and GPU**: The JAX installation must match the CUDA version of the system. For a CUDA 12.2 system, use:
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### System Configuration
The experiments were conducted on a machine with the following specifications:
*   **GPU**: NVIDIA Tesla T4 (16GB VRAM)
*   **CPU**: 4 cores, 8 logical processors
*   **RAM**: 32 GB
*   **Operating System**: Linux (Ubuntu 20.04)
*   **CUDA Version**: 12.2

## Data

The primary dataset is the authentic GW150914 strain data from the LIGO Open Science Center (LOSC).

### Data Acquisition
Download the following HDF5 files and place them in the project's root directory:
*   [H-H1_LOSC_4_V2-1126259446-32.hdf5](https://www.gw-openscience.org/eventapi/html/GWTC-1-confident/GW150914/v3/H1/H-H1_LOSC_4_V2-1126259446-32.hdf5)
*   [L-L1_LOSC_4_V2-1126259446-32.hdf5](https://www.gw-openscience.org/eventapi/html/GWTC-1-confident/GW150914/v3/L1/L-L1_LOSC_4_V2-1126259446-32.hdf5)


### Data Processing
The dataset is created by the `create_real_ligo_dataset` function with the following parameters:
*   `num_samples`: 2000
*   `window_size`: 256
*   `quick_mode`: False (enables data augmentation)
*   `return_split`: True
*   `train_ratio`: 0.8

This results in a training set of 1600 samples and a test set of 400 samples, with a stratified split to ensure balanced class representation.


## Experiment Execution

### Configuration
Use the `configs/final_framework_config.yaml` file as the base configuration. The key parameters for the reproduction are:
*   `batch_size`: 1
*   `learning_rate`: 0.0005
*   `num_epochs`: 10
*   `gradient_accumulation_steps`: 4
*   `use_real_ligo_data`: true
*   `output_dir`: "outputs/repro_run"

### Running the Experiment
Execute the complete enhanced training pipeline:
```bash
python cli.py --mode train --config configs/final_framework_config.yaml --output-dir outputs/repro_run
```

This command will automatically apply the 6-stage GPU warmup, load the real LIGO data, perform the training, and run the final evaluation.

## Result Validation

After the run completes, validate the results by examining the output files in the `outputs/repro_run/` directory.

1.  **Check the Logs**: Open `logs/training.log` and verify that:
    *   The GPU warmup completed successfully.
    *   The real LIGO data was loaded.
    *   The CPC loss is non-zero and decreases over time.
    *   The final training accuracy is reported.

2.  **Check the Final Metrics**: The `evaluate_on_test_set` function will print a comprehensive summary. Verify that the test accuracy, sensitivity, specificity, and other metrics are within an acceptable range of the reported values (e.g., test accuracy ~40.2%).
3.  **Check for Warnings**: Ensure there are no warnings about model collapse, data leakage, or suspiciously high accuracy.

## Statistical Reproducibility

To account for the stochastic nature of training (e.g., random initialization, data shuffling), the experiment should be repeated multiple times (e.g., 5-10 runs) with different random seeds. The `random_seed` parameter in the configuration or the `jax.random.PRNGKey` can be modified for each run. The final reported performance should be the mean and standard deviation (or a confidence interval) of the results from these multiple runs, providing a robust statistical foundation for the claims made about the system's performance.