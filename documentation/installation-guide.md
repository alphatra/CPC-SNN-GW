# Installation Guide

## Prerequisites

Before installing the CPC-SNN-GW system, ensure your system meets the following requirements:

*   **Operating System**: Linux or macOS. The system has been tested on Ubuntu 20.04+ and macOS with Apple Silicon.
*   **Hardware**: A GPU with CUDA support (e.g., NVIDIA T4, V100) is strongly recommended for training. The system can run on CPU for inference, but performance will be significantly slower.
*   **Python**: Python 3.8 or later. We recommend using a virtual environment (e.g., `venv` or `conda`) to manage dependencies.
*   **Package Manager**: `pip` for installing Python packages.

## Installation Steps

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-repo/cpc-snn-gw.git
    cd cpc-snn-gw
    ```

2.  **Create and Activate a Virtual Environment (Recommended)**:
    ```bash
    python -m venv cpc-snn-env
    source cpc-snn-env/bin/activate  # On Windows, use `cpc-snn-env\Scripts\activate`
    ```

3.  **Install System Dependencies**:
    The system relies on several key libraries. Install them using `pip`:
    ```bash
    pip install --upgrade pip
    pip install jax[cpu]  # For CPU-only. For GPU, use `jax[cuda11_pip]` or `jax[cuda12_pip]`
    pip install spyx  # The neuromorphic computing library
    pip install gwpy  # For data access (alternative to ReadLIGO)
    pip install readligo.py  # For direct HDF5 access to LIGO data
    pip install wandb  # For experiment tracking
    pip install tensorboard  # For logging
    ```
    **Note on JAX and GPU**: The correct JAX installation for your GPU and CUDA version is critical. Consult the [JAX installation guide](https://github.com/google/jax#installation) for the appropriate command. For a CUDA 12.2 system, you would use `pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`.

4.  **Install the CPC-SNN-GW Package**:
    Install the package in development mode to ensure all local changes are reflected:
    ```bash
    pip install -e .
    ```
    This command reads the `setup.py` or `pyproject.toml` file in the repository root and installs the package and its dependencies.

5.  **Download Required Data**:
    The system requires the GW150914 HDF5 files. These can be downloaded from the LIGO Open Science Center (LOSC):
    *   [H1 Detector Data](https://www.gw-openscience.org/eventapi/html/GWTC-1-confident/GW150914/v3/H1/H-H1_LOSC_4_V2-1126259446-32.hdf5)
    *   [L1 Detector Data](https://www.gw-openscience.org/eventapi/html/GWTC-1-confident/GW150914/v3/L1/L-L1_LOSC_4_V2-1126259446-32.hdf5)
    Place these files in the project root directory (or the directory specified in the configuration).

6.  **Verify the Installation**:
    Run a quick test to ensure all components are working:
    ```bash
    python -c "import jax; print(jax.devices())"
    python -c "from data.real_ligo_integration import download_gw150914_data; data = download_gw150914_data(); print(f'Data shape: {data.shape}' if data is not None else 'Failed to load data')"
    ```
    The first command should list your available JAX devices (e.g., `[CudaDevice(id=0)]`). The second command should successfully load the GW150914 strain data and print its shape.

## Configuration

The system's behavior is controlled by YAML configuration files located in the `configs/` directory. The primary configuration file is `final_framework_config.yaml`. You can override settings from the command line or by creating a custom config file.

Key configuration parameters include:
*   `batch_size`: The number of samples processed in a single forward/backward pass. Set to `1` for memory-constrained GPUs.
*   `learning_rate`: The initial learning rate for the optimizer.
*   `num_epochs`: The total number of training epochs.
*   `model_name`: The name of the model architecture to use.
*   `use_wandb`: Whether to enable Weights & Biases experiment tracking.

For a full list of parameters, refer to the `config.yaml` file in the `outputs/` directory of a previous run, which contains the complete, resolved configuration.