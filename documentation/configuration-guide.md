# Configuration Guide

## Configuration Files

The CPC-SNN-GW system uses YAML configuration files to manage its settings. This approach provides a clear, human-readable way to define hyperparameters and system behavior, making experiments easy to configure, reproduce, and share.

The primary configuration files are located in the `configs/` directory:

*   `final_framework_config.yaml`: The main configuration file that defines the complete setup for the enhanced training pipeline. This includes model architecture, training hyperparameters, data processing options, and logging settings.
*   `enhanced_wandb_config.yaml`: A configuration file specifically for Weights & Biases (W&B) experiment tracking, defining project name, run name, and logging metrics.

When the system starts (e.g., via `cli.py`), it loads the default configuration from `configs/final_framework_config.yaml`. Users can override these settings by providing a path to a custom configuration file using the `--config` command-line argument.


## Key Configuration Parameters

### Model Configuration
These parameters define the architecture and components of the neuromorphic model.

*   `model_name` (str): The name of the model. Default: `"cpc_snn_gw"`.
*   `cpc_latent_dim` (int): The dimensionality of the latent space in the CPC encoder. A higher value allows for more complex representations but increases memory usage. Default: `128`.
*   `snn_hidden_size` (int): The number of neurons in the hidden layers of the SNN classifier. Default: `96`.
*   `snn_neurons_per_layer` (List[int]): The number of neurons in each layer of the SNN. This allows for a custom network depth and width (e.g., `[512, 256, 128, 64]`).
*   `snn_num_layers` (int): The total number of layers in the SNN classifier. Default: `4`.
*   `lif_membrane_tau` (float): The membrane time constant for the LIF neurons, controlling the rate of membrane potential decay. Default: `5e-5`.
*   `surrogate_gradient_beta` (float): The beta parameter for the surrogate gradient function used during SNN backpropagation. A higher value creates a sharper gradient approximation. Default: `4.0`.
*   `simulation_time_steps` (int): The number of discrete time steps the SNN is simulated for during a forward pass. A longer simulation allows the network more time to integrate information. Default: `4096`.

### Training Configuration
These parameters control the training process.

*   `batch_size` (int): The number of samples processed in a single batch. Set to `1` for memory-constrained GPUs. Default: `16`.
*   `learning_rate` (float): The initial learning rate for the optimizer (e.g., SGD). Default: `0.0005`.
*   `weight_decay` (float): The L2 regularization coefficient to prevent overfitting. Default: `0.0001`.
*   `num_epochs` (int): The total number of epochs to train for. Default: `10`.
*   `optimizer` (str): The optimization algorithm to use (e.g., `"sgd"`, `"adam"`). Default: `"sgd"`.
*   `scheduler` (str): The learning rate scheduler (e.g., `"cosine"` for cosine annealing). Default: `"cosine"`.
*   `gradient_clipping` (bool): Whether to clip gradients to prevent exploding gradients. Default: `true`.
*   `gradient_accumulation_steps` (int): The number of steps to accumulate gradients before an optimizer update. This simulates a larger effective batch size. Default: `4`.
*   `early_stopping_patience` (int): The number of epochs to wait for improvement before stopping training early. Default: `8`.
*   `early_stopping_metric` (str): The metric to monitor for early stopping (e.g., `"loss"`). Default: `"loss"`.

### Data Configuration
These parameters control how data is loaded and processed.

*   `use_real_ligo_data` (bool): Whether to use real GW150914 data from ReadLIGO. If `false`, synthetic data is generated. Default: `true`.
*   `num_samples` (int): The target number of samples to generate for the dataset. Default: `2000`.
*   `sequence_length` (int): The length of each input sequence (window size). Default: `256`.
*   `signal_noise_ratio` (float): The target signal-to-noise ratio for synthetic data generation. Default: `0.4`.

### Logging and Output Configuration
These parameters control where results are saved and how they are tracked.

*   `output_dir` (str): The directory where model checkpoints, logs, and results are saved. Default: `"outputs/complete_enhanced_training"`.
*   `use_wandb` (bool): Whether to enable Weights & Biases for experiment tracking. Default: `true`.
*   `use_tensorboard` (bool): Whether to enable TensorBoard for logging. Default: `true`.
*   `project_name` (str): The name of the W&B project. Default: `"cpc_snn_gw_complete_enhanced"`.
*   `log_every` (int): Log training metrics every N steps. Default: `10`.
*   `eval_every` (int): Evaluate on the test set every N steps. Default: `100`.
*   `save_every` (int): Save a model checkpoint every N steps. Default: `1000`.

## Overriding Configuration

Configuration parameters can be overridden in several ways:

1.  **Command-Line Arguments**: The most common method. Any parameter in the YAML file can be overridden by a command-line flag. For example:
    ```bash
    python cli.py --mode train --learning-rate 0.001 --num-epochs 100
    ```
    This will use a learning rate of 0.001 and train for 100 epochs, regardless of the values in the YAML file.

2.  **Custom YAML File**: Create a new YAML file (e.g., `my_experiment.yaml`) with your desired settings and pass it to the script:
    ```bash
    python cli.py --config configs/my_experiment.yaml
    ```

The system resolves configuration in the following order of precedence: command-line arguments > custom YAML file > default YAML file. This allows for maximum flexibility in experimentation.