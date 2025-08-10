# Main CLI

## Overview

The `cli.py` script is the primary command-line interface (CLI) for the CPC-SNN-GW system. It provides a simple, user-friendly way to interact with the model for training, evaluation, and inference without writing any code. It is designed to be the go-to tool for most users.


The CLI is built using the `argparse` module and provides a clear help message when run with the `--help` flag. It orchestrates the entire pipeline by calling the appropriate functions from the `training`, `data`, and `utils` modules based on the provided command-line arguments.


A key feature of the main CLI is the automatic application of the **6-stage GPU warmup**. This is a critical step for ensuring stable and efficient GPU performance, and it is performed at the very beginning of any command that requires GPU computation.


## Commands

The CLI supports three main commands, specified by the `--mode` argument:

### 1. `train`
This command initiates the complete training pipeline.

**Usage**:
```bash
python cli.py --mode train [OPTIONS]
```

**Key Options**:
*   `--data-source`: The source of the data (`real_ligo` or `synthetic`). Default: `real_ligo`.
*   `--epochs`: The number of training epochs. Default: `10`.
*   `--batch-size`: The batch size for training. Default: `1`.
*   `--config`: Path to a custom YAML configuration file.

**Example**:
```bash
# Train with real LIGO data for 50 epochs
python cli.py --mode train --data-source real_ligo --epochs 50
```

### 2. `eval`
This command evaluates a trained model on the test set and generates a comprehensive report.

**Usage**:
```bash
python cli.py --mode eval [OPTIONS]
```

**Key Options**:
*   `--report`: The type of report to generate (`scientific` or `brief`). Default: `scientific`.
*   `--model-path`: Path to a specific trained model checkpoint (if not using the latest).

**Example**:
```bash
# Evaluate the model with a full scientific report
python cli.py --mode eval --report scientific
```

### 3. `infer`
This command performs inference on new, unseen data to make a detection prediction.

**Usage**:
```bash
python cli.py --mode infer --input-file PATH_TO_FILE [OPTIONS]
```

**Key Options**:
*   `--input-file`: The path to a `.npy` file containing the strain data to be classified.
*   `--model-path`: Path to the trained model to use for inference.

**Example**:
```bash
# Classify a new signal from a file
python cli.py --mode infer --input-file /path/to/new_signal.npy
```

## Implementation

The core of the CLI is the `main` function in `cli.py`. It parses the arguments and dispatches to the appropriate function.

```python
import argparse
import jax
from data.real_ligo_integration import create_real_ligo_dataset
from training.unified_trainer import UnifiedTrainer
from training.test_evaluation import evaluate_on_test_set
from models.cpc_encoder import CPCEncoder
from models.spike_bridge import SpikeBridge
from models.snn_classifier import SNNClassifier
import yaml
import os

def perform_six_stage_gpu_warmup():
    """A function that performs the 6-stage GPU warmup (implementation omitted for brevity)."""
    print("🔥 Performing 6-stage GPU warmup...")
    # ... (code from techContext.md) ...
    print("✅ GPU warmup complete.")

def main():
    parser = argparse.ArgumentParser(description='CPC-SNN-GW: Neuromorphic Gravitational Wave Detector')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval', 'infer'],
                        help='The mode to run: train, eval, or infer')
    parser.add_argument('--config', type=str, default='configs/final_framework_config.yaml',
                        help='Path to the YAML configuration file')
    
    # Training-specific arguments
    parser.add_argument('--data-source', type=str, default='real_ligo', choices=['real_ligo', 'synthetic'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    
    # Evaluation-specific arguments
    parser.add_argument('--report', type=str, default='scientific', choices=['scientific', 'brief'])
    parser.add_argument('--model-path', type=str, default=None)
    
    # Inference-specific arguments
    parser.add_argument('--input-file', type=str, default=None)
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure we are using a GPU
    print(f"JAX devices: {jax.devices()}")
    
    # Perform the critical 6-stage GPU warmup
    perform_six_stage_gpu_warmup()
    
    if args.mode == 'train':
        # Load data
        (train_signals, train_labels), (test_signals, test_labels) = create_real_ligo_dataset(
            num_samples=config.get('num_samples', 1000),
            window_size=config.get('sequence_length', 256),
            return_split=True,
            train_ratio=config.get('train_ratio', 0.8)
        )
        
        # Initialize model components
        encoder = CPCEncoder(latent_dim=config['cpc_latent_dim'])
        bridge = SpikeBridge()
        classifier = SNNClassifier(num_classes=2, hidden_sizes=config['snn_neurons_per_layer'])
        
        # Create and run the trainer
        trainer = UnifiedTrainer(
            cpc_encoder=encoder,
            spike_bridge=bridge,
            snn_classifier=classifier,
            learning_rate=config['learning_rate']
        )
        trainer_state, metrics = trainer.train(
            train_signals=train_signals,
            train_labels=train_labels,
            num_epochs=args.epochs
        )
        
        # Save the model
        model_dir = config['output_dir']
        os.makedirs(model_dir, exist_ok=True)
        # ... (save logic) ...
        
    elif args.mode == 'eval':
        # ... (load model and test set, then call evaluate_on_test_set) ...
        pass
        
    elif args.mode == 'infer':
        # ... (load model and input file, then perform inference) ...
        pass

if __name__ == '__main__':
    main()
```

## Usage

The main CLI is designed to be simple and intuitive. Users can start with the default settings and gradually customize their runs with command-line flags.

```bash
# Get help
python cli.py --help

# Start a default training run
python cli.py --mode train

# Evaluate the trained model
python cli.py --mode eval --report scientific
```

This CLI serves as the primary entry point for users, providing a robust and well-optimized interface to the powerful underlying system.