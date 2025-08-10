# Usage Examples

## Command-Line Interface (CLI)

The system provides a powerful command-line interface (CLI) for training, evaluation, and inference. The main entry points are `cli.py` and `enhanced_cli.py`.


### Training a Model
To start a complete training run with the default configuration:
```bash
python cli.py --mode train
```

To train with specific parameters, such as a different number of epochs and a real LIGO data source:
```bash
python cli.py --mode train --data-source real_ligo --epochs 50 --batch-size 1
```

The `enhanced_cli.py` script offers additional features. To run training with enhanced logging and gradient accumulation:
```bash
python enhanced_cli.py --mode train --use-gradient-accumulation --accumulation-steps 4
```

### Evaluating a Trained Model
To evaluate a model on the test set and generate a comprehensive report:
```bash
python cli.py --mode eval --report scientific
```
This command will load the latest trained model, run it on the test set, and print a detailed summary including accuracy, sensitivity, specificity, and model collapse detection.

### Performing Inference
To use a trained model to make predictions on new, unseen data:
```bash
python cli.py --mode infer --input-file new_strain_data.npy
```
This will load the model and the specified input file, perform inference, and output the predicted class (0 for noise, 1 for signal) and confidence.

---

## Advanced Pipeline

The `run_advanced_pipeline.py` script executes a multi-phase training workflow, integrating data preparation, training, and evaluation in a single command.

```bash
python run_advanced_pipeline.py
```
This script will automatically:
1.  Prepare the real LIGO dataset with data augmentation.
2.  Execute the training phase with the configured number of epochs.
3.  Perform a final evaluation and generate a professional report.

You can customize the pipeline by passing arguments, such as `--config configs/custom_config.yaml` to use a different configuration file.

---

## Python API

For programmatic use, the system can be imported and used as a Python library.

### Loading and Preparing Data
```python
from data.real_ligo_integration import create_real_ligo_dataset

# Create a dataset with a stratified train/test split
(train_signals, train_labels), (test_signals, test_labels) = create_real_ligo_dataset(
    num_samples=1200,
    window_size=512,
    return_split=True,
    train_ratio=0.8
)
print(f"Training set: {train_signals.shape[0]} samples")
print(f"Test set: {test_signals.shape[0]} samples")
```

### Training a Model
```python
from training.unified_trainer import UnifiedTrainer
from models.cpc_encoder import CPCEncoder
from models.snn_classifier import SNNClassifier
from models.spike_bridge import SpikeBridge
import jax.numpy as jnp

# Initialize the model components
encoder = CPCEncoder(latent_dim=128)
bridge = SpikeBridge()
classifier = SNNClassifier(num_classes=2)

# Create the trainer
trainer = UnifiedTrainer(
    cpc_encoder=encoder,
    spike_bridge=bridge,
    snn_classifier=classifier,
    learning_rate=1e-4
)

# Train the model
trainer_state, metrics = trainer.train(
    train_signals=jnp.array(train_signals),
    train_labels=jnp.array(train_labels),
    num_epochs=10
)
print(f"Final training accuracy: {metrics['accuracy']:.4f}")
```

### Evaluating a Model
```python
from training.test_evaluation import evaluate_on_test_set

# Evaluate on the test set
test_results = evaluate_on_test_set(
    trainer_state=trainer_state,
    test_signals=jnp.array(test_signals),
    test_labels=jnp.array(test_labels),
    verbose=True
)

print(f"Test Accuracy: {test_results['test_accuracy']:.4f}")
print(f"Model Collapse: {test_results['model_collapse']}")
print(f"F1-Score: {test_results['f1_score']:.4f}")
```

### Making Predictions
```python
# Make a prediction on a single new signal
new_signal = jnp.array(test_signals[0:1])  # Example: first test sample
logits = trainer_state.apply_fn(trainer_state.params, new_signal, train=False)
prediction = jnp.argmax(logits, axis=-1)[0]
print(f"Prediction: {'Signal' if prediction == 1 else 'Noise'}")
```

These examples demonstrate the flexibility of the system, allowing users to interact with it at different levels of abstraction, from simple command-line commands to fine-grained control via the Python API.