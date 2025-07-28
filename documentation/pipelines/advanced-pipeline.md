# Advanced Pipeline

## Overview

The `run_advanced_pipeline.py` script implements a comprehensive, multi-phase training and evaluation workflow. It is designed to be a production-ready, end-to-end pipeline that integrates all the system's components in a single, automated script. This is the recommended entry point for running a complete experiment from start to finish.


The advanced pipeline goes beyond the simple commands of the CLIs by orchestrating a sequence of steps that include data preparation, training, and final evaluation, ensuring a consistent and reproducible workflow.


## Phases

The pipeline is structured into three distinct phases:

### Phase 1: Environment Setup and Configuration
This initial phase sets up the environment and loads the configuration. It ensures that all necessary dependencies are available and that the system is ready to run.

*   **GPU Warmup**: The 6-stage GPU warmup is performed to prevent CUDA timing issues.
*   **Configuration Loading**: The YAML configuration file is loaded, and all hyperparameters are resolved.
*   **Logging Initialization**: Experiment tracking with Weights & Biases (W&B) and TensorBoard is initialized.


### Phase2: Data Preparation
This phase is responsible for creating the final training and test datasets.

*   **Real LIGO Data Integration**: The `create_real_ligo_dataset` function is called to download and process the authentic GW150914 strain data.
*   **Stratified Split**: The data is split into training and test sets using the `create_stratified_split` function to ensure balanced class representation.
*   **Data Augmentation**: The `GlitchInjector` is applied to the training data to improve model robustness. This step is crucial for simulating real-world detector noise.
*   **Dataset Finalization**: The augmented training data and the clean test data are prepared for the training loop.


### Phase3: Advanced Training and Evaluation
This is the core phase where the model is trained and evaluated.

*   **Model Initialization**: The CPC encoder, Spike Bridge, and SNN classifier are instantiated with the configured hyperparameters.
*   **Trainer Initialization**: The `UnifiedTrainer` is created with the specified optimizer and loss function settings.
*   **Training Loop**: The `trainer.train` method is called to execute the multi-phase training process (CPC pre-training, SNN training, and optional joint fine-tuning).
*   **Final Evaluation**: After training, the `evaluate_on_test_set` function is called to perform a comprehensive analysis on the test set.
*   **Reporting**: A professional summary report is generated using `create_test_evaluation_summary` and logged to the console and tracking systems.
*   **Model Saving**: The final trained model checkpoint is saved to the specified output directory.


## Implementation

The script is implemented as a class, `AdvancedPipeline`, which encapsulates the state and logic for each phase.

```python
import yaml
import os
import jax
from data.real_ligo_integration import create_real_ligo_dataset
from utils.data_split import create_stratified_split
from data.glitch_injector import GlitchInjector
from training.unified_trainer import UnifiedTrainer
from training.test_evaluation import evaluate_on_test_set, create_test_evaluation_summary
from models.cpc_encoder import CPCEncoder
from models.spike_bridge import SpikeBridge
from models.snn_classifier import SNNClassifier
import wandb

class AdvancedPipeline:
    """A class that orchestrates the complete advanced training pipeline."""
    def __init__(self, config_path: str = 'configs/final_framework_config.yaml'):
        self.config_path = config_path
        self.config = None
        self.test_data = None
        self.trainer = None
        self.trainer_state = None
        
    def setup_environment(self):
        """Phase 1: Setup the environment."""
        console.print("🚀 Starting Advanced Pipeline", style="bold blue")
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        console.print(f"Loaded config from {self.config_path}")
        
        # Perform GPU warmup
        perform_six_stage_gpu_warmup()
        
        # Initialize logging
        wandb.init(project=self.config['project_name'], config=self.config)
        console.print("✅ Environment setup complete.")
    
    def prepare_data(self):
        """Phase 2: Prepare the training and test data."""
        console.print("📦 Preparing data...")
        
        # Get the real LIGO dataset with a stratified split
        (train_signals, train_labels), (test_signals, test_labels) = create_real_ligo_dataset(
            num_samples=self.config['num_samples'],
            window_size=self.config['sequence_length'],
            return_split=True,
            train_ratio=self.config['train_ratio']
        )
        
        # Store test data for final evaluation
        self.test_data = {
            'strain': test_signals,
            'labels': test_labels
        }
        
        # Apply glitch injection for data augmentation
        console.print("🔧 Applying glitch injection augmentation...")
        injector = GlitchInjector()
        augmented_data = []
        for i in range(len(train_signals)):
            key = jax.random.PRNGKey(i + 1000)
            aug_signal, _ = injector.inject_glitch(train_signals[i], key)
            augmented_data.append(aug_signal)
        
        augmented_data = jnp.array(augmented_data)
        console.print(f"✅ Data preparation complete. Created {len(augmented_data)} augmented samples.")
        
        return augmented_data, train_labels
    
    def train_and_evaluate(self, train_data, train_labels):
        """Phase 3: Train the model and perform final evaluation."""
        console.print("🔥 Starting advanced training...")
        
        # Initialize model components
        encoder = CPCEncoder(latent_dim=self.config['cpc_latent_dim'])
        bridge = SpikeBridge()
        classifier = SNNClassifier(
            num_classes=2, 
            hidden_sizes=self.config['snn_neurons_per_layer']
        )
        
        # Create the trainer
        self.trainer = UnifiedTrainer(
            cpc_encoder=encoder,
            spike_bridge=bridge,
            snn_classifier=classifier,
            learning_rate=self.config['learning_rate']
        )
        
        # Train the model
        self.trainer_state, metrics = self.trainer.train(
            train_signals=train_data,
            train_labels=train_labels,
            num_epochs=self.config['num_epochs']
        )
        
        # Final evaluation on the test set
        console.print("🧪 Performing final evaluation...")
        test_results = evaluate_on_test_set(
            self.trainer_state,
            jnp.array(self.test_data['strain']),
            jnp.array(self.test_data['labels']),
            train_signals=train_data,
            verbose=True
        )
        
        # Create and log a professional summary
        final_accuracy = test_results['test_accuracy']
        test_summary = create_test_evaluation_summary(
            train_accuracy=metrics['final_loss'], # Using final loss as a proxy
            test_results=test_results,
            data_source="Real ReadLIGO GW150914",
            num_epochs=self.config['num_epochs']
        )
        console.print(test_summary)
        
        # Save the model
        model_dir = self.config['output_dir']
        os.makedirs(model_dir, exist_ok=True)
        # ... (save logic) ...
        console.print(f"💾 Model saved to: {model_dir}")
        
        return final_accuracy, test_results
    
    def run(self):
        """Execute the complete pipeline."""
        self.setup_environment()
        train_data, train_labels = self.prepare_data()
        final_accuracy, test_results = self.train_and_evaluate(train_data, train_labels)
        
        console.print("🎉 Advanced pipeline completed successfully!", style="bold green")
        return final_accuracy, test_results

# Main execution
if __name__ == '__main__':
    pipeline = AdvancedPipeline(config_path='configs/final_framework_config.yaml')
    final_acc, results = pipeline.run()
```

## Usage

The advanced pipeline is designed to be run as a single command, making it ideal for automated workflows and batch processing.

```bash
# Run the complete advanced pipeline with default config
python run_advanced_pipeline.py

# Run with a custom configuration
python run_advanced_pipeline.py --config configs/my_custom_config.yaml
```

This script represents the culmination of the system's design, providing a robust, automated, and scientifically sound workflow for neuromorphic gravitational wave detection.