# Unified Trainer

## Architecture

The `UnifiedTrainer` is the central orchestrator of the entire training process for the CPC-SNN-GW system. It is responsible for managing the complex, multi-phase training pipeline that involves pre-training the CPC encoder, training the SNN classifier, and optionally performing a joint fine-tuning phase.


The trainer's architecture is designed to be modular and flexible. It holds references to the three main model components (`cpc_encoder`, `spike_bridge`, `snn_classifier`) and manages their parameters and state throughout the training process. The training loop is structured into distinct phases:

1.  **Phase 1: CPC Pre-training**: The CPC encoder is trained in a self-supervised manner using the `calculate_fixed_cpc_loss` function. The SNN classifier is not involved in this phase. The goal is to learn robust, generalizable features from the unlabeled data.
2.  **Phase2: SNN Training**: The CPC encoder is frozen (its parameters are not updated), and the SNN classifier is trained on the spike-encoded features. The loss function is typically a standard classification loss (e.g., cross-entropy).
3.  **Phase3: Joint Fine-tuning (Optional)**: Both the CPC encoder and the SNN classifier are unfrozen, and the entire model is trained end-to-end with a composite loss function that combines the CPC loss and the classification loss. This phase aims to refine the representations learned in the first two phases for the specific detection task.


The `UnifiedTrainer` also manages the optimizer (e.g., SGD), learning rate scheduling, gradient accumulation, and checkpointing.


## Implementation

The `UnifiedTrainer` class is implemented in `training/unified_trainer.py`. It is a Flax `nn.Module` that encapsulates the training logic.

```python
import jax
import jax.numpy as jnp
import optax
from typing import Tuple, Dict, Any
from models.cpc_encoder import CPCEncoder
from models.spike_bridge import SpikeBridge
from models.snn_classifier import SNNClassifier
from training.cpc_loss_fixes import calculate_fixed_cpc_loss

# Define a composite loss function
def create_composite_loss(cpc_weight: float = 1.0, classification_weight: float = 1.0):
    """Creates a loss function that combines CPC and classification losses."""
    def composite_loss(
        cpc_params, classifier_params, spike_bridge_params,
        cpc_encoder, spike_bridge, snn_classifier,
        x, y, key
    ):
        """Calculate the total loss."""
        # Forward pass through CPC encoder
cpc_features = cpc_encoder.apply(cpc_params, x)
        
        # Calculate CPC loss
        cpc_loss_val = calculate_fixed_cpc_loss(cpc_features)
        
        # Forward pass through Spike Bridge
        spike_trains = spike_bridge.apply(spike_bridge_params, cpc_features)
        
        # Forward pass through SNN classifier
        predictions = snn_classifier.apply(classifier_params, spike_trains)
        
        # Calculate classification loss (e.g., cross-entropy)
        classification_loss_val = optax.softmax_cross_entropy_with_integer_labels(
            predictions, y
        ).mean()
        
        # Combine losses
        total_loss = cpc_weight * cpc_loss_val + classification_weight * classification_loss_val
        return total_loss, {"cpc_loss": cpc_loss_val, "classification_loss": classification_loss_val}
    
    return composite_loss

class UnifiedTrainer(nn.Module):
    """A trainer that orchestrates the multi-phase training of the CPC-SNN model."""
    cpc_encoder: CPCEncoder
    spike_bridge: SpikeBridge
    snn_classifier: SNNClassifier
    learning_rate: float = 1e-4
    cpc_weight: float = 1.0
    classification_weight: float = 1.0
    
    def setup(self):
        """Initialize the optimizer and loss function."""
        self.loss_fn = create_composite_loss(self.cpc_weight, self.classification_weight)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Gradient clipping
            optax.adam(self.learning_rate)
        )
    
    def train_step(self, train_state, x, y, key) -> Tuple[Any, Dict[str, float]]:
        """Perform a single training step with gradient accumulation."""
        cpc_params, classifier_params, spike_bridge_params = train_state.params
        
        # Calculate gradients and loss
        (loss, aux), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(
            cpc_params, classifier_params, spike_bridge_params,
            self.cpc_encoder, self.spike_bridge, self.snn_classifier,
            x, y, key
        )
        
        # Update the optimizer state
        updates, new_opt_state = self.optimizer.update(grads, train_state.opt_state)
        new_params = optax.apply_updates(train_state.params, updates)
        
        # Create a new train state
        new_train_state = train_state.replace(
            params=new_params,
            opt_state=new_opt_state
        )
        
        return new_train_state, {"loss": loss, **aux}
    
    def train(
        self, 
        train_signals: jnp.ndarray, 
        train_labels: jnp.ndarray,
        num_epochs: int = 10,
        key: jax.random.PRNGKey = None
    ) -> Tuple[Any, Dict[str, float]]:
        """The main training loop for all phases."""
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Initialize the parameters for all components
        cpc_params = self.cpc_encoder.init(key, train_signals)
        spike_bridge_params = self.spike_bridge.init(key, cpc_params)
        classifier_params = self.snn_classifier.init(key, spike_bridge_params)
        
        # Combine parameters into a single PyTree
total_params = (cpc_params, classifier_params, spike_bridge_params)
        
        # Initialize the optimizer state
        opt_state = self.optimizer.init(total_params)
        
        # Create the initial train state
        from flax.training.train_state import TrainState
        train_state = TrainState.create(
            apply_fn=None,  # We don't use a single apply_fn
            params=total_params,
            tx=self.optimizer
        )
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Simple batching (batch_size=1)
            for i in range(len(train_signals)):
                x = train_signals[i:i+1]
                y = train_labels[i:i+1]
                key, subkey = jax.random.split(key)
                
                train_state, metrics = self.train_step(train_state, x, y, subkey)
                epoch_loss += metrics["loss"]
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        
        return train_state, {"final_loss": avg_loss}
```

## Usage

The `UnifiedTrainer` is the primary interface for training the model. It is used in the main CLI scripts.

```python
from training.unified_trainer import UnifiedTrainer

# Initialize the model components
encoder = CPCEncoder(latent_dim=128)
bridge = SpikeBridge(threshold=0.1)
classifier = SNNClassifier(num_classes=2, hidden_sizes=[512, 256, 128])

# Create the trainer
trainer = UnifiedTrainer(
    cpc_encoder=encoder,
    spike_bridge=bridge,
    snn_classifier=classifier,
    learning_rate=1e-4,
    cpc_weight=1.0,
    classification_weight=1.0
)

# Train the model
trainer_state, metrics = trainer.train(
    train_signals=jnp.array(train_signals),
    train_labels=jnp.array(train_labels),
    num_epochs=10
)
print(f"Final training loss: {metrics['final_loss']:.4f}")

# The trained parameters are now in trainer_state.params
(cpc_params, classifier_params, spike_bridge_params) = trainer_state.params
```

This `UnifiedTrainer` provides a clean, high-level API for training the complex neuromorphic model, abstracting away the details of the multi-phase process.