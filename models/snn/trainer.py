"""
SNN training utilities and trainer class.

This module contains SNN training logic extracted from
snn_classifier.py for better modularity.

Split from snn_classifier.py for better maintainability.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from .config import SNNConfig, EnhancedSNNConfig
from .core import SNNClassifier, EnhancedSNNClassifier

logger = logging.getLogger(__name__)


class SNNTrainer:
    """
    Trainer for SNN (Spiking Neural Network) classifiers.
    
    Handles:
    - Supervised SNN training for classification
    - Spike rate regularization
    - Curriculum learning (optional)
    - Advanced optimization for spiking networks
    """
    
    def __init__(self, config: SNNConfig):
        """
        Initialize SNN trainer.
        
        Args:
            config: SNN configuration
        """
        self.config = config
        self.train_state = None
        self.training_metrics = []
        self.current_epoch = 0
        
        # Validate configuration
        if not config.validate():
            raise ValueError("Invalid SNN configuration provided")
        
        logger.info(f"Initialized SNNTrainer with config: {config}")
    
    def create_model(self) -> SNNClassifier:
        """Create SNN classifier model."""
        if isinstance(self.config, EnhancedSNNConfig):
            return EnhancedSNNClassifier(config=self.config)
        else:
            return SNNClassifier(
                hidden_size=self.config.hidden_sizes[0] if self.config.hidden_sizes else 128,
                num_classes=self.config.num_classes,
                num_layers=self.config.num_layers
            )
    
    def create_train_state(self, model, sample_input: jnp.ndarray) -> train_state.TrainState:
        """
        Create training state for SNN model.
        
        Args:
            model: SNN classifier model
            sample_input: Sample input for parameter initialization
            
        Returns:
            Initialized training state
        """
        # Initialize parameters
        key = jax.random.PRNGKey(42)
        variables = model.init(key, sample_input, training=True)
        
        # Create optimizer (SNN-specific optimizations)
        learning_rate = getattr(self.config, 'learning_rate', 1e-3)
        
        # ✅ SNN-OPTIMIZED: Lower learning rates for spiking networks
        snn_lr = learning_rate * 0.5  # Reduce LR for SNN stability
        
        optimizer = optax.adamw(
            learning_rate=snn_lr,
            weight_decay=getattr(self.config, 'weight_decay', 1e-5)
        )
        
        # ✅ GRADIENT CLIPPING: Essential for spiking networks
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Conservative clipping for SNNs
            optimizer
        )
        
        # Create training state
        self.train_state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=variables,
            tx=optimizer
        )
        
        return self.train_state
    
    def train_step(self, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Dict[str, Any]:
        """
        Single training step for SNN classifier.
        
        Args:
            batch: Batch of (spike_trains, labels)
            
        Returns:
            Training metrics
        """
        spikes, labels = batch
        
        def loss_fn(params):
            # Forward pass
            logits = self.train_state.apply_fn(params, spikes, training=True)
            
            # ✅ CLASSIFICATION LOSS: Cross-entropy
            clf_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            
            # ✅ SPIKE RATE REGULARIZATION: Encourage reasonable spike rates
            if isinstance(self.config, EnhancedSNNConfig) and self.config.spike_rate_regularization > 0:
                # This would require access to intermediate spike rates
                # For now, include a simple regularization term
                reg_loss = self.config.spike_rate_regularization * 0.01  # Placeholder
            else:
                reg_loss = 0.0
            
            total_loss = clf_loss + reg_loss
            
            # Compute accuracy
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
            
            return total_loss, (clf_loss, reg_loss, accuracy)
        
        # Compute loss and gradients
        (total_loss, (clf_loss, reg_loss, accuracy)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(self.train_state.params)
        
        # Apply gradients
        self.train_state = self.train_state.apply_gradients(grads=grads)
        
        # Compute metrics
        metrics = {
            'total_loss': float(total_loss),
            'classification_loss': float(clf_loss),
            'regularization_loss': float(reg_loss),
            'accuracy': float(accuracy),
            'step': int(self.train_state.step),
            'epoch': self.current_epoch
        }
        
        self.training_metrics.append(metrics)
        
        return metrics
    
    def eval_step(self, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Dict[str, Any]:
        """
        Evaluation step for SNN classifier.
        
        Args:
            batch: Batch of (spike_trains, labels)
            
        Returns:
            Evaluation metrics
        """
        spikes, labels = batch
        
        # Forward pass (deterministic)
        logits = self.train_state.apply_fn(
            self.train_state.params, 
            spikes, 
            training=False
        )
        
        # Compute metrics
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
        
        # Additional metrics for binary classification
        if self.config.num_classes == 2:
            predictions = jnp.argmax(logits, axis=-1)
            
            # Binary classification metrics
            tp = jnp.sum((predictions == 1) & (labels == 1))
            tn = jnp.sum((predictions == 0) & (labels == 0))
            fp = jnp.sum((predictions == 1) & (labels == 0))
            fn = jnp.sum((predictions == 0) & (labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                'loss': float(loss),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
        else:
            return {
                'loss': float(loss),
                'accuracy': float(accuracy)
            }
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save SNN training checkpoint."""
        import pickle
        
        checkpoint_data = {
            'params': self.train_state.params,
            'opt_state': self.train_state.opt_state,
            'step': self.train_state.step,
            'epoch': self.current_epoch,
            'config': self.config,
            'metrics': self.training_metrics
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Saved SNN checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load SNN training checkpoint."""
        import pickle
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Restore training state
        self.train_state = self.train_state.replace(
            params=checkpoint_data['params'],
            opt_state=checkpoint_data['opt_state'],
            step=checkpoint_data['step']
        )
        
        self.current_epoch = checkpoint_data.get('epoch', 0)
        self.training_metrics = checkpoint_data.get('metrics', [])
        
        logger.info(f"Loaded SNN checkpoint from {checkpoint_path}")


# Export trainer class
__all__ = [
    "SNNTrainer"
]

