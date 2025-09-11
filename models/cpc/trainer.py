"""
CPC training utilities and trainer class.

This module contains CPC training logic extracted from
cpc_encoder.py for better modularity.

Split from cpc_encoder.py for better maintainability.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from .config import RealCPCConfig
from .core import RealCPCEncoder

logger = logging.getLogger(__name__)


class CPCTrainer:
    """
    Trainer for CPC (Contrastive Predictive Coding) models.
    
    Handles:
    - Self-supervised CPC training with InfoNCE loss
    - Learning rate scheduling
    - Gradient optimization
    - Training monitoring and logging
    """
    
    def __init__(self, config: RealCPCConfig):
        """
        Initialize CPC trainer.
        
        Args:
            config: CPC configuration
        """
        self.config = config
        self.train_state = None
        self.training_metrics = []
        
        # Validate configuration
        if not config.validate():
            raise ValueError("Invalid CPC configuration provided")
        
        logger.info(f"Initialized CPCTrainer with config: {config}")
    
    def create_model(self) -> RealCPCEncoder:
        """Create CPC encoder model."""
        return RealCPCEncoder(config=self.config)
    
    def create_train_state(self, model: RealCPCEncoder, sample_input: jnp.ndarray) -> train_state.TrainState:
        """
        Create training state for CPC model.
        
        Args:
            model: CPC encoder model
            sample_input: Sample input for parameter initialization
            
        Returns:
            Initialized training state
        """
        # Initialize parameters
        key = jax.random.PRNGKey(42)
        variables = model.init(key, sample_input, training=True)
        
        # Create optimizer with schedule
        if self.config.scheduler == "cosine":
            schedule = optax.cosine_decay_schedule(
                init_value=self.config.learning_rate,
                decay_steps=self.config.num_epochs * 100,  # Estimate
                alpha=0.1
            )
        elif self.config.scheduler == "linear":
            schedule = optax.linear_schedule(
                init_value=self.config.learning_rate,
                end_value=self.config.learning_rate * 0.1,
                transition_steps=self.config.num_epochs * 100
            )
        else:
            schedule = self.config.learning_rate  # Constant
        
        # Create optimizer based on config
        if self.config.optimizer == "adamw":
            optimizer = optax.adamw(
                learning_rate=schedule,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adam":
            optimizer = optax.adam(learning_rate=schedule)
        else:  # sgd
            optimizer = optax.sgd(learning_rate=schedule)
        
        # Add gradient clipping
        if self.config.gradient_clipping > 0:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.gradient_clipping),
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
        Single training step for CPC.
        
        Args:
            batch: Batch of sequences (x, _) - labels ignored for self-supervised learning
            
        Returns:
            Training metrics
        """
        x, _ = batch  # Ignore labels in self-supervised learning
        
        def loss_fn(params):
            # Forward pass
            latent_features = self.train_state.apply_fn(params, x, training=True)
            
            # InfoNCE loss (simplified)
            # Context: all but last time step
            # Targets: all but first time step  
            context = latent_features[:, :-1, :]
            targets = latent_features[:, 1:, :]
            
            # Compute InfoNCE loss
            batch_size, time_steps, latent_dim = context.shape
            
            if time_steps > 0:
                # Reshape for contrastive learning
                context_flat = context.reshape(-1, latent_dim)
                targets_flat = targets.reshape(-1, latent_dim)
                
                # L2 normalize
                context_norm = context_flat / (jnp.linalg.norm(context_flat, axis=-1, keepdims=True) + 1e-8)
                targets_norm = targets_flat / (jnp.linalg.norm(targets_flat, axis=-1, keepdims=True) + 1e-8)
                
                # Similarity matrix
                similarity = jnp.dot(context_norm, targets_norm.T) / self.config.temperature
                
                # InfoNCE loss
                num_samples = similarity.shape[0]
                labels = jnp.arange(num_samples)
                
                log_softmax = jax.nn.log_softmax(similarity, axis=1)
                loss = -jnp.mean(log_softmax[labels, labels])
            else:
                loss = jnp.array(0.0)
            
            return loss
        
        # Compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(self.train_state.params)
        
        # Apply gradients
        self.train_state = self.train_state.apply_gradients(grads=grads)
        
        # Compute metrics
        metrics = {
            'loss': float(loss),
            'step': int(self.train_state.step),
            'learning_rate': float(self.get_current_lr())
        }
        
        self.training_metrics.append(metrics)
        
        return metrics
    
    def get_current_lr(self) -> float:
        """Get current learning rate from optimizer state."""
        try:
            # Extract learning rate from optimizer state
            if hasattr(self.train_state.opt_state, 'hyperparams'):
                return float(self.train_state.opt_state.hyperparams['learning_rate'])
            else:
                return self.config.learning_rate
        except:
            return self.config.learning_rate
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save training checkpoint."""
        import pickle
        
        checkpoint_data = {
            'params': self.train_state.params,
            'opt_state': self.train_state.opt_state,
            'step': self.train_state.step,
            'config': self.config,
            'metrics': self.training_metrics
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Saved CPC checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        import pickle
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Restore training state
        self.train_state = self.train_state.replace(
            params=checkpoint_data['params'],
            opt_state=checkpoint_data['opt_state'],
            step=checkpoint_data['step']
        )
        
        self.training_metrics = checkpoint_data.get('metrics', [])
        
        logger.info(f"Loaded CPC checkpoint from {checkpoint_path}")


# Export trainer class
__all__ = [
    "CPCTrainer"
]

