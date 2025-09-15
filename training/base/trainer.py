"""
Base trainer implementations.

This module contains base trainer classes extracted from
base_trainer.py for better modularity.

Split from base_trainer.py for better maintainability.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state, checkpoints

from .config import TrainingConfig
from ..utils import (
    setup_professional_logging, setup_directories, optimize_jax_for_device,
    validate_config, save_config_to_file, compute_gradient_norm, 
    check_for_nans, ProgressTracker
)
from ..monitoring import (
    TrainingMetrics, ExperimentTracker, EarlyStoppingMonitor,
    PerformanceProfiler, create_training_metrics,
    EnhancedMetricsLogger, create_enhanced_metrics_logger
)

# Import models
from models.cpc import RealCPCEncoder, RealCPCConfig
from models.snn.core import SNNClassifier
from models.bridge.core import ValidatedSpikeBridge

logger = logging.getLogger(__name__)


class TrainerBase(ABC):
    """
    Abstract base class for all CPC-SNN trainers.
    
    Provides unified interface and common functionality while allowing
    specialized implementations for different training strategies.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Setup infrastructure
        self.directories = setup_directories(config.output_dir)
        self.logger = setup_professional_logging(
            log_file=str(self.directories['log'] / 'training.log')
        )
        
        # Device optimization
        self.device_info = optimize_jax_for_device()
        
        # Enhanced experiment tracking
        if hasattr(config, 'wandb_config') and getattr(config, 'wandb_config', None):
            # Use enhanced metrics logger with comprehensive tracking
            self.enhanced_logger = create_enhanced_metrics_logger(
                config=config.__dict__ if hasattr(config, '__dict__') else vars(config),
                experiment_name=getattr(config, 'experiment_name', f"base-trainer-{getattr(config, 'seed', 42)}"),
                output_dir=config.output_dir
            )
            self.tracker = self.enhanced_logger  # Use enhanced logger as tracker
            logger.info("🚀 Using enhanced W&B metrics logger")
        else:
            # Fallback to basic tracker
            self.tracker = ExperimentTracker(
                experiment_name=config.project_name,
                output_dir=config.output_dir
            )
            self.enhanced_logger = None
            logger.info("Using basic experiment tracker")
        
        # Monitoring utilities
        self.early_stopping = EarlyStoppingMonitor(
            patience=config.early_stopping_patience,
            metric_name=config.early_stopping_metric,
            mode=config.early_stopping_mode
        )
        
        if config.enable_profiling:
            self.profiler = PerformanceProfiler()
        else:
            self.profiler = None
        
        # Validate configuration
        if not validate_config(config):
            raise ValueError("Invalid training configuration")
        
        # Save configuration
        save_config_to_file(config, self.directories['config'] / 'config.json')
        
        logger.info(f"TrainerBase initialized: {config.model_name}")
    
    @abstractmethod
    def create_model(self):
        """Create model architecture - must be implemented by subclasses."""
        pass
    
    @abstractmethod  
    def train_step(self, train_state, batch):
        """Execute single training step - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def eval_step(self, train_state, batch):
        """Execute single evaluation step - must be implemented by subclasses."""
        pass
    
    def save_checkpoint(self, train_state, epoch: int, metrics: Optional[Dict] = None):
        """Save training checkpoint."""
        checkpoint_dir = self.directories['checkpoints']
        
        # Save training state
        checkpoints.save_checkpoint(
            ckpt_dir=checkpoint_dir,
            target=train_state,
            step=epoch,
            overwrite=True
        )
        
        # Save additional metadata
        if metrics:
            metadata_file = checkpoint_dir / f'epoch_{epoch}_metrics.json'
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        logger.info(f"Checkpoint saved at epoch {epoch}")
    
    def load_checkpoint(self, train_state, checkpoint_path: Optional[str] = None):
        """Load training checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.directories['checkpoints']
        
        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_path,
            target=train_state
        )
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return restored_state


class CPCSNNTrainer(TrainerBase):
    """
    Standard CPC+SNN trainer implementation.
    
    Implements basic CPC encoder + spike bridge + SNN classifier pipeline
    with standard training and evaluation procedures.
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        
        # Model components will be created lazily
        self.cpc_encoder = None
        self.spike_bridge = None
        self.snn_classifier = None
        self.model = None
        
        logger.info("CPCSNNTrainer initialized")
    
    def create_model(self):
        """Create standard CPC+SNN model."""
        if self.model is not None:
            return self.model
        
        # Create model components
        cpc_config = RealCPCConfig(
            latent_dim=256,
            context_length=64,
            prediction_steps=self.config.cpc_prediction_steps,
            temperature=self.config.cpc_temperature
        )
        
        self.cpc_encoder = RealCPCEncoder(config=cpc_config)
        
        self.spike_bridge = ValidatedSpikeBridge(
            spike_encoding="temporal_contrast",
            time_steps=self.config.spike_time_steps,
            threshold=self.config.spike_threshold,
            surrogate_type=self.config.spike_surrogate_type,
            surrogate_beta=self.config.spike_surrogate_beta
        )
        
        self.snn_classifier = SNNClassifier(
            hidden_size=self.config.snn_hidden_sizes[0],
            num_classes=self.config.num_classes,
            num_layers=self.config.snn_num_layers
        )
        
        # Create combined model
        class StandardCPCSNNModel(nn.Module):
            """Standard CPC+SNN model."""
            
            def setup(self):
                self.cpc = cpc_encoder
                self.bridge = spike_bridge  
                self.snn = snn_classifier
            
            def __call__(self, x, training=True):
                cpc_features = self.cpc(x, training=training)
                # Reduce feature dimension for SpikeBridge (expects [batch, time])
                spike_in = jnp.mean(cpc_features, axis=-1)
                spikes = self.bridge(spike_in, training=training)
                logits = self.snn(spikes, training=training)
                return logits
        
        # Store references for individual component access
        cpc_encoder = self.cpc_encoder
        spike_bridge = self.spike_bridge
        snn_classifier = self.snn_classifier
        
        self.model = StandardCPCSNNModel()
        
        logger.info("✅ Standard CPC+SNN model created")
        return self.model
    
    def train_step(self, train_state, batch):
        """Execute single training step."""
        signals, labels = batch
        
        def loss_fn(params):
            logits = train_state.apply_fn(
                params, signals, training=True,
                rngs={'dropout': jax.random.PRNGKey(2)}
            )
            
            # Classification loss
            if self.config.use_focal_loss:
                # Focal loss for class imbalance
                probs = jax.nn.softmax(logits, axis=-1)
                ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
                pt = probs[jnp.arange(len(labels)), labels]
                focal_weight = (1 - pt) ** self.config.focal_gamma
                loss = jnp.mean(focal_weight * ce_loss)
            else:
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            
            # Compute accuracy
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
            
            return loss, accuracy
        
        # Compute gradients
        (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        
        # Apply gradients
        train_state = train_state.apply_gradients(grads=grads)
        
        # Create metrics
        metrics = create_training_metrics(
            step=train_state.step,
            epoch=getattr(self, 'current_epoch', 0),
            loss=float(loss),
            accuracy=float(accuracy)
        )
        
        return train_state, metrics
    
    def eval_step(self, train_state, batch):
        """Execute single evaluation step."""
        signals, labels = batch
        
        # Forward pass without gradients
        logits = train_state.apply_fn(train_state.params, signals, training=False)
        
        # Compute metrics
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
        
        return create_training_metrics(
            step=train_state.step,
            epoch=getattr(self, 'current_epoch', 0),
            loss=float(loss),
            accuracy=float(accuracy)
        )

    def train(self, train_signals: jnp.ndarray, train_labels: jnp.ndarray,
              test_signals: jnp.ndarray, test_labels: jnp.ndarray) -> Dict[str, Any]:
        """Simple training loop using provided arrays (data-only dependency)."""
        model = self.create_model()
        # Initialize train state with RNGs (for dropout)
        sample_input = train_signals[:1]
        init_rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}
        params = model.init(init_rngs, sample_input, training=True)
        tx = optax.adam(self.config.learning_rate)
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        
        batch_size = self.config.batch_size
        num_epochs = self.config.num_epochs
        num_samples = len(train_signals)
        steps_per_epoch = max(1, (num_samples + batch_size - 1) // batch_size)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            # Naive epoch loop
            for step in range(steps_per_epoch):
                s = step * batch_size
                e = min(s + batch_size, num_samples)
                batch = (train_signals[s:e], train_labels[s:e])
                state, metrics = self.train_step(state, batch)
            
            # Eval at end of epoch
            eval_batch = (test_signals[:batch_size], test_labels[:batch_size])
            _ = self.eval_step(state, eval_batch)
        
        # Final simple evaluation over entire test set in batches
        correct = 0
        total = 0
        for s in range(0, len(test_signals), batch_size):
            e = min(s + batch_size, len(test_signals))
            batch = (test_signals[s:e], test_labels[s:e])
            logits = state.apply_fn(state.params, batch[0], training=False)
            preds = jnp.argmax(logits, axis=-1)
            correct += int(jnp.sum(preds == batch[1]))
            total += (e - s)
        test_accuracy = correct / max(1, total)
        
        return {
            'success': True,
            'test_accuracy': float(test_accuracy),
            'epochs_completed': num_epochs
        }

# Export trainer classes
__all__ = [
    "TrainerBase",
    "CPCSNNTrainer"
]
