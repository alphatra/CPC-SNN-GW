"""
Base Trainer: Abstract Training Interface

Clean abstract base class for all CPC-SNN trainers with:
- Unified training interface and lifecycle management
- Modular configuration via utility modules
- Professional experiment tracking integration
- Comprehensive error handling and validation
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

# Import local utilities
from .training_utils import (
    setup_professional_logging, setup_directories, optimize_jax_for_device,
    validate_config, save_config_to_file, compute_gradient_norm, 
    check_for_nans, ProgressTracker
)
from .training_metrics import (
    TrainingMetrics, ExperimentTracker, EarlyStoppingMonitor,
    PerformanceProfiler, create_training_metrics
)

# Import models
from models.cpc_encoder import CPCEncoder
from models.snn_classifier import SNNClassifier  
from models.spike_bridge import ValidatedSpikeBridge

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Simplified training configuration - core parameters only."""
    # Model parameters
    model_name: str = "cpc_snn_gw"
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    
    # Training optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_clipping: float = 1.0
    mixed_precision: bool = True
    
    # Monitoring
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 1000
    
    # Paths and experiment tracking
    output_dir: str = "outputs"
    use_wandb: bool = True
    use_tensorboard: bool = True
    project_name: str = "cpc-snn-gw"
    
    # Performance
    max_memory_gb: float = 8.0
    enable_profiling: bool = False
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "loss"
    early_stopping_mode: str = "min"


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
        
        # Experiment tracking
        self.tracker = ExperimentTracker(
            project_name=config.project_name,
            output_dir=config.output_dir,
            use_wandb=config.use_wandb,
            use_tensorboard=config.use_tensorboard
        )
        
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
        
        # Training state
        self.train_state = None
        self.start_time = None
        
        # Save configuration
        save_config_to_file(config, str(self.directories['output'] / 'config.json'))
        self.tracker.log_hyperparameters(config.__dict__)
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {config.model_name}")
    
    def create_optimizer(self) -> optax.GradientTransformation:
        """Create optimizer with specified configuration."""
        if self.config.optimizer == "adamw":
            optimizer = optax.adamw(
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adam":
            optimizer = optax.adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == "sgd":
            optimizer = optax.sgd(learning_rate=self.config.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Add gradient clipping
        if self.config.gradient_clipping > 0:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.gradient_clipping),
                optimizer
            )
        
        return optimizer
    
    def create_scheduler(self) -> Optional[optax.Schedule]:
        """Create learning rate scheduler."""
        if self.config.scheduler == "cosine":
            return optax.cosine_decay_schedule(
                init_value=self.config.learning_rate,
                decay_steps=self.config.num_epochs
            )
        elif self.config.scheduler == "linear":
            return optax.linear_schedule(
                init_value=self.config.learning_rate,
                end_value=0.0,
                transition_steps=self.config.num_epochs
            )
        else:
            return None
    
    def validate_and_log_step(self, metrics: TrainingMetrics, prefix: str = "train") -> bool:
        """Validate metrics and log to all tracking systems."""
        # Check for NaN values
        if check_for_nans(metrics.to_dict(), metrics.step):
            logger.error(f"NaN detected at step {metrics.step}. Stopping training.")
            return False
        
        # Log to experiment tracker
        self.tracker.log_metrics(metrics, prefix)
        
        # Update progress
        if hasattr(self, 'progress_tracker'):
            self.progress_tracker.update(metrics.step, metrics.to_dict())
        
        return True
    
    def should_stop_training(self, metrics: TrainingMetrics) -> bool:
        """Check if training should stop early."""
        metric_value = getattr(metrics, self.config.early_stopping_metric, metrics.loss)
        return self.early_stopping.update(
            metric_value, 
            metrics.epoch, 
            self.train_state.params if self.train_state else None
        )
    
    def cleanup(self):
        """Clean up resources and finalize tracking."""
        if self.tracker:
            self.tracker.finish()
        
        if self.profiler:
            perf_summary = self.profiler.get_summary()
            logger.info(f"Performance summary: {perf_summary}")
    
    # Abstract methods that subclasses must implement
    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create the neural network model."""
        pass
    
    @abstractmethod
    def create_train_state(self, model: nn.Module, sample_input: jnp.ndarray) -> train_state.TrainState:
        """Create training state with model, optimizer, and initial parameters."""
        pass
    
    @abstractmethod
    def train_step(self, train_state: train_state.TrainState, batch: Any) -> Tuple[train_state.TrainState, TrainingMetrics]:
        """Execute single training step."""
        pass
    
    @abstractmethod
    def eval_step(self, train_state: train_state.TrainState, batch: Any) -> TrainingMetrics:
        """Execute single evaluation step."""
        pass


class CPCSNNTrainer(TrainerBase):
    """
    Concrete implementation for CPC+SNN training pipeline.
    
    Implements the full neuromorphic pipeline with:
    - CPC encoder for representation learning
    - Spike bridge for continuous-to-spike conversion
    - SNN classifier for final prediction
    """
    
    def create_model(self) -> nn.Module:
        """Create CPC+SNN model architecture."""
        
        class CPCSNNModel(nn.Module):
            """Complete CPC+SNN pipeline model."""
            
            def setup(self):
                self.cpc_encoder = CPCEncoder(latent_dim=256)
                self.spike_bridge = ValidatedSpikeBridge()
                self.snn_classifier = SNNClassifier(hidden_size=128, num_classes=3)
            
            @nn.compact  
            def __call__(self, x, train: bool = True):
                # CPC encoding
                latents = self.cpc_encoder(x)
                
                # Convert to spikes
                key = self.make_rng('spike_bridge') if train else jax.random.PRNGKey(42)
                spikes = self.spike_bridge(latents, key)
                
                # SNN classification
                logits = self.snn_classifier(spikes)
                
                return logits
        
        return CPCSNNModel()
    
    def create_train_state(self, model: nn.Module, sample_input: jnp.ndarray) -> train_state.TrainState:
        """Initialize training state with model parameters."""
        key = jax.random.PRNGKey(42)
        params = model.init({'params': key, 'spike_bridge': key}, sample_input, train=True)
        
        optimizer = self.create_optimizer()
        
        return train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer
        )
    
    def train_step(self, train_state: train_state.TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[train_state.TrainState, TrainingMetrics]:
        """Execute one training step with gradient update."""
        x, y = batch
        
        def loss_fn(params):
            logits = train_state.apply_fn(
                params, x, train=True, 
                rngs={'spike_bridge': jax.random.PRNGKey(int(time.time()))}
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
            return loss, accuracy
        
        # Compute gradients
        (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        
        # Update parameters
        train_state = train_state.apply_gradients(grads=grads)
        
        # Compute gradient norm
        grad_norm = compute_gradient_norm(grads)
        
        # Create metrics
        metrics = create_training_metrics(
            step=train_state.step,
            epoch=0,  # Will be set by caller
            loss=float(loss),
            accuracy=float(accuracy),
            learning_rate=self.config.learning_rate,
            grad_norm=float(grad_norm)
        )
        
        return train_state, metrics
    
    def eval_step(self, train_state: train_state.TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> TrainingMetrics:
        """Execute one evaluation step."""
        x, y = batch
        
        # Forward pass without gradients
        logits = train_state.apply_fn(
            train_state.params, x, train=False,
            rngs={'spike_bridge': jax.random.PRNGKey(42)}
        )
        
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        
        metrics = create_training_metrics(
            step=train_state.step,
            epoch=0,
            loss=float(loss), 
            accuracy=float(accuracy)
        )
        
        return metrics


def create_cpc_snn_trainer(config: TrainingConfig) -> CPCSNNTrainer:
    """Factory function to create CPC-SNN trainer."""
    validate_config(config, ['model_name', 'batch_size', 'learning_rate', 'num_epochs'])
    return CPCSNNTrainer(config) 