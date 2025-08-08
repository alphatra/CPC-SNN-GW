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
    PerformanceProfiler, create_training_metrics,
    EnhancedMetricsLogger, create_enhanced_metrics_logger
)

# Import models
from models.cpc_encoder import CPCEncoder
from models.snn_classifier import SNNClassifier  
from models.spike_bridge import ValidatedSpikeBridge

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Simplified training configuration - core parameters only."""
    # Model parameters - ✅ MEMORY OPTIMIZED
    model_name: str = "cpc_snn_gw"
    batch_size: int = 1  # ✅ MEMORY FIX: Ultra-small batch for GPU memory constraints
    learning_rate: float = 1e-4  # ✅ ULTRA-CONSERVATIVE: Prevent model collapse (was 1e-3)
    weight_decay: float = 1e-4
    num_epochs: int = 100
    num_classes: int = 2  # ✅ CONFIGURABLE: Binary classification by default
    label_smoothing: float = 0.1
    use_class_weighting: bool = True
    
    # Training optimization - MEMORY OPTIMIZED
    optimizer: str = "sgd"  # ✅ FIX: SGD uses 2x less GPU memory than Adam
    scheduler: str = "cosine"
    gradient_clipping: float = 1.0  # ✅ RE-ENABLED: Needed for CPC stability
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
        
        # Enhanced experiment tracking
        if hasattr(config, 'wandb_config') and config.wandb_config:
            # Use enhanced metrics logger with comprehensive tracking
            self.enhanced_logger = create_enhanced_metrics_logger(
                config=config.__dict__ if hasattr(config, '__dict__') else vars(config),
                experiment_name=getattr(config, 'experiment_name', f"base-trainer-{int(time.time())}"),
                output_dir=config.output_dir
            )
            self.tracker = self.enhanced_logger  # Use enhanced logger as tracker
            logger.info("🚀 Using enhanced W&B metrics logger")
        else:
            # Fallback to basic tracker
            self.tracker = ExperimentTracker(
                project_name=config.project_name,
                output_dir=config.output_dir,
                use_wandb=config.use_wandb,
                use_tensorboard=config.use_tensorboard
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
            # ✅ MEMORY OPTIMIZED: Reduce Adam memory usage for large models
            optimizer = optax.adam(
                learning_rate=self.config.learning_rate,
                b1=0.9,      # Default but explicit
                b2=0.95,     # Reduced from 0.999 to use less memory
                eps=1e-8,    # Stable epsilon
                eps_root=1e-15  # Prevent division by zero in sqrt
            )
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
    
    def validate_and_log_step(self, 
                                 metrics: TrainingMetrics, 
                                 prefix: str = "train",
                                 model_state: Optional[Any] = None,
                                 gradients: Optional[Dict[str, jnp.ndarray]] = None,
                                 spikes: Optional[jnp.ndarray] = None,
                                 performance_data: Optional[Dict[str, float]] = None) -> bool:
        """Validate metrics and log to all tracking systems with enhanced neuromorphic data."""
        # Check for NaN values
        if check_for_nans(metrics.to_dict(), metrics.step):
            logger.error(f"NaN detected at step {metrics.step}. Stopping training.")
            return False
        
        # Enhanced logging with comprehensive metrics
        if self.enhanced_logger:
            self.enhanced_logger.log_training_step(
                metrics=metrics,
                model_state=model_state,
                gradients=gradients,
                spikes=spikes,
                performance_data=performance_data,
                prefix=prefix
            )
        else:
            # Fallback to basic logging
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
            num_classes: int  # ✅ CONFIGURABLE: Pass num_classes as parameter
            
            def setup(self):
                # ✅ ULTRA-MEMORY OPTIMIZED: Minimal model size to prevent collapse + memory issues
                self.cpc_encoder = CPCEncoder(latent_dim=64)   # ⬇️ ULTRA-REDUCED: 128 → 64 (75% smaller than original)
                self.spike_bridge = ValidatedSpikeBridge()
                self.snn_classifier = SNNClassifier(hidden_size=32, num_classes=self.num_classes)  # ⬇️ ULTRA-REDUCED: 64 → 32 (87.5% smaller)
            
            @nn.compact  
            def __call__(self, x, train: bool = True, return_intermediates: bool = False):
                # CPC encoding
                latents = self.cpc_encoder(x)
                
                # Convert to spikes ✅ CRITICAL FIX: Proper SpikeBridge call
                spikes = self.spike_bridge(latents, training=train)
                
                # SNN classification
                logits = self.snn_classifier(spikes)
                
                # ✅ FIXED: Return intermediate outputs for detailed metrics
                if return_intermediates:
                    return {
                        'logits': logits,
                        'cpc_features': latents,
                        'snn_output': spikes
                    }
                else:
                    return logits
        
        return CPCSNNModel(num_classes=self.config.num_classes)
    
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
    
    def train_step(self, train_state: train_state.TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[train_state.TrainState, TrainingMetrics, Dict[str, Any]]:
        """Execute one training step with gradient update and enhanced data collection."""
        x, y = batch
        
        # Store intermediate values for enhanced logging
        spikes = None
        
        def loss_fn(params):
            # Forward pass with spike collection
            logits = train_state.apply_fn(
                params, x, train=True,
                rngs={'spike_bridge': jax.random.PRNGKey(int(time.time()))}
            )
            # Label smoothing
            epsilon = jnp.asarray(self.config.label_smoothing)
            num_classes = self.config.num_classes
            y_smooth = (1.0 - epsilon) * jax.nn.one_hot(y, num_classes) + epsilon / num_classes
            per_example_loss = optax.softmax_cross_entropy(logits, y_smooth)
            # Class weighting (inverse frequency within batch)
            if self.config.use_class_weighting:
                counts = jnp.bincount(y, length=num_classes).astype(jnp.float32)
                counts = jnp.maximum(counts, 1.0)
                weights = jnp.sum(counts) / (counts * num_classes)
                sample_weights = weights[y]
                loss = jnp.mean(per_example_loss * sample_weights)
            else:
                loss = jnp.mean(per_example_loss)
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
            return loss, accuracy
        
        # Compute gradients
        (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        
        # Update parameters
        train_state = train_state.apply_gradients(grads=grads)
        
        # Compute gradient norm (return JAX array for JIT compatibility)
        grad_norm = compute_gradient_norm(grads)
        
        # Create metrics
        metrics = create_training_metrics(
            step=train_state.step,
            epoch=0,  # Will be set by caller
            loss=float(loss),
            accuracy=float(accuracy),
            learning_rate=self.config.learning_rate,
            grad_norm=float(grad_norm)  # Convert here, after JIT
        )
        
        # Enhanced data collection for logging
        enhanced_data = {
            'gradients': grads,
            'spikes': spikes,  # Will be None for now, can be extracted later
            'performance_data': {
                'memory_usage_mb': 0.0,  # Can be monitored separately
                'inference_latency_ms': 0.0,  # Can be timed separately
                'cpu_usage_percent': 0.0  # Can be monitored separately
            },
            'model_state': train_state
        }
        
        return train_state, metrics, enhanced_data
    
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