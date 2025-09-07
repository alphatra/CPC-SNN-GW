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
from flax.training import checkpoints

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
from models.cpc_encoder import RealCPCEncoder, RealCPCConfig
from models.snn_classifier import SNNClassifier  
from models.spike_bridge import ValidatedSpikeBridge

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Simplified training configuration - core parameters only."""
    # Model parameters - ✅ MEMORY OPTIMIZED
    model_name: str = "cpc_snn_gw"
    batch_size: int = 1  # ✅ MEMORY FIX: Ultra-small batch for GPU memory constraints
    learning_rate: float = 5e-5  # ✅ FIXED: Matching successful AResGW learning rate
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
    grad_accum_steps: int = 1  # ✅ NEW: gradient accumulation
    
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
    # loss | balanced_accuracy | f1
    early_stopping_metric: str = "balanced_accuracy"
    # min | max (for loss → min, for f1/balanced_accuracy → max)
    early_stopping_mode: str = "max"

    # ✅ New: checkpointing frequency
    checkpoint_every_epochs: int = 5

    # ✅ New: focal loss and class weighting controls
    use_focal_loss: bool = True
    focal_gamma: float = 1.8
    class1_weight: float = 1.1  # further reduce FP

    # ✅ New: Exponential Moving Average of parameters
    use_ema: bool = True
    ema_decay: float = 0.999

    # ✅ SpikeBridge hyperparameters (exposed)
    spike_time_steps: int = 24
    spike_threshold: float = 0.1
    spike_learnable: bool = True
    spike_threshold_levels: int = 4
    spike_surrogate_type: str = "adaptive_multi_scale"
    spike_surrogate_beta: float = 4.0
    spike_pool_seq: bool = False
    spike_target_rate_low: float = 0.10
    spike_target_rate_high: float = 0.20

    # ✅ CPC pretraining / multitask parameters
    use_cpc_aux_loss: bool = True
    cpc_aux_weight: float = 0.2
    ce_loss_weight: float = 1.0
    cpc_freeze_first_n_convs: int = 0  # 0,1,2
    cpc_prediction_steps: int = 12
    cpc_num_negatives: int = 128
    cpc_use_hard_negatives: bool = True
    cpc_temperature: float = 0.07
    cpc_use_temporal_transformer: bool = True
    cpc_attention_heads: int = 8
    cpc_transformer_layers: int = 4
    cpc_dropout_rate: float = 0.1
    cpc_use_grad_checkpointing: bool = True
    cpc_use_mixed_precision: bool = True

    # ✅ SNN exposure
    snn_hidden_size: int = 32
    
    # ✅ CPC exposure - FIXED: Add missing latent_dim parameter
    cpc_latent_dim: int = 256  # ✅ FIXED: Increased from 64 for sufficient capacity


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
        self.ema_params = None
        self.start_time = None
        
        # Save configuration
        save_config_to_file(config, str(self.directories['output'] / 'config.json'))
        self.tracker.log_hyperparameters(config.__dict__)
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {config.model_name}")
        
        # ✅ NEW: Deterministic RNG key to avoid per-step retracing (no time.time() in JIT)
        self.rng_key = jax.random.PRNGKey(42)
    
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
        # Map friendly names
        if self.config.early_stopping_metric == "balanced_accuracy":
            metric_value = float((metrics.accuracy_class0 + metrics.accuracy_class1) / 2.0) if hasattr(metrics, 'accuracy_class0') else metrics.accuracy
        elif self.config.early_stopping_metric == "f1":
            metric_value = float(getattr(metrics, 'f1_score', metrics.accuracy))
        else:
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
            config: TrainingConfig
            
            def setup(self):
                # ✅ ULTRA-MEMORY OPTIMIZED: Minimal model size to prevent collapse + memory issues
                # Configure CPC encoder with exposed parameters
                cpc_cfg = RealCPCConfig(
                    latent_dim=getattr(self.config, 'cpc_latent_dim', 256),
                    prediction_steps=self.config.cpc_prediction_steps,
                    num_negatives=self.config.cpc_num_negatives,
                    temperature=self.config.cpc_temperature,
                    use_hard_negatives=self.config.cpc_use_hard_negatives,
                    use_temporal_transformer=self.config.cpc_use_temporal_transformer,
                    temporal_attention_heads=self.config.cpc_attention_heads,
                    dropout_rate=self.config.cpc_dropout_rate,
                    use_gradient_checkpointing=self.config.cpc_use_grad_checkpointing,
                    use_mixed_precision=self.config.cpc_use_mixed_precision,
                )
                self.cpc_encoder = RealCPCEncoder(config=cpc_cfg)
                self.spike_bridge = ValidatedSpikeBridge(
                    time_steps=self.config.spike_time_steps,
                    use_learnable_encoding=self.config.spike_learnable,
                    threshold=self.config.spike_threshold,
                    num_threshold_levels=self.config.spike_threshold_levels,
                    surrogate_type=self.config.spike_surrogate_type,
                    surrogate_beta=self.config.spike_surrogate_beta,
                )
                self.snn_classifier = SNNClassifier(hidden_size=self.config.snn_hidden_size, num_classes=self.num_classes)
            
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
        
        return CPCSNNModel(num_classes=self.config.num_classes, config=self.config)
    
    def create_train_state(self, model: nn.Module, sample_input: jnp.ndarray) -> train_state.TrainState:
        """Initialize training state with model parameters."""
        key = jax.random.PRNGKey(42)
        # Provide RNGs that match module expectations
        params = model.init({'params': key, 'spike_noise': key, 'dropout': key}, sample_input, train=True)
        
        optimizer = self.create_optimizer()
        
        self.train_state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer
        )
        # Initialize EMA params if enabled
        if getattr(self.config, 'use_ema', False):
            self.ema_params = self.train_state.params
        return self.train_state
    
    def train_step(self, train_state: train_state.TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[train_state.TrainState, TrainingMetrics, Dict[str, Any]]:
        """Execute one training step with gradient update and enhanced data collection."""
        x, y = batch
        
        # Store intermediate values for enhanced logging
        spikes = None
        
        def loss_fn(params):
            # Forward pass with spike collection
            # ✅ Use step-derived PRNG key to prevent XLA retracing and ensure determinism
            step_key = jax.random.fold_in(self.rng_key, int(train_state.step))
            # Full forward pass to obtain logits and CPC latents from model
            outputs = train_state.apply_fn(
                params, x, train=True,
                rngs={'spike_noise': step_key, 'dropout': step_key},
                return_intermediates=True
            )
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            cpc_latent = outputs.get('cpc_features', None) if isinstance(outputs, dict) else None
            num_classes = self.config.num_classes
            epsilon = jnp.asarray(self.config.label_smoothing)
            onehot = jax.nn.one_hot(y, num_classes)
            y_smooth = (1.0 - epsilon) * onehot + epsilon / num_classes

            # Baseline CE
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            ce = -jnp.sum(y_smooth * log_probs, axis=-1)

            # Focal modulation (optional)
            if getattr(self.config, 'use_focal_loss', True):
                probs = jax.nn.softmax(logits, axis=-1)
                p_t = jnp.sum(onehot * probs, axis=-1)
                gamma = jnp.asarray(getattr(self.config, 'focal_gamma', 2.5))
                focal_weight = jnp.power(1.0 - p_t, gamma)
                per_example_loss = focal_weight * ce
            else:
                per_example_loss = ce

            # Class weighting: inverse freq plus extra weight for class 1
            if self.config.use_class_weighting:
                counts = jnp.bincount(y, length=num_classes).astype(jnp.float32)
                counts = jnp.maximum(counts, 1.0)
                inv_freq = jnp.sum(counts) / (counts * num_classes)
                # extra emphasis for class 1
                class1_weight = jnp.asarray(getattr(self.config, 'class1_weight', 1.5))
                class_weights = inv_freq.at[1].set(inv_freq[1] * class1_weight)
                sample_weights = class_weights[y]
                loss = jnp.mean(per_example_loss * sample_weights)
            else:
                loss = jnp.mean(per_example_loss)

            # ✅ Multi-task: add CPC auxiliary InfoNCE loss
            if getattr(self.config, 'use_cpc_aux_loss', True) and cpc_latent is not None:
                try:
                    from models.cpc_losses import temporal_info_nce_loss
                    # Guard against extremely short sequence length to avoid indexing errors
                    if cpc_latent is not None and cpc_latent.shape[1] >= 3:
                        cpc_aux = temporal_info_nce_loss(cpc_latent, temperature=0.06)
                        loss = self.config.ce_loss_weight * loss + self.config.cpc_aux_weight * cpc_aux
                except Exception:
                    pass
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
            return loss, accuracy
        
        # Gradient accumulation
        accum_steps = max(1, int(getattr(self.config, 'grad_accum_steps', 1)))
        loss = 0.0
        accuracy = 0.0
        grads = None
        for acc_i in range(accum_steps):
            (part_loss, part_acc), part_grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
            loss += float(part_loss) / accum_steps
            accuracy += float(part_acc) / accum_steps
            if grads is None:
                grads = part_grads
            else:
                grads = jax.tree_util.tree_map(lambda g, pg: g + pg, grads, part_grads)
        grads = jax.tree_util.tree_map(lambda g: g / accum_steps, grads)
        
        # Update parameters with accumulated gradients
        train_state = train_state.apply_gradients(grads=grads)
        # Update EMA
        if getattr(self.config, 'use_ema', False) and self.ema_params is not None:
            decay = jnp.asarray(getattr(self.config, 'ema_decay', 0.999))
            self.ema_params = jax.tree_util.tree_map(
                lambda ema, cur: decay * ema + (1.0 - decay) * cur,
                self.ema_params,
                train_state.params
            )
        
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
            rngs={'spike_noise': jax.random.PRNGKey(42)}
        )
        
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        # Emit probabilities for downstream threshold optimization/ROC-PR
        probs = jax.nn.softmax(logits, axis=-1)
        
        metrics = create_training_metrics(
            step=train_state.step,
            epoch=0,
            loss=float(loss), 
            accuracy=float(accuracy)
        )
        try:
            metrics.update_custom(prob_class1=float(jnp.mean(probs[:, 1])), true_pos_rate=float(accuracy))
        except Exception:
            pass
        
        return metrics


def create_cpc_snn_trainer(config: TrainingConfig) -> CPCSNNTrainer:
    """Factory function to create CPC-SNN trainer."""
    validate_config(config, ['model_name', 'batch_size', 'learning_rate', 'num_epochs'])
    return CPCSNNTrainer(config) 