"""
Comprehensive Training Base Class

Unified training interface for all CPC-SNN models with:
- W&B and TensorBoard integration
- Hydra CLI configuration system
- Checkpointing and resumption
- Comprehensive metrics tracking
- Memory optimization
- Professional logging and reporting
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    hydra = None
    DictConfig = None
    OmegaConf = None

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Comprehensive training configuration."""
    # Model configuration
    model_name: str = "cpc_snn_gw"
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    warmup_epochs: int = 5
    
    # Optimizer configuration
    optimizer: str = "adamw"
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduler configuration
    scheduler: str = "cosine"
    scheduler_config: Dict[str, Any] = field(default_factory=dict)
    
    # Data configuration
    data_config: Dict[str, Any] = field(default_factory=dict)
    
    # Logging and monitoring
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 1000
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Experiment tracking
    use_wandb: bool = True
    use_tensorboard: bool = True
    wandb_config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance optimization
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Checkpointing
    save_best_only: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Advanced features
    use_profiler: bool = False
    debug_mode: bool = False
    seed: int = 42


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int
    epoch: int
    loss: float
    accuracy: Optional[float] = None
    learning_rate: float = 0.0
    grad_norm: Optional[float] = None
    wall_time: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        metrics = {
            "step": self.step,
            "epoch": self.epoch,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "wall_time": self.wall_time
        }
        
        if self.accuracy is not None:
            metrics["accuracy"] = self.accuracy
        if self.grad_norm is not None:
            metrics["grad_norm"] = self.grad_norm
        
        metrics.update(self.custom_metrics)
        return metrics


class TrainerBase(ABC):
    """
    Abstract base class for all CPC-SNN trainers.
    
    Provides unified interface with:
    - Experiment tracking (W&B, TensorBoard)
    - Checkpointing and resumption
    - Metrics collection and reporting
    - Memory optimization
    - Professional logging
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize base trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.step_counter = 0
        self.epoch_counter = 0
        self.best_metric = float('inf')
        self.early_stopping_counter = 0
        
        # Setup directories
        self._setup_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Setup experiment tracking
        self._setup_experiment_tracking()
        
        # Setup checkpointing
        self._setup_checkpointing()
        
        # Training state
        self.train_state = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Metrics tracking
        self.training_metrics = []
        self.validation_metrics = []
        
        # Timing
        self.start_time = None
        self.epoch_start_time = None
    
    def _setup_directories(self):
        """Setup output directories."""
        self.output_dir = Path(self.config.output_dir)
        self.checkpoint_dir = self.output_dir / self.config.checkpoint_dir
        self.log_dir = self.output_dir / self.config.log_dir
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup professional logging."""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / 'training.log')
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Configure logger
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        logger.propagate = False
    
    def _setup_experiment_tracking(self):
        """Setup W&B and TensorBoard tracking."""
        self.wandb_run = None
        self.tensorboard_writer = None
        
        # Setup W&B
        if self.config.use_wandb and WANDB_AVAILABLE:
            try:
                wandb_config = {
                    "model_name": self.config.model_name,
                    "learning_rate": self.config.learning_rate,
                    "batch_size": self.config.batch_size,
                    "num_epochs": self.config.num_epochs,
                    **self.config.wandb_config
                }
                
                self.wandb_run = wandb.init(
                    project=self.config.model_name,
                    config=wandb_config,
                    dir=str(self.output_dir)
                )
                
                logger.info("W&B tracking initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
                self.wandb_run = None
        
        # Setup TensorBoard
        if self.config.use_tensorboard and TENSORBOARD_AVAILABLE:
            try:
                self.tensorboard_writer = SummaryWriter(
                    log_dir=str(self.log_dir / "tensorboard")
                )
                logger.info("TensorBoard tracking initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard: {e}")
                self.tensorboard_writer = None
    
    def _setup_checkpointing(self):
        """Setup checkpointing system."""
        self.checkpoint_manager = ocp.CheckpointManager(
            directory=str(self.checkpoint_dir),
            options=ocp.CheckpointManagerOptions(
                save_interval_steps=self.config.save_every,
                max_to_keep=5
            )
        )
    
    def create_optimizer(self) -> optax.GradientTransformation:
        """Create optimizer based on configuration."""
        if self.config.optimizer == "adamw":
            optimizer = optax.adamw(
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **self.config.optimizer_config
            )
        elif self.config.optimizer == "adam":
            optimizer = optax.adam(
                learning_rate=self.config.learning_rate,
                **self.config.optimizer_config
            )
        elif self.config.optimizer == "sgd":
            optimizer = optax.sgd(
                learning_rate=self.config.learning_rate,
                **self.config.optimizer_config
            )
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
                decay_steps=self.config.num_epochs,
                **self.config.scheduler_config
            )
        elif self.config.scheduler == "exponential":
            return optax.exponential_decay(
                init_value=self.config.learning_rate,
                **self.config.scheduler_config
            )
        elif self.config.scheduler == "warmup_cosine":
            warmup_steps = self.config.warmup_epochs
            total_steps = self.config.num_epochs
            
            return optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.config.learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=total_steps,
                **self.config.scheduler_config
            )
        else:
            return None
    
    def log_metrics(self, metrics: TrainingMetrics, prefix: str = "train"):
        """Log metrics to all tracking systems."""
        metrics_dict = metrics.to_dict()
        
        # Add prefix to metrics
        prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics_dict.items()}
        
        # Log to W&B
        if self.wandb_run:
            self.wandb_run.log(prefixed_metrics, step=metrics.step)
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            for key, value in prefixed_metrics.items():
                # Convert JAX arrays to Python scalars for TensorBoard compatibility
                if hasattr(value, 'item'):
                    scalar_value = float(value.item())
                elif isinstance(value, (jnp.ndarray, jnp.DeviceArray)):
                    scalar_value = float(value)
                else:
                    scalar_value = float(value)
                self.tensorboard_writer.add_scalar(key, scalar_value, metrics.step)
        
        # Log to console
        if metrics.step % self.config.log_every == 0:
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics_dict.items()])
            logger.info(f"Step {metrics.step}: {metric_str}")
    
    def save_checkpoint(self, train_state: train_state.TrainState, 
                       metrics: Optional[TrainingMetrics] = None, 
                       is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_data = {
            "train_state": train_state,
            "step": self.step_counter,
            "epoch": self.epoch_counter,
            "best_metric": self.best_metric,
            "config": self.config
        }
        
        if metrics:
            checkpoint_data["metrics"] = metrics.to_dict()
        
        # Save regular checkpoint
        self.checkpoint_manager.save(self.step_counter, checkpoint_data)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model"
            best_path.mkdir(exist_ok=True)
            
            # Save using orbax
            orbax_checkpointer = ocp.StandardCheckpointer()
            orbax_checkpointer.save(str(best_path), checkpoint_data)
            
            logger.info(f"Best model saved at step {self.step_counter}")
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """Load model checkpoint."""
        try:
            if checkpoint_path:
                # Load specific checkpoint
                orbax_checkpointer = ocp.StandardCheckpointer()
                checkpoint_data = orbax_checkpointer.restore(checkpoint_path)
            else:
                # Load latest checkpoint
                checkpoint_data = self.checkpoint_manager.restore(
                    self.checkpoint_manager.latest_step()
                )
            
            if checkpoint_data:
                self.train_state = checkpoint_data["train_state"]
                self.step_counter = checkpoint_data["step"]
                self.epoch_counter = checkpoint_data["epoch"]
                self.best_metric = checkpoint_data["best_metric"]
                
                logger.info(f"Checkpoint loaded from step {self.step_counter}")
                return True
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
        
        return False
    
    def should_stop_early(self, current_metric: float) -> bool:
        """Check if training should stop early."""
        if current_metric < self.best_metric - self.config.early_stopping_min_delta:
            self.best_metric = current_metric
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return self.early_stopping_counter >= self.config.early_stopping_patience
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        return {
            "total_steps": self.step_counter,
            "total_epochs": self.epoch_counter,
            "best_metric": self.best_metric,
            "config": self.config,
            "training_metrics": len(self.training_metrics),
            "validation_metrics": len(self.validation_metrics)
        }
    
    def save_training_summary(self):
        """Save training summary to file."""
        summary = self.get_training_summary()
        
        with open(self.output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("Training summary saved")
    
    def cleanup(self):
        """Clean up resources."""
        if self.wandb_run:
            self.wandb_run.finish()
        
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        logger.info("Training cleanup completed")
    
    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create model instance."""
        pass
    
    @abstractmethod
    def create_train_state(self, model: nn.Module, 
                          sample_input: jnp.ndarray) -> train_state.TrainState:
        """Create training state."""
        pass
    
    @abstractmethod
    def train_step(self, train_state: train_state.TrainState, 
                  batch: Any) -> Tuple[train_state.TrainState, TrainingMetrics]:
        """Single training step."""
        pass
    
    @abstractmethod
    def eval_step(self, train_state: train_state.TrainState, 
                 batch: Any) -> TrainingMetrics:
        """Single evaluation step."""
        pass
    
    def train(self, train_dataloader, val_dataloader=None):
        """Main training loop."""
        logger.info("Starting training...")
        self.start_time = time.time()
        
        # Create model and training state
        self.model = self.create_model()
        
        # Try to load checkpoint
        if not self.load_checkpoint():
            # Create new training state
            sample_batch = next(iter(train_dataloader))
            sample_input = sample_batch[0] if isinstance(sample_batch, tuple) else sample_batch
            self.train_state = self.create_train_state(self.model, sample_input)
        
        try:
            for epoch in range(self.epoch_counter, self.config.num_epochs):
                self.epoch_counter = epoch
                self.epoch_start_time = time.time()
                
                # Training epoch
                train_metrics = self._train_epoch(train_dataloader)
                
                # Validation epoch
                if val_dataloader and epoch % self.config.eval_every == 0:
                    val_metrics = self._eval_epoch(val_dataloader)
                    self.validation_metrics.append(val_metrics)
                    
                    # Check for early stopping
                    if self.should_stop_early(val_metrics.loss):
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                    
                    # Save best model
                    is_best = val_metrics.loss < self.best_metric
                    self.save_checkpoint(self.train_state, val_metrics, is_best)
                
                # Log epoch summary
                epoch_time = time.time() - self.epoch_start_time
                logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            self.save_training_summary()
            self.cleanup()
        
        total_time = time.time() - self.start_time
        logger.info(f"Training completed in {total_time:.2f}s")
    
    def _train_epoch(self, dataloader) -> TrainingMetrics:
        """Train for one epoch."""
        epoch_metrics = []
        
        for batch in dataloader:
            self.train_state, metrics = self.train_step(self.train_state, batch)
            epoch_metrics.append(metrics)
            
            # Log metrics
            self.log_metrics(metrics, "train")
            
            # Update counters
            self.step_counter += 1
            
            # Save checkpoint
            if self.step_counter % self.config.save_every == 0:
                self.save_checkpoint(self.train_state, metrics)
        
        # Compute epoch averages
        avg_loss = sum(m.loss for m in epoch_metrics) / len(epoch_metrics)
        avg_accuracy = None
        if all(m.accuracy is not None for m in epoch_metrics):
            avg_accuracy = sum(m.accuracy for m in epoch_metrics) / len(epoch_metrics)
        
        epoch_summary = TrainingMetrics(
            step=self.step_counter,
            epoch=self.epoch_counter,
            loss=avg_loss,
            accuracy=avg_accuracy,
            wall_time=time.time() - self.epoch_start_time
        )
        
        self.training_metrics.append(epoch_summary)
        return epoch_summary
    
    def _eval_epoch(self, dataloader) -> TrainingMetrics:
        """Evaluate for one epoch."""
        epoch_metrics = []
        
        for batch in dataloader:
            metrics = self.eval_step(self.train_state, batch)
            epoch_metrics.append(metrics)
        
        # Compute epoch averages
        avg_loss = sum(m.loss for m in epoch_metrics) / len(epoch_metrics)
        avg_accuracy = None
        if all(m.accuracy is not None for m in epoch_metrics):
            avg_accuracy = sum(m.accuracy for m in epoch_metrics) / len(epoch_metrics)
        
        epoch_summary = TrainingMetrics(
            step=self.step_counter,
            epoch=self.epoch_counter,
            loss=avg_loss,
            accuracy=avg_accuracy,
            wall_time=time.time() - self.epoch_start_time
        )
        
        self.log_metrics(epoch_summary, "val")
        return epoch_summary


class HydraTrainerMixin:
    """Mixin for Hydra CLI integration."""
    
    @staticmethod
    def create_config_from_hydra(cfg: DictConfig) -> TrainingConfig:
        """Create TrainingConfig from Hydra config."""
        if not HYDRA_AVAILABLE:
            raise ImportError("Hydra not available. Install with: pip install hydra-core")
        
        # Convert DictConfig to dict
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Create TrainingConfig
        return TrainingConfig(**config_dict)
    
    @staticmethod
    def save_hydra_config(cfg: DictConfig, output_dir: str):
        """Save Hydra config to output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "config.yaml", "w") as f:
            OmegaConf.save(cfg, f)


def create_hydra_cli_app(trainer_class: type, config_path: str = "configs", 
                        config_name: str = "config.yaml"):
    """
    Create Hydra CLI application for trainer.
    
    Args:
        trainer_class: Trainer class that inherits from TrainerBase
        config_path: Path to config directory
        config_name: Name of config file
    """
    if not HYDRA_AVAILABLE:
        raise ImportError("Hydra not available. Install with: pip install hydra-core")
    
    @hydra.main(version_base=None, config_path=config_path, config_name=config_name)
    def train_app(cfg: DictConfig):
        """Hydra CLI training application."""
        # Create training config
        training_config = HydraTrainerMixin.create_config_from_hydra(cfg)
        
        # Save config
        HydraTrainerMixin.save_hydra_config(cfg, training_config.output_dir)
        
        # Create trainer
        trainer = trainer_class(training_config)
        
        # Load data (implement in specific trainer)
        train_dataloader = trainer.create_train_dataloader()
        val_dataloader = trainer.create_val_dataloader()
        
        # Train
        trainer.train(train_dataloader, val_dataloader)
    
    return train_app


# Example implementation for CPC-SNN training
class CPCSNNTrainer(TrainerBase):
    """Example CPC-SNN trainer implementation."""
    
    def create_model(self) -> nn.Module:
        """Create CPC-SNN model."""
        from ..models import create_enhanced_cpc_encoder, create_enhanced_snn_classifier, create_optimized_spike_bridge
        
        # Create CPC encoder
        cpc_encoder = create_enhanced_cpc_encoder()
        
        # ✅ Create SpikeBridge - kluczowy element konwersji features->spikes
        spike_bridge = create_optimized_spike_bridge()
        
        # Create SNN classifier
        snn_classifier = create_enhanced_snn_classifier()
        
        # Create combined model with complete pipeline
        class CPCSNNModel(nn.Module):
            def setup(self):
                self.cpc_encoder = cpc_encoder
                self.spike_bridge = spike_bridge
                self.snn_classifier = snn_classifier
            
            @nn.compact
            def __call__(self, x, train: bool = True):
                # Generate PRNG key for SpikeBridge
                key = self.make_rng('dropout')  # Use dropout RNG stream
                
                # ✅ Complete pipeline: CPC -> SpikeBridge -> SNN
                latent_features = self.cpc_encoder(x, train=train)
                spikes = self.spike_bridge.encode(latent_features, key)
                output = self.snn_classifier(spikes, training=train)
                return output
        
        return CPCSNNModel()
    
    def create_train_state(self, model: nn.Module, 
                          sample_input: jnp.ndarray) -> train_state.TrainState:
        """Create training state."""
        # Initialize model parameters
        key = jax.random.PRNGKey(self.config.seed)
        variables = model.init(key, sample_input)
        
        # Create optimizer
        optimizer = self.create_optimizer()
        
        # Create training state
        return train_state.TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=optimizer
        )
    
    def train_step(self, train_state: train_state.TrainState, 
                  batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[train_state.TrainState, TrainingMetrics]:
        """Single training step."""
        inputs, labels = batch
        
        def loss_fn(params):
            logits = train_state.apply_fn({'params': params}, inputs)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            return jnp.mean(loss), logits
        
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        
        # Apply gradients
        train_state = train_state.apply_gradients(grads=grads)
        
        # Compute metrics
        accuracy = jnp.mean(jnp.argmax(logits, axis=1) == labels)
        # ✅ Poprawne obliczanie globalnej normy gradientu
        grad_norm = optax.global_norm(grads)
        
        metrics = TrainingMetrics(
            step=self.step_counter,
            epoch=self.epoch_counter,
            loss=float(loss),
            accuracy=float(accuracy),
            learning_rate=self.config.learning_rate,
            grad_norm=float(grad_norm)
        )
        
        return train_state, metrics
    
    def eval_step(self, train_state: train_state.TrainState, 
                 batch: Tuple[jnp.ndarray, jnp.ndarray]) -> TrainingMetrics:
        """Single evaluation step."""
        inputs, labels = batch
        
        # Forward pass
        logits = train_state.apply_fn({'params': train_state.params}, inputs)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        accuracy = jnp.mean(jnp.argmax(logits, axis=1) == labels)
        
        return TrainingMetrics(
            step=self.step_counter,
            epoch=self.epoch_counter,
            loss=float(loss),
            accuracy=float(accuracy)
        )
    
    def create_train_dataloader(self):
        """Create training dataloader using CPC-SNN-GW data pipeline."""
        from ..data import ContinuousGWGenerator, create_mixed_gw_dataset
        
        # Create continuous GW generator
        continuous_generator = ContinuousGWGenerator(
            base_frequency=50.0,
            freq_range=(40.0, 60.0),
            duration=4.0,
            include_doppler=True
        )
        
        # Generate training dataset
        key = jax.random.PRNGKey(self.config.seed)
        train_dataset = create_mixed_gw_dataset(
            continuous_generator=continuous_generator,
            binary_data=None,  # Could be enhanced with real GWOSC data
            mix_ratio=0.5,
            num_total_signals=self.config.batch_size * 10,  # 10 batches worth
            signal_duration=4.0,
            key=key
        )
        
        # Create simple dataloader generator
        def dataloader_generator():
            data = train_dataset['data']
            labels = train_dataset['labels']
            
            # Simple batching
            num_samples = len(data)
            indices = jnp.arange(num_samples)
            
            for i in range(0, num_samples, self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                if len(batch_indices) < self.config.batch_size:
                    continue  # Skip incomplete batches
                
                batch_data = jnp.array([data[j] for j in batch_indices])
                batch_labels = jnp.array([labels[j] for j in batch_indices])
                
                yield batch_data, batch_labels
        
        return dataloader_generator()
    
    def create_val_dataloader(self):
        """Create validation dataloader using CPC-SNN-GW data pipeline."""
        from ..data import ContinuousGWGenerator, create_mixed_gw_dataset
        
        # Create continuous GW generator
        continuous_generator = ContinuousGWGenerator(
            base_frequency=50.0,
            freq_range=(40.0, 60.0),
            duration=4.0,
            include_doppler=True
        )
        
        # Generate validation dataset with different seed
        key = jax.random.PRNGKey(self.config.seed + 1000)
        val_dataset = create_mixed_gw_dataset(
            continuous_generator=continuous_generator,
            binary_data=None,
            mix_ratio=0.5,
            num_total_signals=self.config.batch_size * 3,  # 3 batches worth
            signal_duration=4.0,
            key=key
        )
        
        # Create simple dataloader generator
        def dataloader_generator():
            data = val_dataset['data']
            labels = val_dataset['labels']
            
            # Simple batching
            num_samples = len(data)
            indices = jnp.arange(num_samples)
            
            for i in range(0, num_samples, self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                if len(batch_indices) < self.config.batch_size:
                    continue  # Skip incomplete batches
                
                batch_data = jnp.array([data[j] for j in batch_indices])
                batch_labels = jnp.array([labels[j] for j in batch_indices])
                
                yield batch_data, batch_labels
        
        return dataloader_generator()


# Factory functions
def create_training_config(**kwargs) -> TrainingConfig:
    """Create training configuration with overrides."""
    return TrainingConfig(**kwargs)


def create_cpc_snn_trainer(config: TrainingConfig) -> CPCSNNTrainer:
    """Create CPC-SNN trainer."""
    return CPCSNNTrainer(config)


def create_cpc_snn_cli_app(config_path: str = "configs", 
                          config_name: str = "training_config.yaml"):
    """Create CPC-SNN Hydra CLI application."""
    return create_hydra_cli_app(CPCSNNTrainer, config_path, config_name)


if __name__ == "__main__":
    # Example usage
    print("🚀 Training Base Class Implementation")
    print(f"W&B available: {WANDB_AVAILABLE}")
    print(f"TensorBoard available: {TENSORBOARD_AVAILABLE}")
    print(f"Hydra available: {HYDRA_AVAILABLE}")
    
    # Create example config
    config = create_training_config(
        model_name="test_cpc_snn",
        learning_rate=1e-4,
        batch_size=16,
        num_epochs=10,
        use_wandb=False,
        use_tensorboard=False
    )
    
    print("✅ Training configuration created successfully!")
    print(f"Model: {config.model_name}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}") 