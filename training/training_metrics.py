"""
Training Metrics: Monitoring and Experiment Tracking

Comprehensive metrics and monitoring infrastructure:
- TrainingMetrics dataclass with standard metrics
- Weights & Biases integration
- TensorBoard integration  
- Early stopping with configurable patience
- Performance monitoring and profiling
- Real-time visualization utilities
"""

import time
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import jax.numpy as jnp
import numpy as np

# Optional dependencies with fallbacks
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

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """
    Standard container for training metrics across all trainers.
    
    Provides consistent interface for logging, monitoring, and comparison.
    """
    step: int
    epoch: int
    loss: float
    accuracy: Optional[float] = None
    learning_rate: float = 0.0
    grad_norm: Optional[float] = None
    wall_time: float = 0.0
    
    # Model-specific metrics
    cpc_loss: Optional[float] = None
    snn_loss: Optional[float] = None
    spike_rate: Optional[float] = None
    
    # Performance metrics
    throughput: Optional[float] = None  # samples/second
    memory_usage: Optional[float] = None  # GB
    
    # Custom metrics dictionary
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
        
        # Add optional metrics if present
        optional_fields = [
            'accuracy', 'grad_norm', 'cpc_loss', 'snn_loss', 
            'spike_rate', 'throughput', 'memory_usage'
        ]
        
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                metrics[field_name] = value
        
        # Add custom metrics
        metrics.update(self.custom_metrics)
        
        return metrics
    
    def update_custom(self, **kwargs) -> None:
        """Update custom metrics."""
        self.custom_metrics.update(kwargs)


class ExperimentTracker:
    """
    Unified experiment tracking with support for multiple backends.
    
    Handles W&B, TensorBoard, and local JSON logging simultaneously.
    """
    
    def __init__(self, 
                 project_name: str = "cpc-snn-gw",
                 experiment_name: Optional[str] = None,
                 output_dir: str = "outputs",
                 use_wandb: bool = True,
                 use_tensorboard: bool = True,
                 wandb_config: Optional[Dict[str, Any]] = None,
                 tags: Optional[List[str]] = None):
        
        self.project_name = project_name
        self.experiment_name = experiment_name or f"experiment_{int(time.time())}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize backends
        self.wandb_run = None
        self.tensorboard_writer = None
        self.metrics_history = []
        
        # Setup W&B
        if use_wandb and WANDB_AVAILABLE:
            try:
                self.wandb_run = wandb.init(
                    project=project_name,
                    name=self.experiment_name,
                    config=wandb_config or {},
                    tags=tags or [],
                    dir=str(self.output_dir)
                )
                logger.info("W&B tracking initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
                self.wandb_run = None
        
        # Setup TensorBoard
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            try:
                tb_dir = self.output_dir / "tensorboard"
                tb_dir.mkdir(exist_ok=True)
                self.tensorboard_writer = SummaryWriter(log_dir=str(tb_dir))
                logger.info(f"TensorBoard logging to: {tb_dir}")
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard: {e}")
                self.tensorboard_writer = None
    
    def log_metrics(self, metrics: TrainingMetrics, prefix: str = "train") -> None:
        """
        Log metrics to all configured backends.
        
        Args:
            metrics: TrainingMetrics object
            prefix: Prefix for metric names (train/val/test)
        """
        metrics_dict = metrics.to_dict()
        step = metrics.step
        
        # Add prefix to metric names
        prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics_dict.items() 
                           if k not in ['step', 'epoch']}
        prefixed_metrics.update({
            'step': step,
            'epoch': metrics.epoch
        })
        
        # Log to W&B
        if self.wandb_run:
            try:
                self.wandb_run.log(prefixed_metrics, step=step)
            except Exception as e:
                logger.warning(f"W&B logging failed: {e}")
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            try:
                for key, value in prefixed_metrics.items():
                    if isinstance(value, (int, float)) and key not in ['step', 'epoch']:
                        self.tensorboard_writer.add_scalar(key, value, step)
                self.tensorboard_writer.flush()
            except Exception as e:
                logger.warning(f"TensorBoard logging failed: {e}")
        
        # Save to local history
        self.metrics_history.append({
            'timestamp': time.time(),
            'prefix': prefix,
            **prefixed_metrics
        })
        
        # Periodically save metrics to JSON
        if len(self.metrics_history) % 100 == 0:
            self._save_metrics_history()
    
    def log_hyperparameters(self, config: Dict[str, Any]) -> None:
        """Log hyperparameters to tracking systems."""
        if self.wandb_run:
            try:
                self.wandb_run.config.update(config)
            except Exception as e:
                logger.warning(f"W&B config update failed: {e}")
        
        if self.tensorboard_writer:
            try:
                # Convert config to string representation for TensorBoard
                config_str = json.dumps(config, indent=2, default=str)
                self.tensorboard_writer.add_text("hyperparameters", config_str, 0)
            except Exception as e:
                logger.warning(f"TensorBoard hyperparameter logging failed: {e}")
    
    def log_model_summary(self, model_info: Dict[str, Any]) -> None:
        """Log model architecture and parameter count."""
        if self.wandb_run:
            try:
                self.wandb_run.summary.update(model_info)
            except Exception as e:
                logger.warning(f"W&B model summary failed: {e}")
        
        # Save model info locally
        model_info_path = self.output_dir / "model_info.json"
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
    
    def _save_metrics_history(self) -> None:
        """Save metrics history to JSON file."""
        history_path = self.output_dir / "metrics_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
    
    def finish(self) -> None:
        """Clean up and finish experiment tracking."""
        # Save final metrics
        self._save_metrics_history()
        
        # Close W&B
        if self.wandb_run:
            try:
                self.wandb_run.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.warning(f"W&B finish failed: {e}")
        
        # Close TensorBoard
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.close()
                logger.info("TensorBoard writer closed")
            except Exception as e:
                logger.warning(f"TensorBoard close failed: {e}")


class EarlyStoppingMonitor:
    """
    Early stopping with configurable patience and monitoring criteria.
    
    Supports multiple metrics and custom improvement criteria.
    """
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 metric_name: str = "loss",
                 mode: str = "min",
                 restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            metric_name: Metric to monitor for early stopping
            mode: 'min' for decreasing metrics, 'max' for increasing
            restore_best_weights: Whether to restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.wait_count = 0
        self.best_weights = None
        
        self.is_better = self._get_comparison_fn()
    
    def _get_comparison_fn(self):
        """Get comparison function based on mode."""
        if self.mode == 'min':
            return lambda current, best: current < best - self.min_delta
        else:
            return lambda current, best: current > best + self.min_delta
    
    def update(self, current_value: float, epoch: int, model_weights=None) -> bool:
        """
        Update early stopping monitor with current metric value.
        
        Args:
            current_value: Current value of monitored metric
            epoch: Current epoch number
            model_weights: Current model weights (for restoration)
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait_count = 0
            
            if self.restore_best_weights and model_weights is not None:
                self.best_weights = model_weights
            
            logger.info(f"New best {self.metric_name}: {current_value:.6f} at epoch {epoch}")
            return False
        else:
            self.wait_count += 1
            logger.debug(f"No improvement for {self.wait_count}/{self.patience} epochs")
            
            if self.wait_count >= self.patience:
                logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
                logger.info(f"Best {self.metric_name}: {self.best_value:.6f} at epoch {self.best_epoch}")
                return True
            
            return False
    
    def get_best_weights(self):
        """Return best weights if available."""
        return self.best_weights


class PerformanceProfiler:
    """
    Performance profiling utility for training optimization.
    
    Tracks timing, memory usage, and throughput metrics.
    """
    
    def __init__(self):
        self.timings = {}
        self.memory_snapshots = []
        self.throughput_history = []
    
    def start_timer(self, name: str) -> None:
        """Start timing a section."""
        self.timings[name] = {'start': time.perf_counter()}
    
    def end_timer(self, name: str) -> float:
        """End timing and return elapsed time."""
        if name not in self.timings:
            logger.warning(f"Timer '{name}' not started")
            return 0.0
        
        elapsed = time.perf_counter() - self.timings[name]['start']
        self.timings[name]['elapsed'] = elapsed
        return elapsed
    
    def record_throughput(self, samples_processed: int, time_elapsed: float) -> float:
        """Record throughput measurement."""
        throughput = samples_processed / time_elapsed if time_elapsed > 0 else 0.0
        self.throughput_history.append({
            'timestamp': time.time(),
            'samples': samples_processed,
            'time': time_elapsed,
            'throughput': throughput
        })
        return throughput
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {
            'timings': {name: data.get('elapsed', 0.0) 
                       for name, data in self.timings.items()},
            'average_throughput': np.mean([t['throughput'] for t in self.throughput_history]) 
                                if self.throughput_history else 0.0,
            'peak_throughput': max([t['throughput'] for t in self.throughput_history], default=0.0)
        }
        return summary


def create_training_metrics(step: int, 
                          epoch: int,
                          loss: float,
                          **kwargs) -> TrainingMetrics:
    """
    Convenience function to create TrainingMetrics with common values.
    
    Args:
        step: Training step
        epoch: Training epoch  
        loss: Training loss
        **kwargs: Additional metric values
        
    Returns:
        TrainingMetrics object
    """
    return TrainingMetrics(
        step=step,
        epoch=epoch, 
        loss=loss,
        wall_time=time.time(),
        **kwargs
    ) 