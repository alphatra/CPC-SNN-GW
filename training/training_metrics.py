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
    from utils.wandb_enhanced_logger import (
        EnhancedWandbLogger, create_enhanced_wandb_logger,
        NeuromorphicMetrics, PerformanceMetrics,
        create_neuromorphic_metrics, create_performance_metrics
    )
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    EnhancedWandbLogger = None
    create_enhanced_wandb_logger = None

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
                # ✅ FIX: Check if W&B run already exists
                if wandb.run is not None:
                    self.wandb_run = wandb.run
                    logger.info("W&B tracking - using existing run")
                else:
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
                # ✅ FIX: Allow value changes for W&B config updates
                self.wandb_run.config.update(config, allow_val_change=True)
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


class EnhancedMetricsLogger:
    """
    🚀 Enhanced metrics logger with comprehensive neuromorphic tracking
    
    Integrates EnhancedWandbLogger with complete neuromorphic-specific metrics,
    performance monitoring, and interactive visualizations.
    """
    
    def __init__(self,
                 project_name: str = "neuromorphic-gw-detection",
                 experiment_name: Optional[str] = None,
                 output_dir: str = "outputs",
                 wandb_config: Optional[Dict[str, Any]] = None,
                 config: Optional[Dict[str, Any]] = None,
                 tags: Optional[List[str]] = None):
        
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.config = config or {}
        
        # Initialize enhanced W&B logger
        self.enhanced_wandb = None
        if WANDB_AVAILABLE and wandb_config and wandb_config.get('enabled', True):
            try:
                wandb_settings = wandb_config.copy()
                wandb_settings.update({
                    'project': wandb_settings.get('project', project_name),
                    'name': wandb_settings.get('name', experiment_name),
                    'config': config,
                    'tags': wandb_settings.get('tags', tags or []),
                    'output_dir': str(self.output_dir)
                })
                
                self.enhanced_wandb = create_enhanced_wandb_logger(**wandb_settings)
                logger.info("🚀 Enhanced W&B logger initialized")
                
            except Exception as e:
                logger.warning(f"Enhanced W&B initialization failed: {e}")
                self.enhanced_wandb = None
        
        # Fallback to basic logger if enhanced fails
        if not self.enhanced_wandb and WANDB_AVAILABLE:
            try:
                from training.training_metrics import WandbLogger
                self.fallback_logger = WandbLogger(
                    project_name=project_name,
                    experiment_name=experiment_name,
                    output_dir=output_dir,
                    use_wandb=True,
                    wandb_config=wandb_config
                )
                logger.info("Using fallback W&B logger")
            except Exception as e:
                logger.warning(f"Fallback logger initialization failed: {e}")
                self.fallback_logger = None
        else:
            self.fallback_logger = None
        
        # Metrics buffers
        self.step_count = 0
        self.performance_buffer = []
        self.neuromorphic_buffer = []
    
    def log_training_step(self, 
                         metrics: TrainingMetrics,
                         model_state: Any = None,
                         gradients: Optional[Dict[str, jnp.ndarray]] = None,
                         spikes: Optional[jnp.ndarray] = None,
                         performance_data: Optional[Dict[str, float]] = None,
                         prefix: str = "train"):
        """
        Log comprehensive training step with neuromorphic and performance metrics
        """
        
        if self.enhanced_wandb:
            with self.enhanced_wandb.log_step_context(step=self.step_count):
                
                # 1. Log basic training metrics
                basic_metrics = {
                    f"{prefix}/loss": float(metrics.loss),
                    f"{prefix}/accuracy": float(getattr(metrics, 'accuracy', 0.0)),
                    f"{prefix}/epoch": metrics.epoch,
                    f"{prefix}/learning_rate": float(getattr(metrics, 'learning_rate', 0.0))
                }
                
                if hasattr(metrics, 'custom_metrics') and getattr(metrics, 'custom_metrics'):
                    for key, value in getattr(metrics, 'custom_metrics').items():
                        basic_metrics[f"{prefix}/{key}"] = float(value)
                
                self.enhanced_wandb.run.log(basic_metrics, step=self.step_count)
                
                # 2. Log neuromorphic-specific metrics
                if spikes is not None:
                    self._log_neuromorphic_metrics(spikes, prefix)
                
                # 3. Log performance metrics
                if performance_data:
                    self._log_performance_metrics(performance_data, prefix)
                
                # 4. Log gradient statistics
                if gradients:
                    self.enhanced_wandb.log_gradient_stats(gradients, f"{prefix}_gradients")
                
                # 5. Log spike patterns
                if spikes is not None:
                    self.enhanced_wandb.log_spike_patterns(spikes, f"{prefix}_spikes")
        
        elif self.fallback_logger:
            # Use fallback logger
            self.fallback_logger.log_metrics(metrics, prefix)
        
        self.step_count += 1
    
    def _log_neuromorphic_metrics(self, spikes: jnp.ndarray, prefix: str):
        """Extract and log neuromorphic-specific metrics"""
        try:
            spikes_np = np.array(spikes)
            
            # Calculate neuromorphic metrics
            spike_rate = float(np.mean(spikes_np))
            spike_std = float(np.std(spikes_np))
            spike_sparsity = 1.0 - spike_rate
            
            # Create neuromorphic metrics object
            if create_neuromorphic_metrics:
                neuro_metrics = create_neuromorphic_metrics(
                    spike_rate=spike_rate,
                    spike_sparsity=spike_sparsity,
                    encoding_fidelity=min(spike_rate * 10, 1.0),  # Heuristic
                    network_activity=spike_rate
                )
                
                # Log to enhanced wandb
                self.enhanced_wandb.log_neuromorphic_metrics(neuro_metrics, prefix)
            
            # Store for analysis
            self.neuromorphic_buffer.append({
                'step': self.step_count,
                'spike_rate': spike_rate,
                'sparsity': spike_sparsity
            })
            
        except Exception as e:
            logger.warning(f"Neuromorphic metrics calculation failed: {e}")
    
    def _log_performance_metrics(self, performance_data: Dict[str, float], prefix: str):
        """Log performance and hardware metrics"""
        try:
            # Create performance metrics object
            if create_performance_metrics:
                perf_metrics = create_performance_metrics(
                    inference_latency_ms=performance_data.get('inference_latency_ms', 0.0),
                    memory_usage_mb=performance_data.get('memory_usage_mb', 0.0),
                    cpu_usage_percent=performance_data.get('cpu_usage_percent', 0.0),
                    samples_per_second=performance_data.get('samples_per_second', 0.0)
                )
                
                # Log to enhanced wandb
                self.enhanced_wandb.log_performance_metrics(perf_metrics, prefix)
            
            # Store for analysis
            self.performance_buffer.append({
                'step': self.step_count,
                'latency': performance_data.get('inference_latency_ms', 0.0),
                'memory': performance_data.get('memory_usage_mb', 0.0)
            })
            
        except Exception as e:
            logger.warning(f"Performance metrics logging failed: {e}")
    
    def finish(self):
        """Finish logging and cleanup"""
        if self.enhanced_wandb:
            self.enhanced_wandb.finish()
        if self.fallback_logger:
            self.fallback_logger.finish()
        
        logger.info("🏁 Enhanced metrics logging finished")


# Factory function for easy creation
def create_enhanced_metrics_logger(config: Dict[str, Any],
                                 experiment_name: Optional[str] = None,
                                 output_dir: str = "outputs") -> EnhancedMetricsLogger:
    """Factory function to create enhanced metrics logger from config"""
    
    # Extract wandb config
    wandb_config = config.get('wandb', {})
    
    # Set project name from config or default
    project_name = wandb_config.get('project', 'neuromorphic-gw-detection')
    
    # Generate experiment name if not provided
    if not experiment_name:
        experiment_name = wandb_config.get('name') or f"neuromorphic-gw-{int(time.time())}"
    
    return EnhancedMetricsLogger(
        project_name=project_name,
        experiment_name=experiment_name,
        output_dir=output_dir,
        wandb_config=wandb_config,
        config=config,
        tags=wandb_config.get('tags', ['neuromorphic', 'gravitational-waves'])
    ) 