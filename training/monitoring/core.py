"""
Core monitoring and metrics classes.

This module contains core monitoring components extracted from
training_metrics.py for better modularity.

Split from training_metrics.py for better maintainability.
"""

import time
import logging
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import jax.numpy as jnp
import numpy as np

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
    
    # Additional metrics (optional)
    cpc_loss: Optional[float] = None
    snn_loss: Optional[float] = None
    spike_rate: Optional[float] = None
    
    # Custom metrics storage
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.wall_time == 0.0:
            self.wall_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        base_dict = {
            'step': self.step,
            'epoch': self.epoch,
            'loss': self.loss,
            'wall_time': self.wall_time
        }
        
        # Add optional metrics if present
        if self.accuracy is not None:
            base_dict['accuracy'] = self.accuracy
        if self.learning_rate != 0.0:
            base_dict['learning_rate'] = self.learning_rate
        if self.grad_norm is not None:
            base_dict['grad_norm'] = self.grad_norm
        if self.cpc_loss is not None:
            base_dict['cpc_loss'] = self.cpc_loss
        if self.snn_loss is not None:
            base_dict['snn_loss'] = self.snn_loss
        if self.spike_rate is not None:
            base_dict['spike_rate'] = self.spike_rate
        
        # Add custom metrics
        if self.custom_metrics:
            base_dict.update(self.custom_metrics)
        
        return base_dict
    
    def update_custom(self, **kwargs):
        """Update custom metrics."""
        self.custom_metrics.update(kwargs)
    
    def log(self, logger_instance: Optional[logging.Logger] = None):
        """Log metrics to specified logger."""
        if logger_instance is None:
            logger_instance = logger
        
        metrics_str = f"Step {self.step} (Epoch {self.epoch}): loss={self.loss:.4f}"
        
        if self.accuracy is not None:
            metrics_str += f", acc={self.accuracy:.4f}"
        if self.cpc_loss is not None:
            metrics_str += f", cpc_loss={self.cpc_loss:.4f}"
        if self.spike_rate is not None:
            metrics_str += f", spike_rate={self.spike_rate:.3f}"
        
        logger_instance.info(metrics_str)


class ExperimentTracker:
    """
    Experiment tracking and management system.
    
    Handles experiment metadata, results storage, and comparison utilities.
    """
    
    def __init__(self, 
                 experiment_name: str,
                 output_dir: Union[str, Path],
                 save_frequency: int = 10,
                 enable_auto_save: bool = True):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory for saving experiment data
            save_frequency: How often to save (every N steps)
            enable_auto_save: Whether to auto-save metrics
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.save_frequency = save_frequency
        self.enable_auto_save = enable_auto_save
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.metrics_history: List[TrainingMetrics] = []
        self.experiment_metadata = {
            'experiment_name': experiment_name,
            'created_at': time.time(),
            'status': 'running'
        }
        
        logger.info(f"ExperimentTracker initialized: {experiment_name}")
    
    def log_metrics(self, metrics: TrainingMetrics):
        """Log training metrics to tracker."""
        self.metrics_history.append(metrics)
        
        # Auto-save if enabled
        if (self.enable_auto_save and 
            len(self.metrics_history) % self.save_frequency == 0):
            self.save_metrics()
        
        # Log to console
        metrics.log(logger)
    
    def save_metrics(self):
        """Save metrics to file."""
        metrics_file = self.output_dir / f"{self.experiment_name}_metrics.json"
        
        # Convert metrics to serializable format
        metrics_data = [metrics.to_dict() for metrics in self.metrics_history]
        
        # Save with metadata
        save_data = {
            'experiment_metadata': self.experiment_metadata,
            'metrics': metrics_data
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.debug(f"Metrics saved: {len(metrics_data)} entries to {metrics_file}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get experiment summary statistics."""
        if not self.metrics_history:
            return {'no_metrics': True}
        
        # Extract metric arrays
        losses = [m.loss for m in self.metrics_history]
        accuracies = [m.accuracy for m in self.metrics_history if m.accuracy is not None]
        
        summary = {
            'experiment_name': self.experiment_name,
            'total_steps': len(self.metrics_history),
            'duration_minutes': (time.time() - self.experiment_metadata['created_at']) / 60.0,
            
            # Loss statistics
            'final_loss': losses[-1] if losses else 0.0,
            'best_loss': min(losses) if losses else 0.0,
            'loss_improvement': (losses[0] - losses[-1]) if len(losses) > 1 else 0.0,
            
            # Accuracy statistics  
            'final_accuracy': accuracies[-1] if accuracies else 0.0,
            'best_accuracy': max(accuracies) if accuracies else 0.0,
            'accuracy_improvement': (accuracies[-1] - accuracies[0]) if len(accuracies) > 1 else 0.0,
            
            # Training quality
            'converged': self._check_convergence(),
            'stable_training': self._check_stability()
        }
        
        return summary
    
    def _check_convergence(self, window_size: int = 10) -> bool:
        """Check if training has converged."""
        if len(self.metrics_history) < window_size * 2:
            return False
        
        # Check if loss has stabilized in recent window
        recent_losses = [m.loss for m in self.metrics_history[-window_size:]]
        older_losses = [m.loss for m in self.metrics_history[-window_size*2:-window_size]]
        
        recent_mean = jnp.mean(jnp.array(recent_losses))
        older_mean = jnp.mean(jnp.array(older_losses))
        
        # Converged if improvement is very small
        improvement = older_mean - recent_mean
        convergence_threshold = 0.001  # 0.1% improvement threshold
        
        return improvement < convergence_threshold
    
    def _check_stability(self, window_size: int = 20) -> bool:
        """Check if training is stable (no sudden jumps)."""
        if len(self.metrics_history) < window_size:
            return True  # Too early to judge
        
        # Check variance in recent window
        recent_losses = [m.loss for m in self.metrics_history[-window_size:]]
        loss_variance = jnp.var(jnp.array(recent_losses))
        
        # Stable if variance is low relative to loss magnitude
        mean_loss = jnp.mean(jnp.array(recent_losses))
        relative_variance = loss_variance / (mean_loss**2 + 1e-8)
        
        stability_threshold = 0.01  # 1% relative variance threshold
        return relative_variance < stability_threshold
    
    def mark_experiment_complete(self, final_results: Optional[Dict[str, Any]] = None):
        """Mark experiment as completed."""
        self.experiment_metadata['status'] = 'completed'
        self.experiment_metadata['completed_at'] = time.time()
        
        if final_results:
            self.experiment_metadata['final_results'] = final_results
        
        # Final save
        self.save_metrics()
        
        logger.info(f"Experiment {self.experiment_name} marked as completed")


def create_training_metrics(step: int, 
                           epoch: int,
                           loss: float,
                           accuracy: Optional[float] = None,
                           learning_rate: float = 0.0,
                           **kwargs) -> TrainingMetrics:
    """
    Factory function for creating TrainingMetrics.
    
    Args:
        step: Training step number
        epoch: Training epoch number
        loss: Training loss value
        accuracy: Training accuracy (optional)
        learning_rate: Current learning rate
        **kwargs: Additional custom metrics
        
    Returns:
        TrainingMetrics instance
    """
    metrics = TrainingMetrics(
        step=step,
        epoch=epoch,
        loss=loss,
        accuracy=accuracy,
        learning_rate=learning_rate,
        wall_time=time.time()
    )
    
    # Add any additional metrics
    if kwargs:
        metrics.update_custom(**kwargs)
    
    return metrics


def create_experiment_tracker(experiment_name: str,
                            output_dir: Union[str, Path],
                            **kwargs) -> ExperimentTracker:
    """
    Factory function for creating ExperimentTracker.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Output directory for experiment data
        **kwargs: Additional tracker configuration
        
    Returns:
        ExperimentTracker instance
    """
    return ExperimentTracker(
        experiment_name=experiment_name,
        output_dir=output_dir,
        **kwargs
    )


# Export core monitoring components
__all__ = [
    "TrainingMetrics",
    "ExperimentTracker",
    "create_training_metrics",
    "create_experiment_tracker"
]
