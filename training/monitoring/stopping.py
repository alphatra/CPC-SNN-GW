"""
Early stopping monitor for training control.

This module contains early stopping functionality extracted from
training_metrics.py for better modularity.

Split from training_metrics.py for better maintainability.
"""

import logging
from typing import Optional, Any, Callable

logger = logging.getLogger(__name__)


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
        Initialize early stopping monitor.
        
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
        
        logger.info(f"EarlyStoppingMonitor initialized: patience={patience}, "
                   f"metric={metric_name}, mode={mode}")
    
    def _get_comparison_fn(self) -> Callable[[float, float], bool]:
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
        # Check if current value is better than best
        if self.is_better(current_value, self.best_value):
            # New best value found
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait_count = 0
            
            # Store best weights if requested
            if self.restore_best_weights and model_weights is not None:
                self.best_weights = model_weights
            
            logger.debug(f"New best {self.metric_name}: {current_value:.6f} at epoch {epoch}")
            return False
        else:
            # No improvement
            self.wait_count += 1
            
            logger.debug(f"No improvement in {self.metric_name}: {current_value:.6f} "
                        f"(best: {self.best_value:.6f}, wait: {self.wait_count}/{self.patience})")
            
            # Check if patience exceeded
            if self.wait_count >= self.patience:
                logger.info(f"Early stopping triggered after {self.wait_count} epochs without improvement")
                logger.info(f"Best {self.metric_name}: {self.best_value:.6f} at epoch {self.best_epoch}")
                return True
            
            return False
    
    def should_stop(self) -> bool:
        """Check if training should stop."""
        return self.wait_count >= self.patience
    
    def get_best_weights(self):
        """Get best weights for restoration."""
        return self.best_weights
    
    def get_best_metrics(self) -> dict:
        """Get best metrics achieved."""
        return {
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'metric_name': self.metric_name,
            'mode': self.mode,
            'patience_used': self.wait_count,
            'early_stopped': self.should_stop()
        }
    
    def reset(self):
        """Reset early stopping monitor."""
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.wait_count = 0
        self.best_weights = None
        
        logger.info("EarlyStoppingMonitor reset")


def create_early_stopping_monitor(metric_name: str = "loss",
                                patience: int = 10,
                                min_delta: float = 1e-4,
                                mode: str = "min") -> EarlyStoppingMonitor:
    """
    Factory function for creating early stopping monitor.
    
    Args:
        metric_name: Metric to monitor
        patience: Patience in epochs
        min_delta: Minimum improvement delta
        mode: 'min' or 'max'
        
    Returns:
        Configured EarlyStoppingMonitor
    """
    return EarlyStoppingMonitor(
        patience=patience,
        min_delta=min_delta,
        metric_name=metric_name,
        mode=mode
    )


def create_multi_metric_early_stopping(metrics_config: dict) -> dict:
    """
    Create multiple early stopping monitors for different metrics.
    
    Args:
        metrics_config: Dictionary mapping metric names to configurations
        
    Returns:
        Dictionary of EarlyStoppingMonitor instances
    """
    monitors = {}
    
    for metric_name, config in metrics_config.items():
        monitor = EarlyStoppingMonitor(
            metric_name=metric_name,
            **config
        )
        monitors[metric_name] = monitor
    
    logger.info(f"Created {len(monitors)} early stopping monitors")
    return monitors


# Export early stopping components
__all__ = [
    "EarlyStoppingMonitor",
    "create_early_stopping_monitor",
    "create_multi_metric_early_stopping"
]
