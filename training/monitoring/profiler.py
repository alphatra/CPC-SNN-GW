"""
Performance profiling and enhanced logging components.

This module contains profiling and logging components extracted from
training_metrics.py for better modularity.

Split from training_metrics.py for better maintainability.
"""

import time
import logging
import psutil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import jax
import jax.numpy as jnp

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
class PerformanceMetrics:
    """Container for performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    gpu_memory_used_gb: float = 0.0
    wall_time: float = 0.0
    step_time: float = 0.0


class PerformanceProfiler:
    """
    Performance profiler for training monitoring.
    
    Tracks CPU, memory, GPU utilization, and timing metrics.
    """
    
    def __init__(self, 
                 enable_gpu_monitoring: bool = True,
                 profile_frequency: int = 10):
        """
        Initialize performance profiler.
        
        Args:
            enable_gpu_monitoring: Whether to monitor GPU if available
            profile_frequency: How often to profile (every N steps)
        """
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.profile_frequency = profile_frequency
        
        # Performance history
        self.performance_history: List[PerformanceMetrics] = []
        self.step_times: List[float] = []
        
        # Baseline measurements
        self.baseline_memory = self._get_memory_usage()
        self.start_time = time.time()
        
        logger.info(f"PerformanceProfiler initialized: GPU monitoring={enable_gpu_monitoring}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024**3  # Convert to GB
    
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage if available."""
        if not self.enable_gpu_monitoring:
            return 0.0
        
        try:
            # Try to get JAX device memory info
            devices = jax.devices()
            if devices and hasattr(devices[0], 'memory_stats'):
                memory_stats = devices[0].memory_stats()
                if memory_stats and 'bytes_in_use' in memory_stats:
                    return memory_stats['bytes_in_use'] / 1024**3
        except:
            pass
        
        return 0.0
    
    def profile_step(self, step: int) -> Optional[PerformanceMetrics]:
        """
        Profile current step performance.
        
        Args:
            step: Current training step
            
        Returns:
            PerformanceMetrics if profiling step, None otherwise
        """
        # Only profile at specified frequency
        if step % self.profile_frequency != 0:
            return None
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        memory_used_gb = self._get_memory_usage()
        gpu_memory_used_gb = self._get_gpu_memory_usage()
        
        # Calculate timing metrics
        current_time = time.time()
        wall_time = current_time - self.start_time
        
        # Estimate step time from recent history
        if len(self.step_times) > 0:
            step_time = jnp.mean(jnp.array(self.step_times[-10:]))  # Average of last 10
        else:
            step_time = 0.0
        
        metrics = PerformanceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            gpu_memory_used_gb=gpu_memory_used_gb,
            wall_time=wall_time,
            step_time=float(step_time)
        )
        
        self.performance_history.append(metrics)
        
        # Log performance if concerning
        if memory_percent > 90:
            logger.warning(f"High memory usage: {memory_percent:.1f}%")
        if cpu_percent > 95:
            logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
        
        return metrics
    
    def record_step_time(self, step_time: float):
        """Record timing for a training step."""
        self.step_times.append(step_time)
        
        # Keep only recent history
        if len(self.step_times) > 1000:
            self.step_times = self.step_times[-500:]  # Keep last 500
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.performance_history:
            return {'no_data': True}
        
        # Extract metrics arrays
        cpu_usage = [m.cpu_percent for m in self.performance_history]
        memory_usage = [m.memory_percent for m in self.performance_history]
        memory_gb = [m.memory_used_gb for m in self.performance_history]
        gpu_memory_gb = [m.gpu_memory_used_gb for m in self.performance_history]
        
        summary = {
            'cpu_stats': {
                'mean': float(jnp.mean(jnp.array(cpu_usage))),
                'max': float(jnp.max(jnp.array(cpu_usage))),
                'std': float(jnp.std(jnp.array(cpu_usage)))
            },
            'memory_stats': {
                'mean_percent': float(jnp.mean(jnp.array(memory_usage))),
                'max_percent': float(jnp.max(jnp.array(memory_usage))),
                'mean_gb': float(jnp.mean(jnp.array(memory_gb))),
                'max_gb': float(jnp.max(jnp.array(memory_gb))),
                'memory_growth_gb': memory_gb[-1] - self.baseline_memory if memory_gb else 0.0
            },
            'timing_stats': {
                'total_wall_time': self.performance_history[-1].wall_time if self.performance_history else 0.0,
                'mean_step_time': float(jnp.mean(jnp.array(self.step_times))) if self.step_times else 0.0,
                'steps_recorded': len(self.step_times)
            }
        }
        
        # Add GPU stats if available
        if any(gpu > 0 for gpu in gpu_memory_gb):
            summary['gpu_stats'] = {
                'mean_memory_gb': float(jnp.mean(jnp.array(gpu_memory_gb))),
                'max_memory_gb': float(jnp.max(jnp.array(gpu_memory_gb)))
            }
        
        return summary


class EnhancedMetricsLogger:
    """
    Enhanced metrics logger with multiple backend support.
    
    Supports logging to multiple backends: console, file, wandb, tensorboard.
    """
    
    def __init__(self,
                 experiment_name: str,
                 log_dir: Optional[str] = None,
                 enable_wandb: bool = False,
                 enable_tensorboard: bool = False,
                 wandb_project: str = "cpc-snn-gw"):
        """
        Initialize enhanced metrics logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for file logging
            enable_wandb: Whether to enable W&B logging
            enable_tensorboard: Whether to enable TensorBoard logging
            wandb_project: W&B project name
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.enable_wandb = enable_wandb and WANDB_AVAILABLE
        self.enable_tensorboard = enable_tensorboard and TENSORBOARD_AVAILABLE
        
        # Initialize backends
        self.wandb_run = None
        self.tensorboard_writer = None
        
        self._setup_backends(wandb_project)
        
        logger.info(f"EnhancedMetricsLogger initialized: "
                   f"wandb={self.enable_wandb}, "
                   f"tensorboard={self.enable_tensorboard}")
    
    def _setup_backends(self, wandb_project: str):
        """Setup logging backends."""
        # W&B setup
        if self.enable_wandb:
            try:
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=self.experiment_name,
                    reinit=True
                )
                logger.info("✅ W&B logging enabled")
            except Exception as e:
                logger.warning(f"W&B setup failed: {e}")
                self.enable_wandb = False
        
        # TensorBoard setup
        if self.enable_tensorboard:
            try:
                from pathlib import Path
                tb_log_dir = Path(self.log_dir or ".") / "tensorboard" / self.experiment_name
                tb_log_dir.mkdir(parents=True, exist_ok=True)
                
                self.tensorboard_writer = SummaryWriter(log_dir=str(tb_log_dir))
                logger.info("✅ TensorBoard logging enabled")
            except Exception as e:
                logger.warning(f"TensorBoard setup failed: {e}")
                self.enable_tensorboard = False
    
    def log_metrics(self, metrics_dict: Dict[str, Any], step: int):
        """Log metrics to all enabled backends."""
        # Console logging
        metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                               for k, v in metrics_dict.items())
        logger.info(f"Step {step}: {metrics_str}")
        
        # W&B logging
        if self.enable_wandb and self.wandb_run:
            try:
                self.wandb_run.log(metrics_dict, step=step)
            except Exception as e:
                logger.warning(f"W&B logging failed: {e}")
        
        # TensorBoard logging
        if self.enable_tensorboard and self.tensorboard_writer:
            try:
                for key, value in metrics_dict.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(key, value, step)
                self.tensorboard_writer.flush()
            except Exception as e:
                logger.warning(f"TensorBoard logging failed: {e}")
    
    def log_model_graph(self, model, sample_input):
        """Log model graph if supported."""
        if self.enable_tensorboard and self.tensorboard_writer:
            try:
                # Would add model graph logging here
                logger.debug("Model graph logging not implemented yet")
            except Exception as e:
                logger.warning(f"Model graph logging failed: {e}")
    
    def finish(self):
        """Finish logging and cleanup resources."""
        if self.enable_wandb and self.wandb_run:
            try:
                self.wandb_run.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.warning(f"W&B finish failed: {e}")
        
        if self.enable_tensorboard and self.tensorboard_writer:
            try:
                self.tensorboard_writer.close()
                logger.info("TensorBoard writer closed")
            except Exception as e:
                logger.warning(f"TensorBoard close failed: {e}")


def create_enhanced_metrics_logger(config: Dict[str, Any],
                                 experiment_name: Optional[str] = None) -> EnhancedMetricsLogger:
    """
    Factory function for creating enhanced metrics logger.
    
    Args:
        config: Logger configuration dictionary
        experiment_name: Optional experiment name override
        
    Returns:
        Configured EnhancedMetricsLogger
    """
    if experiment_name is None:
        experiment_name = config.get('experiment_name', 'cpc_snn_experiment')
    
    return EnhancedMetricsLogger(
        experiment_name=experiment_name,
        log_dir=config.get('log_dir'),
        enable_wandb=config.get('enable_wandb', False),
        enable_tensorboard=config.get('enable_tensorboard', False),
        wandb_project=config.get('wandb_project', 'cpc-snn-gw')
    )


# Export profiling and logging components
__all__ = [
    "PerformanceProfiler",
    "PerformanceMetrics",
    "EnhancedMetricsLogger",
    "create_enhanced_metrics_logger"
]
