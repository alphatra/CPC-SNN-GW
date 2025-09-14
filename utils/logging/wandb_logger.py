"""
Main W&B logger class.

Extracted from wandb_enhanced_logger.py for better modularity.
"""

import os
import sys
import time
import json
import logging
import psutil
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from contextlib import contextmanager

import jax
import jax.numpy as jnp

# W&B integration
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

from .metrics import NeuromorphicMetrics, PerformanceMetrics, SystemInfo
from .visualizations import SpikeVisualizer, GradientVisualizer, DashboardCreator

logger = logging.getLogger(__name__)


class EnhancedWandbLogger:
    """
    🚀 Enhanced W&B Logger for Neuromorphic GW Detection
    
    Comprehensive logging with neuromorphic-specific metrics,
    real-time visualizations, and interactive dashboards.
    """
    
    def __init__(self,
                 project: str = "neuromorphic-gw-detection",
                 name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 tags: Optional[List[str]] = None,
                 notes: Optional[str] = None,
                 output_dir: str = "wandb_outputs",
                 enable_hardware_monitoring: bool = True,
                 enable_visualizations: bool = True,
                 enable_alerts: bool = True,
                 log_frequency: int = 10):
        
        if not HAS_WANDB:
            logger.warning("W&B not available. Install with: pip install wandb")
            self.enabled = False
            return
            
        self.enabled = True
        self.project = project
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.enable_hardware_monitoring = enable_hardware_monitoring
        self.enable_visualizations = enable_visualizations
        self.enable_alerts = enable_alerts
        self.log_frequency = log_frequency
        
        # Initialize W&B
        self.run = None
        try:
            self.run = wandb.init(
                project=project,
                name=name or f"neuromorphic-gw-{int(time.time())}",
                config=config or {},
                tags=tags or ["neuromorphic", "gravitational-waves", "snn", "cpc"],
                notes=notes or "Enhanced neuromorphic GW detection with comprehensive monitoring",
                dir=str(self.output_dir),
                reinit=True
            )
            logger.info(f"🚀 Enhanced W&B logging initialized: {self.run.url}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            self.enabled = False
            return
        
        # Tracking variables
        self.step_count = 0
        self.metrics_buffer = []
        self.hardware_stats = []
        self.gradient_history = []
        self.spike_history = []
        
        # Initialize visualizers
        self.spike_visualizer = SpikeVisualizer(enable_wandb=True)
        self.gradient_visualizer = GradientVisualizer(enable_wandb=True)
        self.dashboard_creator = DashboardCreator(enable_wandb=True, output_dir=self.output_dir)
        
        # System info
        self.system_info = self._collect_system_info()
        self._log_system_info()
        
        # Setup hardware monitoring
        if self.enable_hardware_monitoring:
            self._setup_hardware_monitoring()
    
    def _collect_system_info(self) -> SystemInfo:
        """Collect comprehensive system information"""
        info = SystemInfo()
        
        # JAX info
        if 'jax' in sys.modules:
            info.jax_version = jax.__version__
        
        return info
    
    def _log_system_info(self):
        """Log system information to W&B"""
        if not self.enabled:
            return
            
        try:
            system_dict = {
                'platform': self.system_info.platform,
                'python_version': self.system_info.python_version,
                'jax_version': self.system_info.jax_version,
                'cpu_cores': self.system_info.cpu_cores,
                'total_memory_gb': self.system_info.total_memory_gb,
                'hostname': self.system_info.hostname,
                'username': self.system_info.username
            }
            
            self.run.config.update(system_dict)
            logger.info("💻 System information logged to W&B")
            
        except Exception as e:
            logger.warning(f"System info logging failed: {e}")
    
    def _setup_hardware_monitoring(self):
        """Setup hardware monitoring"""
        try:
            # Initial hardware snapshot
            self._log_hardware_snapshot()
            logger.info("📊 Hardware monitoring initialized")
        except Exception as e:
            logger.warning(f"Hardware monitoring setup failed: {e}")
    
    def _log_hardware_snapshot(self):
        """Log current hardware state"""
        if not self.enabled or not self.enable_hardware_monitoring:
            return
            
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            hardware_metrics = {
                'hardware/memory_usage_percent': memory.percent,
                'hardware/memory_available_gb': memory.available / (1024**3),
                'hardware/cpu_usage_percent': cpu_percent,
            }
            
            # GPU metrics if available
            try:
                devices = jax.devices('gpu')
                if devices:
                    hardware_metrics['hardware/gpu_count'] = len(devices)
            except:
                pass
            
            self.run.log(hardware_metrics, step=self.step_count)
            
        except Exception as e:
            logger.warning(f"Hardware snapshot failed: {e}")
    
    def log_neuromorphic_metrics(self, metrics: NeuromorphicMetrics, prefix: str = "neuromorphic"):
        """Log neuromorphic-specific metrics"""
        if not self.enabled:
            return
            
        try:
            # Convert metrics to dictionary with prefix
            metrics_dict = {f"{prefix}/{k}": v for k, v in metrics.to_dict().items()}
            
            self.run.log(metrics_dict, step=self.step_count)
            
            # Store for history tracking
            self.metrics_buffer.append({
                'step': self.step_count,
                'spike_rate': metrics.spike_rate,
                'encoding_fidelity': metrics.encoding_fidelity,
                'contrastive_accuracy': metrics.contrastive_accuracy
            })
            
            logger.debug(f"🧠 Logged neuromorphic metrics: spike_rate={metrics.spike_rate:.3f}")
            
        except Exception as e:
            logger.warning(f"Neuromorphic metrics logging failed: {e}")
    
    def log_performance_metrics(self, metrics: PerformanceMetrics, prefix: str = "performance"):
        """Log performance and hardware metrics"""
        if not self.enabled:
            return
            
        try:
            # Convert metrics to dictionary with prefix
            metrics_dict = {f"{prefix}/{k}": v for k, v in metrics.__dict__.items()}
            
            self.run.log(metrics_dict, step=self.step_count)
            
            # Log hardware snapshot periodically
            if self.step_count % self.log_frequency == 0:
                self._log_hardware_snapshot()
            
            # Store for history
            self.hardware_stats.append({
                'step': self.step_count,
                'memory_usage_mb': metrics.memory_usage_mb,
                'inference_latency_ms': metrics.inference_latency_ms,
                'cpu_usage_percent': metrics.cpu_usage_percent
            })
            
            logger.debug(f"⚡ Logged performance metrics: latency={metrics.inference_latency_ms:.1f}ms")
            
        except Exception as e:
            logger.warning(f"Performance metrics logging failed: {e}")
    
    def log_spike_patterns(self, spikes: jnp.ndarray, name: str = "spike_patterns"):
        """Log spike patterns with visualizations"""
        if not self.enabled:
            return
            
        spike_stats = self.spike_visualizer.log_spike_patterns(
            spikes, name, self.step_count, self.run
        )
        
        # Store for history
        if spike_stats:
            self.spike_history.append({
                'step': self.step_count,
                **spike_stats
            })
    
    def log_gradient_stats(self, gradients: Dict[str, jnp.ndarray], prefix: str = "gradients"):
        """Log gradient statistics with visualizations"""
        if not self.enabled:
            return
            
        gradient_stats = self.gradient_visualizer.log_gradient_stats(
            gradients, prefix, self.step_count, self.run
        )
        
        # Store for history
        if gradient_stats:
            self.gradient_history.append({
                'step': self.step_count,
                'gradient_norm': gradient_stats.get(f'{prefix}/global_norm', 0),
                'vanishing_ratio': gradient_stats.get(f'{prefix}/vanishing_ratio', 0)
            })
    
    def log_learning_curves(self, train_metrics: Dict[str, float], 
                           val_metrics: Optional[Dict[str, float]] = None):
        """Log training and validation learning curves"""
        if not self.enabled:
            return
            
        try:
            # Prepare metrics for logging
            log_dict = {}
            
            # Training metrics
            for key, value in train_metrics.items():
                log_dict[f"train/{key}"] = float(value)
            
            # Validation metrics
            if val_metrics:
                for key, value in val_metrics.items():
                    log_dict[f"val/{key}"] = float(value)
            
            self.run.log(log_dict, step=self.step_count)
            
            logger.debug("📈 Logged learning curves")
            
        except Exception as e:
            logger.warning(f"Learning curves logging failed: {e}")
    
    def log_model_artifacts(self, model_params: Dict[str, jnp.ndarray], 
                           model_path: Optional[str] = None):
        """Log model artifacts and parameters"""
        if not self.enabled:
            return
            
        try:
            # Log model statistics
            param_stats = {}
            total_params = 0
            
            for name, param in model_params.items():
                param_array = jnp.array(param)
                param_count = param_array.size
                total_params += param_count
                
                param_stats[f"model/{name}_shape"] = list(param_array.shape)
                param_stats[f"model/{name}_params"] = param_count
                param_stats[f"model/{name}_norm"] = float(jnp.linalg.norm(param_array))
            
            param_stats["model/total_parameters"] = total_params
            self.run.log(param_stats, step=self.step_count)
            
            # Log model file if provided
            if model_path and Path(model_path).exists():
                artifact = wandb.Artifact(f"model-step-{self.step_count}", type="model")
                artifact.add_file(model_path)
                self.run.log_artifact(artifact)
                logger.info(f"📦 Model artifact logged: {model_path}")
            
        except Exception as e:
            logger.warning(f"Model artifacts logging failed: {e}")
    
    def create_summary_dashboard(self):
        """Create comprehensive summary dashboard"""
        if not self.enabled:
            return
            
        self.dashboard_creator.create_summary_dashboard(
            self.metrics_buffer, self.gradient_history, 
            self.spike_history, self.step_count
        )
    
    def log_step_context(self, step: int):
        """Update step context and create periodic summaries"""
        self.step_count = step
        
        # Create summary dashboard periodically
        if step > 0 and step % (self.log_frequency * 10) == 0:
            self.create_summary_dashboard()
    
    def finish(self):
        """Finish logging session with final summary"""
        if not self.enabled:
            return
            
        try:
            # Create final summary dashboard
            self.create_summary_dashboard()
            
            # Log session summary
            session_summary = {
                'total_steps': self.step_count,
                'metrics_logged': len(self.metrics_buffer),
                'hardware_snapshots': len(self.hardware_stats),
                'gradient_updates': len(self.gradient_history)
            }
            
            self.run.summary.update(session_summary)
            
            # Save local summary
            summary_file = self.output_dir / "session_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(session_summary, f, indent=2)
            
            self.run.finish()
            logger.info("✅ Enhanced W&B logging session finished")
            
        except Exception as e:
            logger.warning(f"Session finish failed: {e}")
            if self.run:
                self.run.finish()
    
    @contextmanager
    def log_context(self, name: str):
        """Context manager for logging code blocks"""
        start_time = time.time()
        try:
            yield
        finally:
            if self.enabled:
                duration = time.time() - start_time
                self.run.log({f"timing/{name}_duration_ms": duration * 1000}, 
                           step=self.step_count)
