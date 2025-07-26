#!/usr/bin/env python3
"""
🚀 Enhanced W&B Logger for Neuromorphic GW Detection

Comprehensive logging system with neuromorphic-specific metrics, 
real-time visualizations, performance monitoring, and interactive dashboards.

Features:
- Neuromorphic metrics (spike rates, patterns, encoding efficiency)
- Performance profiling (latency, memory, hardware utilization)
- Custom visualizations (spike rasters, attention maps, gradient flows)
- Real-time monitoring with alerts
- Hardware telemetry (CPU/GPU/memory)
- Scientific metrics (ROC curves, confusion matrices)
- Artifact management (models, datasets, plots)
- Interactive dashboards and reports
"""

import os
import sys
import time
import json
import logging
import psutil
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Union, Tuple

# Optional plotting dependency
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from pathlib import Path
from dataclasses import dataclass, asdict
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

# Visualization dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger(__name__)

@dataclass
class NeuromorphicMetrics:
    """Neuromorphic-specific metrics for comprehensive tracking"""
    
    # Spike dynamics
    spike_rate: float = 0.0                    # Average spike rate (Hz)
    spike_frequency: float = 0.0               # Dominant frequency (Hz) 
    spike_synchrony: float = 0.0               # Population synchrony measure
    spike_sparsity: float = 0.0                # Sparsity coefficient
    spike_efficiency: float = 0.0              # Information per spike
    
    # Encoding metrics
    encoding_snr: float = 0.0                  # Signal-to-noise ratio
    encoding_fidelity: float = 0.0             # Reconstruction quality
    temporal_precision: float = 0.0            # Timing precision (ms)
    spike_train_correlation: float = 0.0       # Inter-train correlation
    
    # Network dynamics
    membrane_potential_std: float = 0.0        # Membrane potential variability
    synaptic_weight_norm: float = 0.0          # Weight matrix norm
    network_activity: float = 0.0              # Overall network activity
    adaptation_rate: float = 0.0               # Learning adaptation speed
    
    # CPC-specific metrics
    contrastive_accuracy: float = 0.0          # CPC contrastive task accuracy
    representation_rank: float = 0.0           # Effective dimensionality
    mutual_information: float = 0.0            # MI between consecutive states
    prediction_horizon: float = 0.0            # Effective prediction steps
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging"""
        return asdict(self)

@dataclass
class PerformanceMetrics:
    """Performance and hardware monitoring metrics"""
    
    # Latency metrics (milliseconds)
    inference_latency_ms: float = 0.0
    cpc_forward_ms: float = 0.0
    spike_encoding_ms: float = 0.0
    snn_forward_ms: float = 0.0
    total_pipeline_ms: float = 0.0
    
    # Memory metrics (MB)
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    memory_growth_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    swap_usage_mb: float = 0.0
    
    # Hardware metrics
    cpu_usage_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    temperature_celsius: float = 0.0
    power_consumption_watts: float = 0.0
    
    # Throughput metrics
    samples_per_second: float = 0.0
    batches_per_second: float = 0.0
    tokens_per_second: float = 0.0
    
    # JAX compilation metrics
    jit_compilation_time_ms: float = 0.0
    num_compilations: int = 0
    cache_hit_rate: float = 0.0

@dataclass
class SystemInfo:
    """System and environment information"""
    
    # Platform info
    platform: str = ""
    python_version: str = ""
    jax_version: str = ""
    jax_backend: str = ""
    device_count: int = 0
    device_types: List[str] = None
    
    # Hardware info
    cpu_model: str = ""
    cpu_cores: int = 0
    total_memory_gb: float = 0.0
    gpu_model: str = ""
    gpu_memory_gb: float = 0.0
    
    # Environment
    conda_env: str = ""
    cuda_version: str = ""
    git_commit: str = ""
    experiment_id: str = ""
    
    def __post_init__(self):
        if self.device_types is None:
            self.device_types = []

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
        
        # System info
        self.system_info = self._collect_system_info()
        self._log_system_info()
        
        # Setup hardware monitoring
        if self.enable_hardware_monitoring:
            self._setup_hardware_monitoring()
    
    def _collect_system_info(self) -> SystemInfo:
        """Collect comprehensive system information"""
        import platform
        import sys
        
        info = SystemInfo()
        
        # Platform info
        info.platform = platform.platform()
        info.python_version = sys.version.split()[0]
        
        # JAX info
        if 'jax' in sys.modules:
            info.jax_version = jax.__version__
            info.jax_backend = jax.lib.xla_bridge.get_backend().platform
            info.device_count = len(jax.devices())
            info.device_types = [str(device).split(':')[0] for device in jax.devices()]
        
        # Hardware info
        info.cpu_cores = psutil.cpu_count()
        info.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Environment info
        info.conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
        
        try:
            import subprocess
            info.git_commit = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except:
            info.git_commit = "unknown"
        
        info.experiment_id = f"exp_{int(time.time())}"
        
        return info
    
    def _log_system_info(self):
        """Log system information to W&B"""
        if not self.enabled:
            return
            
        try:
            system_table = wandb.Table(
                columns=["Property", "Value"],
                data=[
                    ["Platform", self.system_info.platform],
                    ["Python Version", self.system_info.python_version],
                    ["JAX Version", self.system_info.jax_version],
                    ["JAX Backend", self.system_info.jax_backend], 
                    ["Device Count", self.system_info.device_count],
                    ["Device Types", ', '.join(self.system_info.device_types)],
                    ["CPU Cores", self.system_info.cpu_cores],
                    ["Total Memory (GB)", f"{self.system_info.total_memory_gb:.1f}"],
                    ["Conda Environment", self.system_info.conda_env],
                    ["Git Commit", self.system_info.git_commit],
                    ["Experiment ID", self.system_info.experiment_id]
                ]
            )
            
            self.run.log({"system_info": system_table})
            logger.info("✅ System information logged to W&B")
            
        except Exception as e:
            logger.warning(f"Failed to log system info: {e}")
    
    def _setup_hardware_monitoring(self):
        """Setup hardware monitoring infrastructure"""
        try:
            # Initial hardware state
            self._log_hardware_snapshot()
            logger.info("🖥️  Hardware monitoring enabled")
        except Exception as e:
            logger.warning(f"Hardware monitoring setup failed: {e}")
    
    def _log_hardware_snapshot(self):
        """Log current hardware state"""
        if not self.enabled:
            return
            
        try:
            # CPU info
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            
            # Memory info  
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk info
            disk = psutil.disk_usage('/')
            
            hardware_metrics = {
                "hardware/cpu_percent": cpu_percent,
                "hardware/cpu_freq_mhz": cpu_freq.current if cpu_freq else 0,
                "hardware/memory_percent": memory.percent,
                "hardware/memory_used_gb": memory.used / (1024**3),
                "hardware/memory_available_gb": memory.available / (1024**3),
                "hardware/swap_percent": swap.percent,
                "hardware/disk_percent": (disk.used / disk.total) * 100,
                "hardware/timestamp": time.time()
            }
            
            # Try to get temperature (Linux/macOS specific)
            try:
                if hasattr(psutil, "sensors_temperatures"):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        # Get average temperature
                        all_temps = [temp.current for sensor_list in temps.values() 
                                   for temp in sensor_list if temp.current]
                        if all_temps:
                            hardware_metrics["hardware/temperature_celsius"] = np.mean(all_temps)
            except:
                pass
            
            self.run.log(hardware_metrics, step=self.step_count)
            
        except Exception as e:
            logger.warning(f"Hardware snapshot failed: {e}")
    
    def log_neuromorphic_metrics(self, metrics: NeuromorphicMetrics, prefix: str = "neuromorphic"):
        """Log neuromorphic-specific metrics with visualizations"""
        if not self.enabled:
            return
            
        try:
            # Convert to dict with prefix
            metrics_dict = {f"{prefix}/{k}": v for k, v in metrics.to_dict().items()}
            self.run.log(metrics_dict, step=self.step_count)
            
            # Log to buffer for visualizations
            self.metrics_buffer.append({
                'step': self.step_count,
                'timestamp': time.time(),
                **metrics_dict
            })
            
            logger.debug(f"📊 Logged neuromorphic metrics: {len(metrics_dict)} values")
            
        except Exception as e:
            logger.warning(f"Neuromorphic metrics logging failed: {e}")
    
    def log_performance_metrics(self, metrics: PerformanceMetrics, prefix: str = "performance"):
        """Log performance and hardware metrics"""
        if not self.enabled:
            return
            
        try:
            # Convert to dict with prefix
            metrics_dict = {f"{prefix}/{k}": v for k, v in asdict(metrics).items()}
            self.run.log(metrics_dict, step=self.step_count)
            
            # Hardware monitoring
            if self.enable_hardware_monitoring and self.step_count % self.log_frequency == 0:
                self._log_hardware_snapshot()
            
            # Performance alert checks
            if self.enable_alerts:
                self._check_performance_alerts(metrics)
            
            logger.debug(f"⚡ Logged performance metrics: {len(metrics_dict)} values")
            
        except Exception as e:
            logger.warning(f"Performance metrics logging failed: {e}")
    
    def log_spike_patterns(self, spikes: jnp.ndarray, name: str = "spike_patterns"):
        """Log spike patterns with raster plots and statistics"""
        if not self.enabled or not self.enable_visualizations:
            return
            
        try:
            # Convert to numpy for processing
            spikes_np = np.array(spikes)
            
            # Basic spike statistics
            spike_rate = float(np.mean(spikes_np))
            spike_std = float(np.std(spikes_np))
            spike_count = int(np.sum(spikes_np))
            
            # Log basic stats
            self.run.log({
                f"{name}/spike_rate": spike_rate,
                f"{name}/spike_std": spike_std, 
                f"{name}/spike_count": spike_count,
                f"{name}/sparsity": 1.0 - spike_rate
            }, step=self.step_count)
            
            # Create raster plot (sample subset for performance)
            if len(spikes_np.shape) >= 2 and self.step_count % (self.log_frequency * 5) == 0:
                self._create_spike_raster_plot(spikes_np, name)
            
            # Store for history tracking
            self.spike_history.append({
                'step': self.step_count,
                'spike_rate': spike_rate,
                'spike_count': spike_count
            })
            
            logger.debug(f"🔥 Logged spike patterns: rate={spike_rate:.3f}")
            
        except Exception as e:
            logger.warning(f"Spike pattern logging failed: {e}")
    
    def _create_spike_raster_plot(self, spikes: np.ndarray, name: str):
        """Create and log spike raster plot"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Sample subset for visualization (max 100 neurons, 1000 time steps)
            if spikes.shape[0] > 100:
                neuron_indices = np.random.choice(spikes.shape[0], 100, replace=False)
                spikes_sample = spikes[neuron_indices]
            else:
                spikes_sample = spikes
                
            if spikes_sample.shape[-1] > 1000:
                time_indices = np.random.choice(spikes_sample.shape[-1], 1000, replace=False)
                spikes_sample = spikes_sample[..., time_indices]
            
            # Create raster plot
            if len(spikes_sample.shape) == 2:  # [neurons, time]
                spike_times, spike_neurons = np.where(spikes_sample)
                ax.scatter(spike_times, spike_neurons, s=1, alpha=0.7, c='black')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Neuron Index')
                ax.set_title(f'{name} - Spike Raster Plot')
            
            elif len(spikes_sample.shape) == 3:  # [batch, neurons, time]
                # Show first batch
                spike_times, spike_neurons = np.where(spikes_sample[0])
                ax.scatter(spike_times, spike_neurons, s=1, alpha=0.7, c='black')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Neuron Index')
                ax.set_title(f'{name} - Spike Raster Plot (Batch 0)')
            
            plt.tight_layout()
            
            # Log to W&B
            self.run.log({f"{name}_raster": wandb.Image(fig)}, step=self.step_count)
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Raster plot creation failed: {e}")
    
    def log_gradient_stats(self, gradients: Dict[str, jnp.ndarray], prefix: str = "gradients"):
        """Log gradient statistics with distributions"""
        if not self.enabled:
            return
            
        try:
            gradient_stats = {}
            
            # Flatten all gradients
            all_grads = []
            for name, grad in gradients.items():
                grad_np = np.array(grad).flatten()
                all_grads.extend(grad_np.tolist())
                
                # Per-parameter statistics
                gradient_stats[f"{prefix}/{name}_norm"] = float(np.linalg.norm(grad_np))
                gradient_stats[f"{prefix}/{name}_mean"] = float(np.mean(grad_np))
                gradient_stats[f"{prefix}/{name}_std"] = float(np.std(grad_np))
                gradient_stats[f"{prefix}/{name}_max"] = float(np.max(np.abs(grad_np)))
            
            # Overall gradient statistics
            all_grads = np.array(all_grads)
            gradient_stats[f"{prefix}/total_norm"] = float(np.linalg.norm(all_grads))
            gradient_stats[f"{prefix}/mean"] = float(np.mean(all_grads))
            gradient_stats[f"{prefix}/std"] = float(np.std(all_grads))
            gradient_stats[f"{prefix}/max_abs"] = float(np.max(np.abs(all_grads)))
            
            self.run.log(gradient_stats, step=self.step_count)
            
            # Create histogram periodically
            if self.enable_visualizations and self.step_count % (self.log_frequency * 3) == 0:
                self._create_gradient_histogram(all_grads, prefix)
            
            # Store for history
            self.gradient_history.append({
                'step': self.step_count,
                'total_norm': gradient_stats[f"{prefix}/total_norm"]
            })
            
            logger.debug(f"📈 Logged gradient stats: norm={gradient_stats[f'{prefix}/total_norm']:.6f}")
            
        except Exception as e:
            logger.warning(f"Gradient stats logging failed: {e}")
    
    def _create_gradient_histogram(self, gradients: np.ndarray, prefix: str):
        """Create and log gradient distribution histogram"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Full distribution
            ax1.hist(gradients, bins=50, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Gradient Value')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Gradient Distribution')
            ax1.set_yscale('log')
            
            # Zoomed distribution (remove extreme outliers)
            p5, p95 = np.percentile(gradients, [5, 95])
            filtered_grads = gradients[(gradients >= p5) & (gradients <= p95)]
            ax2.hist(filtered_grads, bins=50, alpha=0.7, edgecolor='black', color='orange')
            ax2.set_xlabel('Gradient Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Gradient Distribution (5th-95th percentile)')
            
            plt.tight_layout()
            
            # Log to W&B
            self.run.log({f"{prefix}_histogram": wandb.Image(fig)}, step=self.step_count)
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Gradient histogram creation failed: {e}")
    
    def log_learning_curves(self, train_metrics: Dict[str, float], 
                          val_metrics: Optional[Dict[str, float]] = None):
        """Log learning curves with interactive plots"""
        if not self.enabled:
            return
            
        try:
            # Log basic metrics
            log_dict = {}
            for key, value in train_metrics.items():
                log_dict[f"train/{key}"] = value
                
            if val_metrics:
                for key, value in val_metrics.items():
                    log_dict[f"val/{key}"] = value
            
            self.run.log(log_dict, step=self.step_count)
            
            logger.debug(f"📚 Logged learning curves: {len(log_dict)} metrics")
            
        except Exception as e:
            logger.warning(f"Learning curves logging failed: {e}")
    
    def log_model_artifacts(self, model_params: Dict[str, Any], 
                          model_path: Optional[str] = None):
        """Log model artifacts and architecture information"""
        if not self.enabled:
            return
            
        try:
            # Model parameter statistics
            param_stats = {}
            total_params = 0
            
            for name, params in model_params.items():
                if hasattr(params, 'shape'):
                    param_count = int(np.prod(params.shape))
                    total_params += param_count
                    
                    param_stats[f"model/params/{name}_count"] = param_count
                    param_stats[f"model/params/{name}_shape"] = str(params.shape)
                    
                    # Parameter statistics
                    params_np = np.array(params)
                    param_stats[f"model/params/{name}_norm"] = float(np.linalg.norm(params_np))
                    param_stats[f"model/params/{name}_mean"] = float(np.mean(params_np))
                    param_stats[f"model/params/{name}_std"] = float(np.std(params_np))
            
            param_stats["model/total_parameters"] = total_params
            self.run.log(param_stats, step=self.step_count)
            
            # Log model file if provided
            if model_path and os.path.exists(model_path):
                artifact = wandb.Artifact("model", type="model")
                artifact.add_file(model_path)
                self.run.log_artifact(artifact)
            
            logger.info(f"🧠 Logged model artifacts: {total_params:,} parameters")
            
        except Exception as e:
            logger.warning(f"Model artifacts logging failed: {e}")
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance issues and log alerts"""
        alerts = []
        
        # Memory usage alerts
        if metrics.memory_usage_mb > 8000:  # 8GB
            alerts.append(f"High memory usage: {metrics.memory_usage_mb:.1f}MB")
        
        if metrics.swap_usage_mb > 100:  # Any swap usage
            alerts.append(f"Swap usage detected: {metrics.swap_usage_mb:.1f}MB")
        
        # Latency alerts  
        if metrics.total_pipeline_ms > 100:  # Target: <100ms
            alerts.append(f"High latency: {metrics.total_pipeline_ms:.1f}ms")
        
        # CPU usage alerts
        if metrics.cpu_usage_percent > 90:
            alerts.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
        
        # Temperature alerts
        if metrics.temperature_celsius > 80:
            alerts.append(f"High temperature: {metrics.temperature_celsius:.1f}°C")
        
        # Log alerts
        if alerts:
            alert_msg = "; ".join(alerts)
            self.run.log({
                "alerts/performance_warning": alert_msg,
                "alerts/alert_count": len(alerts)
            }, step=self.step_count)
            
            logger.warning(f"⚠️  Performance alerts: {alert_msg}")
    
    def create_summary_dashboard(self):
        """Create comprehensive summary dashboard"""
        if not self.enabled or not self.enable_visualizations:
            return
            
        try:
            # Create summary plots
            if len(self.metrics_buffer) > 10:
                self._create_metrics_summary_plot()
            
            if len(self.gradient_history) > 10:
                self._create_gradient_history_plot()
            
            if len(self.spike_history) > 10:
                self._create_spike_history_plot()
            
            logger.info("📊 Created summary dashboard")
            
        except Exception as e:
            logger.warning(f"Summary dashboard creation failed: {e}")
    
    def _create_metrics_summary_plot(self):
        """Create comprehensive metrics summary plot"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Extract data
            steps = [m['step'] for m in self.metrics_buffer[-100:]]  # Last 100 steps
            
            # Plot 1: Spike rates over time
            if any('neuromorphic/spike_rate' in m for m in self.metrics_buffer):
                spike_rates = [m.get('neuromorphic/spike_rate', 0) for m in self.metrics_buffer[-100:]]
                axes[0, 0].plot(steps, spike_rates, 'b-', linewidth=2)
                axes[0, 0].set_title('Spike Rate Over Time')
                axes[0, 0].set_xlabel('Step')
                axes[0, 0].set_ylabel('Spike Rate')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Encoding efficiency
            if any('neuromorphic/encoding_fidelity' in m for m in self.metrics_buffer):
                fidelity = [m.get('neuromorphic/encoding_fidelity', 0) for m in self.metrics_buffer[-100:]]
                axes[0, 1].plot(steps, fidelity, 'g-', linewidth=2)
                axes[0, 1].set_title('Encoding Fidelity')
                axes[0, 1].set_xlabel('Step')
                axes[0, 1].set_ylabel('Fidelity')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Performance metrics
            if any('performance/inference_latency_ms' in m for m in self.metrics_buffer):
                latency = [m.get('performance/inference_latency_ms', 0) for m in self.metrics_buffer[-100:]]
                axes[1, 0].plot(steps, latency, 'r-', linewidth=2)
                axes[1, 0].axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Target: 100ms')
                axes[1, 0].set_title('Inference Latency')
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Latency (ms)')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
            
            # Plot 4: Memory usage
            if any('performance/memory_usage_mb' in m for m in self.metrics_buffer):
                memory = [m.get('performance/memory_usage_mb', 0) for m in self.metrics_buffer[-100:]]
                axes[1, 1].plot(steps, memory, 'purple', linewidth=2)
                axes[1, 1].set_title('Memory Usage')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Memory (MB)')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Log to W&B
            self.run.log({"summary_dashboard": wandb.Image(fig)}, step=self.step_count)
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Metrics summary plot failed: {e}")
    
    def _create_gradient_history_plot(self):
        """Create gradient norm history plot"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            steps = [h['step'] for h in self.gradient_history[-100:]]
            norms = [h['total_norm'] for h in self.gradient_history[-100:]]
            
            ax.plot(steps, norms, 'b-', linewidth=2, label='Gradient Norm')
            ax.set_xlabel('Step')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Norm History')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add gradient explosion warning line
            if max(norms) > 10:
                ax.axhline(y=10, color='r', linestyle='--', alpha=0.7, label='Warning: 10.0')
                ax.legend()
            
            plt.tight_layout()
            
            # Log to W&B
            self.run.log({"gradient_history": wandb.Image(fig)}, step=self.step_count)
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Gradient history plot failed: {e}")
    
    def _create_spike_history_plot(self):
        """Create spike activity history plot"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            steps = [h['step'] for h in self.spike_history[-100:]]
            rates = [h['spike_rate'] for h in self.spike_history[-100:]]
            counts = [h['spike_count'] for h in self.spike_history[-100:]]
            
            # Spike rates
            ax1.plot(steps, rates, 'g-', linewidth=2)
            ax1.set_ylabel('Spike Rate')
            ax1.set_title('Spike Activity History')
            ax1.grid(True, alpha=0.3)
            
            # Spike counts
            ax2.plot(steps, counts, 'orange', linewidth=2)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Spike Count')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Log to W&B
            self.run.log({"spike_history": wandb.Image(fig)}, step=self.step_count)
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Spike history plot failed: {e}")
    
    @contextmanager
    def log_step_context(self, step: Optional[int] = None):
        """Context manager for logging within a training step"""
        if step is not None:
            self.step_count = step
        
        start_time = time.time()
        
        try:
            yield self
        finally:
            # Log step timing
            step_time = (time.time() - start_time) * 1000  # ms
            if self.enabled:
                self.run.log({
                    "timing/step_duration_ms": step_time,
                    "timing/steps_per_second": 1000.0 / step_time if step_time > 0 else 0
                }, step=self.step_count)
            
            # Increment step counter
            self.step_count += 1
            
            # Periodic summary dashboard updates
            if self.step_count % (self.log_frequency * 10) == 0:
                self.create_summary_dashboard()
    
    def finish(self):
        """Finish logging and cleanup"""
        if not self.enabled:
            return
            
        try:
            # Create final summary
            self.create_summary_dashboard()
            
            # Log final metrics
            if self.metrics_buffer:
                final_metrics = {
                    "summary/total_steps": self.step_count,
                    "summary/experiment_duration_minutes": (time.time() - self.run._start_time) / 60,
                    "summary/average_step_time_ms": np.mean([
                        m.get('timing/step_duration_ms', 0) for m in self.metrics_buffer
                        if 'timing/step_duration_ms' in m
                    ]) if self.metrics_buffer else 0
                }
                self.run.log(final_metrics)
            
            # Finish W&B run
            self.run.finish()
            logger.info("🏁 Enhanced W&B logging finished")
            
        except Exception as e:
            logger.warning(f"W&B finish failed: {e}")

# Factory function for easy initialization
def create_enhanced_wandb_logger(project: str = "neuromorphic-gw-detection",
                                name: Optional[str] = None,
                                config: Optional[Dict[str, Any]] = None,
                                tags: Optional[List[str]] = None,
                                notes: Optional[str] = None,
                                output_dir: str = "wandb_outputs",
                                **kwargs) -> EnhancedWandbLogger:
    """Factory function to create enhanced W&B logger"""
    
    # Extract parameters that match EnhancedWandbLogger constructor
    valid_params = {
        'project': project,
        'name': name,
        'config': config,
        'tags': tags,
        'notes': notes,
        'output_dir': output_dir
    }
    
    # Add other parameters that match the constructor
    if 'enable_hardware_monitoring' in kwargs:
        valid_params['enable_hardware_monitoring'] = kwargs['enable_hardware_monitoring']
    if 'enable_visualizations' in kwargs:
        valid_params['enable_visualizations'] = kwargs['enable_visualizations']
    if 'enable_alerts' in kwargs:
        valid_params['enable_alerts'] = kwargs['enable_alerts']
    if 'log_frequency' in kwargs:
        valid_params['log_frequency'] = kwargs['log_frequency']
    
    return EnhancedWandbLogger(**valid_params)

# Convenience functions for common metrics
def create_neuromorphic_metrics(spike_rate: float = 0.0,
                               encoding_fidelity: float = 0.0,
                               contrastive_accuracy: float = 0.0,
                               **kwargs) -> NeuromorphicMetrics:
    """Create neuromorphic metrics object"""
    return NeuromorphicMetrics(
        spike_rate=spike_rate,
        encoding_fidelity=encoding_fidelity,
        contrastive_accuracy=contrastive_accuracy,
        **kwargs
    )

def create_performance_metrics(inference_latency_ms: float = 0.0,
                             memory_usage_mb: float = 0.0,
                             cpu_usage_percent: float = 0.0,
                             **kwargs) -> PerformanceMetrics:
    """Create performance metrics object"""
    return PerformanceMetrics(
        inference_latency_ms=inference_latency_ms,
        memory_usage_mb=memory_usage_mb,
        cpu_usage_percent=cpu_usage_percent,
        **kwargs
    )

if __name__ == "__main__":
    # Example usage
    logger = create_enhanced_wandb_logger(
        project="test-neuromorphic-logging",
        name="test-run",
        config={"learning_rate": 0.001, "batch_size": 32}
    )
    
    # Test logging
    with logger.log_step_context(step=0):
        # Log neuromorphic metrics
        neuro_metrics = create_neuromorphic_metrics(
            spike_rate=15.2,
            encoding_fidelity=0.85,
            contrastive_accuracy=0.92
        )
        logger.log_neuromorphic_metrics(neuro_metrics)
        
        # Log performance metrics
        perf_metrics = create_performance_metrics(
            inference_latency_ms=45.2,
            memory_usage_mb=2048,
            cpu_usage_percent=65.3
        )
        logger.log_performance_metrics(perf_metrics)
    
    logger.finish()
    print("✅ Enhanced W&B logger test completed!") 