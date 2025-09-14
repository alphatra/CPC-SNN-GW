"""
Metric classes for neuromorphic and performance monitoring.

Extracted from wandb_enhanced_logger.py for better modularity.
"""

import platform
import sys
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict


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
    cuda_version: str = ""
    
    # Hardware info
    cpu_model: str = ""
    cpu_cores: int = 0
    total_memory_gb: float = 0.0
    gpu_model: str = ""
    gpu_memory_gb: float = 0.0
    
    # Environment
    hostname: str = ""
    username: str = ""
    working_directory: str = ""
    command_line: str = ""
    
    def __post_init__(self):
        """Auto-populate system information"""
        import os
        import psutil
        
        self.platform = platform.platform()
        self.python_version = sys.version.split()[0]
        self.hostname = platform.node()
        self.username = os.getenv('USER', 'unknown')
        self.working_directory = os.getcwd()
        self.command_line = ' '.join(sys.argv)
        
        # Hardware info
        self.cpu_model = platform.processor() or "Unknown"
        self.cpu_cores = psutil.cpu_count()
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Version info
        try:
            import jax
            self.jax_version = jax.__version__
        except ImportError:
            self.jax_version = "Not installed"
        
        try:
            result = subprocess.run(['nvidia-smi', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.cuda_version = result.stdout.split('\n')[0]
        except (subprocess.SubprocessError, FileNotFoundError):
            self.cuda_version = "Not available"
