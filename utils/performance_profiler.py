"""
JAX Performance Profiler for Neuromorphic GW Detection

Comprehensive benchmarking and optimization framework for achieving <100ms inference.
Implements JAX profiler integration, memory monitoring, and performance optimization.

Key features:
- JAX profiler integration for detailed performance analysis
- Real-time inference benchmarking with statistical validation
- Memory usage monitoring and optimization
- Apple Silicon Metal backend profiling
- Automated performance optimization recommendations
"""

import jax
import jax.numpy as jnp
import jax.profiler
import numpy as np
import time
import psutil
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import matplotlib.pyplot as plt

# Optional plotting dependency
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for inference benchmarking"""
    
    # Timing metrics (milliseconds)
    inference_time_ms: float
    cpc_time_ms: float
    spike_time_ms: float
    snn_time_ms: float
    total_pipeline_ms: float
    
    # Memory metrics (MB)
    peak_memory_mb: float
    memory_growth_mb: float
    gpu_memory_mb: float
    
    # Throughput metrics
    samples_per_second: float
    batch_throughput: float
    
    # Quality metrics
    accuracy: float
    precision: float
    recall: float
    
    # System metrics
    cpu_usage_percent: float
    gpu_utilization_percent: float
    temperature_celsius: Optional[float] = None

@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarking"""
    
    # Test parameters
    batch_sizes: List[int] = None
    num_warmup_runs: int = 10
    num_benchmark_runs: int = 100
    target_inference_ms: float = 100.0
    
    # Profiling options
    enable_jax_profiler: bool = True
    profiler_output_dir: str = "performance_profiles"
    capture_memory_timeline: bool = True
    
    # Platform specific
    device_type: str = "metal"  # metal, cpu, gpu
    precision: str = "float32"
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16, 32]

class JAXPerformanceProfiler:
    """Comprehensive JAX performance profiler for neuromorphic GW detection"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.setup_profiling_environment()
        self.benchmark_results = []
        
    def setup_profiling_environment(self):
        """Setup JAX profiling environment and Metal backend"""
        
        # Setup profiler output directory
        self.profile_dir = Path(self.config.profiler_output_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure JAX for optimal profiling
        import os
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'  # Prevent swap
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        os.environ['JAX_PROFILER_PORT'] = '9999'
        
        # Metal backend optimization (simplified flags)
        if self.config.device_type == "metal":
            os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
        
        logger.info(f"🔧 Profiling environment setup:")
        logger.info(f"   Device: {jax.devices()}")
        logger.info(f"   Platform: {jax.lib.xla_bridge.get_backend().platform}")
        logger.info(f"   Memory fraction: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')}")
        logger.info(f"   Profile dir: {self.profile_dir}")
    
    @contextmanager
    def jax_profiler_context(self, trace_name: str):
        """Context manager for JAX profiler tracing"""
        
        if not self.config.enable_jax_profiler:
            yield
            return
        
        try:
            # Start profiling
            trace_dir = self.profile_dir / trace_name
            jax.profiler.start_trace(str(trace_dir))
            logger.info(f"🔬 Started JAX profiler trace: {trace_name}")
            
            yield
            
        finally:
            # Stop profiling
            jax.profiler.stop_trace()
            logger.info(f"✅ JAX profiler trace saved: {trace_dir}")
    
    def benchmark_full_pipeline(self, 
                               model_components: Dict[str, Any],
                               test_data: jnp.ndarray) -> Dict[str, PerformanceMetrics]:
        """
        🚀 Comprehensive pipeline benchmarking with <100ms target
        
        Args:
            model_components: Dict with 'cpc_encoder', 'spike_bridge', 'snn_classifier'
            test_data: Test strain data [batch_size, sequence_length]
            
        Returns:
            Performance metrics for each batch size
        """
        
        logger.info("🏃‍♂️ COMPREHENSIVE PIPELINE BENCHMARKING")
        logger.info("=" * 60)
        logger.info(f"Target inference time: <{self.config.target_inference_ms}ms")
        
        all_results = {}
        
        for batch_size in self.config.batch_sizes:
            logger.info(f"\n📊 Benchmarking batch size: {batch_size}")
            
            # Prepare test batch
            if test_data.shape[0] < batch_size:
                # Repeat data if not enough samples
                multiplier = (batch_size // test_data.shape[0]) + 1
                batch_data = jnp.tile(test_data, (multiplier, 1))[:batch_size]
            else:
                batch_data = test_data[:batch_size]
            
            # Run benchmark for this batch size
            with self.jax_profiler_context(f"batch_size_{batch_size}"):
                metrics = self._benchmark_single_batch_size(
                    model_components, batch_data, batch_size
                )
            
            all_results[f"batch_{batch_size}"] = metrics
            
            # Log results
            self._log_benchmark_results(batch_size, metrics)
        
        # Generate performance summary
        self._generate_performance_summary(all_results)
        
        return all_results
    
    def _benchmark_single_batch_size(self,
                                   model_components: Dict[str, Any],
                                   batch_data: jnp.ndarray,
                                   batch_size: int) -> PerformanceMetrics:
        """Benchmark single batch size with detailed timing"""
        
        # Extract model components
        cpc_encoder = model_components['cpc_encoder']
        cpc_params = model_components['cpc_params']
        spike_bridge = model_components['spike_bridge']
        spike_params = model_components['spike_params']
        snn_classifier = model_components['snn_classifier']
        snn_params = model_components['snn_params']
        
        # 🔧 Create JIT-compiled pipeline functions
        @jax.jit
        def cpc_forward(data):
            return cpc_encoder.apply(cpc_params, data)
        
        @jax.jit
        def spike_forward(features):
            return spike_bridge.apply(spike_params, features)
        
        @jax.jit
        def snn_forward(spikes):
            return snn_classifier.apply(snn_params, spikes)
        
        @jax.jit
        def full_pipeline(data):
            features = cpc_forward(data)
            spikes = spike_forward(features)
            predictions = snn_forward(spikes)
            return predictions
        
        # 🔧 Warmup phase
        logger.info(f"   🔥 Warmup phase ({self.config.num_warmup_runs} runs)...")
        initial_memory = self._get_memory_usage()
        
        for _ in range(self.config.num_warmup_runs):
            _ = full_pipeline(batch_data)
        
        # Ensure all operations complete
        jax.block_until_ready(_)
        
        post_warmup_memory = self._get_memory_usage()
        memory_growth = post_warmup_memory['total_mb'] - initial_memory['total_mb']
        
        # 🔧 Benchmark phase with detailed timing
        logger.info(f"   ⏱️  Benchmark phase ({self.config.num_benchmark_runs} runs)...")
        
        # Timing lists
        cpc_times = []
        spike_times = []
        snn_times = []
        total_times = []
        
        peak_memory = initial_memory['total_mb']
        
        for run in range(self.config.num_benchmark_runs):
            # Memory monitoring
            current_memory = self._get_memory_usage()
            peak_memory = max(peak_memory, current_memory['total_mb'])
            
            # Individual component timing
            start = time.perf_counter()
            cpc_features = cpc_forward(batch_data)
            jax.block_until_ready(cpc_features)
            cpc_time = (time.perf_counter() - start) * 1000  # ms
            
            start = time.perf_counter()
            spikes = spike_forward(cpc_features)
            jax.block_until_ready(spikes)
            spike_time = (time.perf_counter() - start) * 1000  # ms
            
            start = time.perf_counter()
            predictions = snn_forward(spikes)
            jax.block_until_ready(predictions)
            snn_time = (time.perf_counter() - start) * 1000  # ms
            
            # Full pipeline timing
            start = time.perf_counter()
            _ = full_pipeline(batch_data)
            jax.block_until_ready(_)
            total_time = (time.perf_counter() - start) * 1000  # ms
            
            cpc_times.append(cpc_time)
            spike_times.append(spike_time)
            snn_times.append(snn_time)
            total_times.append(total_time)
        
        # 🔧 Compute statistics
        final_memory = self._get_memory_usage()
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            inference_time_ms=np.mean(total_times),
            cpc_time_ms=np.mean(cpc_times),
            spike_time_ms=np.mean(spike_times),
            snn_time_ms=np.mean(snn_times),
            total_pipeline_ms=np.mean(total_times),
            
            peak_memory_mb=peak_memory,
            memory_growth_mb=memory_growth,
            gpu_memory_mb=final_memory.get('gpu_mb', 0),
            
            samples_per_second=batch_size / (np.mean(total_times) / 1000),
            batch_throughput=1000 / np.mean(total_times),  # batches per second
            
            accuracy=0.0,  # Would need real labels for this
            precision=0.0,
            recall=0.0,
            
            cpu_usage_percent=cpu_percent,
            gpu_utilization_percent=0.0  # Would need GPU monitoring
        )
        
        # Store detailed timing statistics
        metrics.timing_stats = {
            'cpc_std': np.std(cpc_times),
            'spike_std': np.std(spike_times),
            'snn_std': np.std(snn_times),
            'total_std': np.std(total_times),
            'min_time': np.min(total_times),
            'max_time': np.max(total_times),
            'p95_time': np.percentile(total_times, 95),
            'p99_time': np.percentile(total_times, 99)
        }
        
        return metrics
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'total_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'gpu_mb': 0.0  # Would need GPU monitoring for this
        }
    
    def _log_benchmark_results(self, batch_size: int, metrics: PerformanceMetrics):
        """Log benchmark results for single batch size"""
        
        logger.info(f"   📊 Results for batch size {batch_size}:")
        logger.info(f"      Total inference: {metrics.inference_time_ms:.2f}ms")
        logger.info(f"      - CPC encoder:   {metrics.cpc_time_ms:.2f}ms")
        logger.info(f"      - Spike bridge:  {metrics.spike_time_ms:.2f}ms")
        logger.info(f"      - SNN classifier:{metrics.snn_time_ms:.2f}ms")
        logger.info(f"      Memory peak:     {metrics.peak_memory_mb:.1f}MB")
        logger.info(f"      Throughput:      {metrics.samples_per_second:.1f} samples/s")
        
        # Performance assessment
        target_met = metrics.inference_time_ms < self.config.target_inference_ms
        status = "✅ PASS" if target_met else "❌ FAIL"
        logger.info(f"      Target <{self.config.target_inference_ms}ms: {status}")
        
        if hasattr(metrics, 'timing_stats'):
            stats = metrics.timing_stats
            logger.info(f"      P95 latency:     {stats['p95_time']:.2f}ms")
            logger.info(f"      P99 latency:     {stats['p99_time']:.2f}ms")
    
    def _generate_performance_summary(self, all_results: Dict[str, PerformanceMetrics]):
        """Generate comprehensive performance summary"""
        
        logger.info("\n" + "=" * 60)
        logger.info("🎯 PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        
        # Find optimal batch size
        optimal_batch_size = None
        best_throughput = 0
        
        for batch_key, metrics in all_results.items():
            batch_size = int(batch_key.split('_')[1])
            
            if (metrics.inference_time_ms < self.config.target_inference_ms and 
                metrics.samples_per_second > best_throughput):
                best_throughput = metrics.samples_per_second
                optimal_batch_size = batch_size
        
        # Performance recommendations
        if optimal_batch_size:
            logger.info(f"🏆 Optimal batch size: {optimal_batch_size}")
            logger.info(f"   Best throughput: {best_throughput:.1f} samples/s")
            logger.info(f"   Inference time: {all_results[f'batch_{optimal_batch_size}'].inference_time_ms:.2f}ms")
        else:
            logger.info("⚠️  No batch size meets <100ms target")
            
            # Find closest to target
            min_time = float('inf')
            closest_batch = None
            for batch_key, metrics in all_results.items():
                if metrics.inference_time_ms < min_time:
                    min_time = metrics.inference_time_ms
                    closest_batch = int(batch_key.split('_')[1])
            
            logger.info(f"📈 Closest performance: batch size {closest_batch}")
            logger.info(f"   Inference time: {min_time:.2f}ms")
            logger.info(f"   Target miss: +{min_time - self.config.target_inference_ms:.2f}ms")
        
        # Generate optimization recommendations
        self._generate_optimization_recommendations(all_results)
        
        # Save results to file
        self._save_benchmark_results(all_results)
    
    def _generate_optimization_recommendations(self, 
                                             all_results: Dict[str, PerformanceMetrics]):
        """Generate specific optimization recommendations"""
        
        logger.info("\n🔧 OPTIMIZATION RECOMMENDATIONS:")
        
        # Analyze bottlenecks
        sample_metrics = list(all_results.values())[0]
        
        total_time = sample_metrics.inference_time_ms
        cpc_ratio = sample_metrics.cpc_time_ms / total_time
        spike_ratio = sample_metrics.spike_time_ms / total_time
        snn_ratio = sample_metrics.snn_time_ms / total_time
        
        if cpc_ratio > 0.5:
            logger.info("   🎯 CPC Encoder bottleneck (>50% time)")
            logger.info("      → Consider reducing latent dimensions")
            logger.info("      → Optimize convolutional layers")
            logger.info("      → Enable gradient checkpointing")
        
        if spike_ratio > 0.3:
            logger.info("   ⚡ Spike Bridge bottleneck (>30% time)")
            logger.info("      → Optimize temporal contrast encoding")
            logger.info("      → Reduce time steps if possible")
            logger.info("      → Consider vectorized operations")
        
        if snn_ratio > 0.3:
            logger.info("   🧠 SNN Classifier bottleneck (>30% time)")
            logger.info("      → Reduce network depth if possible")
            logger.info("      → Optimize surrogate gradients")
            logger.info("      → Consider simplified LIF dynamics")
        
        # Memory recommendations
        if sample_metrics.peak_memory_mb > 8000:  # 8GB
            logger.info("   💾 High memory usage detected")
            logger.info("      → Reduce batch size")
            logger.info("      → Enable gradient checkpointing")
            logger.info("      → Consider mixed precision")
    
    def _save_benchmark_results(self, all_results: Dict[str, PerformanceMetrics]):
        """Save benchmark results to JSON file"""
        
        # Convert to serializable format
        serializable_results = {}
        for key, metrics in all_results.items():
            serializable_results[key] = asdict(metrics)
        
        # Save to file
        results_file = self.profile_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'config': asdict(self.config),
                'results': serializable_results,
                'summary': {
                    'timestamp': time.time(),
                    'device': str(jax.devices()[0]),
                    'platform': jax.lib.xla_bridge.get_backend().platform
                }
            }, f, indent=2)
        
        logger.info(f"💾 Benchmark results saved: {results_file}")

# Factory function
def create_performance_profiler(target_inference_ms: float = 100.0,
                               device_type: str = "metal") -> JAXPerformanceProfiler:
    """Factory function to create performance profiler"""
    
    config = BenchmarkConfig(
        target_inference_ms=target_inference_ms,
        device_type=device_type,
        enable_jax_profiler=True
    )
    
    return JAXPerformanceProfiler(config) 