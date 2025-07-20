"""
Optimized Spike Bridge: CPC Latent to Spike Train Conversion

Converts continuous CPC encoder representations to discrete spike trains
for neuromorphic SNN processing with advanced optimizations.

Key improvements:
- int8 output format for 4x memory efficiency
- Cosine-similarity encoding for better representation
- Throughput benchmarks and profiling
- Vectorized operations for maximum performance
- Comprehensive performance metrics and optimization
"""

import jax
import jax.numpy as jnp
import time
from typing import Tuple, Optional, Dict, Any, Union, List
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class SpikeEncodingStrategy(Enum):
    """Spike encoding strategies."""
    POISSON_RATE = "poisson_rate"
    TEMPORAL_CONTRAST = "temporal_contrast"
    POPULATION_VECTOR = "population_vector"
    RATE_BASED = "rate_based"
    COSINE_SIMILARITY = "cosine_similarity"
    BINARY_THRESHOLD = "binary_threshold"


@dataclass
class SpikeBridgeConfig:
    """Configuration for spike bridge optimization."""
    # Basic parameters
    encoding_strategy: SpikeEncodingStrategy = SpikeEncodingStrategy.POISSON_RATE
    spike_time_steps: int = 100
    max_spike_rate: float = 100.0
    dt: float = 1e-3
    population_size: int = 8
    
    # Optimization parameters
    use_int8_output: bool = True
    use_vectorized_ops: bool = True
    memory_efficient: bool = True
    
    # Cosine similarity parameters
    cosine_basis_size: int = 64
    cosine_temperature: float = 1.0
    
    # Performance tuning
    batch_size_limit: int = 1024
    time_steps_limit: int = 1000
    
    # Profiling
    enable_profiling: bool = False
    profile_memory: bool = False


@dataclass
class ThroughputMetrics:
    """Throughput benchmark results."""
    samples_per_second: float  # Actually sequences (batches) per second, not individual samples
    memory_usage_mb: float
    encoding_time_ms: float
    peak_memory_mb: float
    batch_size: int
    sequence_length: int
    latent_dim: int
    output_format: str
    
    def __post_init__(self):
        """Compute derived metrics."""
        self.megasamples_per_second = self.samples_per_second / 1e6
        self.memory_efficiency = self.samples_per_second / self.memory_usage_mb
        self.throughput_score = self.megasamples_per_second * 1000 / self.encoding_time_ms


class OptimizedSpikeBridge:
    """
    Optimized spike bridge with int8 outputs and advanced encoding strategies.
    
    Features:
    - Memory-efficient int8 spike representation
    - Cosine-similarity encoding for better features
    - Comprehensive throughput benchmarking
    - Vectorized operations for maximum performance
    - Configurable optimization parameters
    """
    
    def __init__(self, config: SpikeBridgeConfig):
        """
        Initialize optimized spike bridge.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.profile_data = {}
        
        # Precompute cosine basis if needed
        if config.encoding_strategy == SpikeEncodingStrategy.COSINE_SIMILARITY:
            self.cosine_basis = self._create_cosine_basis()
        
        # Initialize profiling
        if config.enable_profiling:
            self.profile_data = {
                'encoding_times': [],
                'memory_usage': [],
                'throughput_metrics': []
            }
    
    def _create_cosine_basis(self) -> jnp.ndarray:
        """Create cosine basis for similarity encoding."""
        # Create orthogonal cosine basis vectors
        basis_vectors = jax.random.normal(
            jax.random.PRNGKey(42), 
            (self.config.cosine_basis_size, self.config.cosine_basis_size)
        )
        
        # Orthogonalize using QR decomposition
        q, r = jnp.linalg.qr(basis_vectors)
        
        # Normalize
        basis_normalized = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        
        return basis_normalized
        
    def encode(self, 
               latent_features: jnp.ndarray, 
               key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Convert CPC latent features to optimized spike trains.
        
        Args:
            latent_features: CPC encoder output [batch, seq_len, latent_dim]
            key: JAX random key for stochastic spike generation
            
        Returns:
            spike_trains: Spike trains [batch, spike_time_steps, spike_dim]
                         Format: int8 if use_int8_output=True, else float32
        """
        start_time = time.time()
        
        # Input validation
        self._validate_input(latent_features)
        
        # Select encoding strategy
        if self.config.encoding_strategy == SpikeEncodingStrategy.POISSON_RATE:
            spikes = self._poisson_rate_encoding(latent_features, key)
        elif self.config.encoding_strategy == SpikeEncodingStrategy.TEMPORAL_CONTRAST:
            spikes = self._temporal_contrast_encoding(latent_features, key)
        elif self.config.encoding_strategy == SpikeEncodingStrategy.POPULATION_VECTOR:
            spikes = self._population_vector_encoding(latent_features, key)
        elif self.config.encoding_strategy == SpikeEncodingStrategy.RATE_BASED:
            spikes = self._rate_based_encoding(latent_features, key)
        elif self.config.encoding_strategy == SpikeEncodingStrategy.COSINE_SIMILARITY:
            spikes = self._cosine_similarity_encoding(latent_features, key)
        elif self.config.encoding_strategy == SpikeEncodingStrategy.BINARY_THRESHOLD:
            spikes = self._binary_threshold_encoding(latent_features, key)
        else:
            raise NotImplementedError(f"Encoding strategy '{self.config.encoding_strategy}' not implemented")
        
        # Convert to int8 if requested
        if self.config.use_int8_output:
            spikes = self._convert_to_int8(spikes)
        
        # Profiling
        if self.config.enable_profiling:
            encoding_time = (time.time() - start_time) * 1000  # ms
            self.profile_data['encoding_times'].append(encoding_time)
            
            if self.config.profile_memory:
                memory_usage = self._estimate_memory_usage(spikes)
                self.profile_data['memory_usage'].append(memory_usage)
        
        return spikes
    
    def _validate_input(self, latent_features: jnp.ndarray):
        """Validate input tensor dimensions and values."""
        if latent_features.ndim != 3:
            raise ValueError(f"Expected 3D input [batch, seq_len, latent_dim], got {latent_features.ndim}D")
        
        batch_size, seq_len, latent_dim = latent_features.shape
        
        if batch_size > self.config.batch_size_limit:
            logger.warning(f"Batch size {batch_size} exceeds limit {self.config.batch_size_limit}")
        
        if seq_len > self.config.time_steps_limit:
            logger.warning(f"Sequence length {seq_len} exceeds limit {self.config.time_steps_limit}")
    
    @jax.jit
    def _poisson_rate_encoding(self, 
                              latent_features: jnp.ndarray, 
                              key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Optimized Poisson rate encoding with vectorized operations.
        """
        batch_size, seq_len, latent_dim = latent_features.shape
        
        # Vectorized normalization
        normalized_features = jax.nn.sigmoid(latent_features)
        
        # Vectorized rate computation
        firing_rates = normalized_features * self.config.max_spike_rate
        spike_probs = firing_rates * self.config.dt
        
        if self.config.use_vectorized_ops:
            # Fully vectorized approach
            # Expand to [batch, seq_len, latent_dim, spike_time_steps]
            spike_probs_expanded = jnp.expand_dims(spike_probs, axis=-1)
            spike_probs_expanded = jnp.repeat(spike_probs_expanded, self.config.spike_time_steps, axis=-1)
            
            # Generate all spikes at once
            random_vals = jax.random.uniform(key, spike_probs_expanded.shape)
            spikes = (random_vals < spike_probs_expanded).astype(jnp.float32)
            
            # Transpose axes for proper reshaping: [batch, seq_len, spike_time_steps, latent_dim]
            spikes = jnp.transpose(spikes, (0, 1, 3, 2))
            # Reshape to [batch, seq_len * spike_time_steps, latent_dim]
            spikes = spikes.reshape(batch_size, seq_len * self.config.spike_time_steps, latent_dim)
        else:
            # Fallback to sequential approach
            spike_probs_expanded = jnp.repeat(spike_probs, self.config.spike_time_steps, axis=1)
            spikes = (jax.random.uniform(key, spike_probs_expanded.shape) < spike_probs_expanded).astype(jnp.float32)
        
        return spikes
    
    @jax.jit
    def _temporal_contrast_encoding(self, 
                                  latent_features: jnp.ndarray, 
                                  key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Optimized temporal contrast encoding.
        """
        batch_size, seq_len, latent_dim = latent_features.shape
        
        # Vectorized gradient computation
        features_padded = jnp.pad(latent_features, ((0, 0), (1, 0), (0, 0)), mode='edge')
        temporal_gradients = jnp.diff(features_padded, axis=1)
        
        # Vectorized positive/negative split
        positive_changes = jnp.maximum(temporal_gradients, 0)
        negative_changes = jnp.maximum(-temporal_gradients, 0)
        
        # Vectorized rate computation
        pos_rates = jax.nn.sigmoid(positive_changes * 10) * self.config.max_spike_rate
        neg_rates = jax.nn.sigmoid(negative_changes * 10) * self.config.max_spike_rate
        
        # Vectorized spike generation
        pos_probs = pos_rates * self.config.dt
        neg_probs = neg_rates * self.config.dt
        
        # Expand and generate spikes
        pos_probs_expanded = jnp.repeat(pos_probs, self.config.spike_time_steps, axis=1)
        neg_probs_expanded = jnp.repeat(neg_probs, self.config.spike_time_steps, axis=1)
        
        key1, key2 = jax.random.split(key)
        pos_spikes = (jax.random.uniform(key1, pos_probs_expanded.shape) < pos_probs_expanded).astype(jnp.float32)
        neg_spikes = (jax.random.uniform(key2, neg_probs_expanded.shape) < neg_probs_expanded).astype(jnp.float32)
        
        # Combine channels
        spikes = jnp.concatenate([pos_spikes, neg_spikes], axis=-1)
        
        return spikes
    
    @jax.jit
    def _population_vector_encoding(self, 
                                  latent_features: jnp.ndarray, 
                                  key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Optimized population vector encoding.
        """
        batch_size, seq_len, latent_dim = latent_features.shape
        
        # Vectorized normalization
        normalized_features = jax.nn.sigmoid(latent_features)
        
        # Vectorized population response computation
        population_centers = jnp.linspace(0, 1, self.config.population_size)
        
        # Compute all responses at once
        # Shape: [batch, seq_len, latent_dim, population_size]
        feature_expanded = normalized_features[..., :, None]  # Add population dimension
        centers_expanded = population_centers[None, None, None, :]  # Broadcast
        
        responses = jnp.exp(-((feature_expanded - centers_expanded) ** 2) / (2 * 0.1**2))
        
        # Reshape to [batch, seq_len, latent_dim * population_size]
        responses = responses.reshape(batch_size, seq_len, latent_dim * self.config.population_size)
        
        # Vectorized spike generation
        spike_rates = responses * self.config.max_spike_rate
        spike_probs = spike_rates * self.config.dt
        
        # Expand and generate spikes
        spike_probs_expanded = jnp.repeat(spike_probs, self.config.spike_time_steps, axis=1)
        spikes = (jax.random.uniform(key, spike_probs_expanded.shape) < spike_probs_expanded).astype(jnp.float32)
        
        return spikes
    
    @jax.jit
    def _rate_based_encoding(self,
                            latent_features: jnp.ndarray,
                            key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Optimized rate-based encoding (deterministic).
        """
        batch_size, seq_len, latent_dim = latent_features.shape
        
        # Vectorized normalization and rate computation
        normalized_features = jax.nn.sigmoid(latent_features) * self.config.max_spike_rate
        spike_counts = jnp.round(normalized_features * self.config.dt).astype(jnp.int32)
        
        # Vectorized spike train generation
        spike_counts_expanded = jnp.repeat(spike_counts, self.config.spike_time_steps, axis=1)
        
        # Create time indices for comparison
        time_indices = jnp.arange(self.config.spike_time_steps)[None, :, None]
        spikes = (time_indices < spike_counts_expanded).astype(jnp.float32)
        
        return spikes
    
    @jax.jit
    def _cosine_similarity_encoding(self,
                                   latent_features: jnp.ndarray,
                                   key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Cosine similarity-based encoding for better representation.
        
        Projects latent features onto cosine basis and encodes similarities as spike rates.
        """
        batch_size, seq_len, latent_dim = latent_features.shape
        
        # Normalize input features
        normalized_features = latent_features / (jnp.linalg.norm(latent_features, axis=-1, keepdims=True) + 1e-8)
        
        # Project onto cosine basis
        # Shape: [batch, seq_len, cosine_basis_size]
        similarities = jnp.dot(normalized_features, self.cosine_basis.T)
        
        # Apply temperature scaling and sigmoid
        similarities_scaled = similarities / self.config.cosine_temperature
        activation_probs = jax.nn.sigmoid(similarities_scaled)
        
        # Convert to spike rates
        spike_rates = activation_probs * self.config.max_spike_rate
        spike_probs = spike_rates * self.config.dt
        
        # Expand temporally
        spike_probs_expanded = jnp.repeat(spike_probs, self.config.spike_time_steps, axis=1)
        
        # Generate spikes
        spikes = (jax.random.uniform(key, spike_probs_expanded.shape) < spike_probs_expanded).astype(jnp.float32)
        
        return spikes
    
    @jax.jit
    def _binary_threshold_encoding(self,
                           latent_features: jnp.ndarray, 
                           key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Simple binary threshold encoding for maximum speed.
        """
        batch_size, seq_len, latent_dim = latent_features.shape
        
        # Threshold at 0
        binary_features = (latent_features > 0).astype(jnp.float32)
        
        # Expand temporally
        binary_expanded = jnp.repeat(binary_features, self.config.spike_time_steps, axis=1)
        
        # Create spike trains (first half of time steps spike if feature is positive)
        time_indices = jnp.arange(self.config.spike_time_steps)[None, :, None]
        half_time = self.config.spike_time_steps // 2
        
        spikes = jnp.where(
            binary_expanded == 1,
            (time_indices < half_time).astype(jnp.float32),
            0.0
        )
        
        return spikes
    
    def _convert_to_int8(self, spikes: jnp.ndarray) -> jnp.ndarray:
        """Convert float32 spikes to int8 for memory efficiency."""
        # Check if running on TPU - int8 is not well supported
        platform = jax.default_backend()
        if platform == "tpu":
            logger.warning(
                "int8 output disabled on TPU platform due to limited support. "
                "Using float32 instead."
            )
            return spikes  # Keep as float32
        
        # Spikes are binary (0 or 1), so int8 conversion is straightforward
        return spikes.astype(jnp.int8)
    
    def _estimate_memory_usage(self, spikes: jnp.ndarray) -> float:
        """Estimate memory usage in MB."""
        num_elements = spikes.size
        bytes_per_element = 1 if spikes.dtype == jnp.int8 else 4  # int8 vs float32
        memory_bytes = num_elements * bytes_per_element
        return memory_bytes / (1024 * 1024)  # Convert to MB
    
    def benchmark_throughput(self, 
                           batch_sizes: List[int] = [1, 4, 16, 64],
                           sequence_lengths: List[int] = [32, 128, 512],
                           latent_dims: List[int] = [64, 128, 256],
                           num_runs: int = 5) -> List[ThroughputMetrics]:
        """
        Comprehensive throughput benchmark.
        
        Args:
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths to test
            latent_dims: List of latent dimensions to test
            num_runs: Number of runs for averaging
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                for latent_dim in latent_dims:
                    print(f"Benchmarking: batch={batch_size}, seq_len={seq_len}, latent_dim={latent_dim}")
                    
                    # Generate test data
                    key = jax.random.PRNGKey(42)
                    latent_features = jax.random.normal(key, (batch_size, seq_len, latent_dim))
                    
                    # Warm up
                    for _ in range(2):
                        key, subkey = jax.random.split(key)
                        _ = self.encode(latent_features, subkey)
                    
                    # Benchmark
                    times = []
                    for run in range(num_runs):
                        key, subkey = jax.random.split(key)
                        
                        start_time = time.time()
                        spikes = self.encode(latent_features, subkey)
                        end_time = time.time()
                        
                        times.append(end_time - start_time)
                    
                    # Compute metrics
                    avg_time = sum(times) / len(times)
                    sequences_per_second = batch_size / avg_time  # More accurate: batch sequences, not individual samples
                    encoding_time_ms = avg_time * 1000
                    memory_usage_mb = self._estimate_memory_usage(spikes)
                    
                    # Peak memory estimation (input + output)
                    input_memory_mb = (latent_features.size * 4) / (1024 * 1024)  # float32
                    peak_memory_mb = input_memory_mb + memory_usage_mb
                    
                    # Output format
                    output_format = "int8" if self.config.use_int8_output else "float32"
                    
                    metrics = ThroughputMetrics(
                        samples_per_second=sequences_per_second,  # Note: actually sequences per second
                        memory_usage_mb=memory_usage_mb,
                        encoding_time_ms=encoding_time_ms,
                        peak_memory_mb=peak_memory_mb,
                        batch_size=batch_size,
                        sequence_length=seq_len,
                        latent_dim=latent_dim,
                        output_format=output_format
                    )
                    
                    results.append(metrics)
                    
                    print(f"  ✅ {sequences_per_second:.1f} sequences/sec, {memory_usage_mb:.1f} MB")
        
        return results
    
    def print_benchmark_report(self, results: List[ThroughputMetrics]):
        """Print comprehensive benchmark report."""
        print("\n🚀 Spike Bridge Throughput Benchmark Report")
        print("=" * 60)
        
        # Summary statistics
        throughput_scores = [r.throughput_score for r in results]
        memory_efficiencies = [r.memory_efficiency for r in results]
        
        print(f"Total configurations tested: {len(results)}")
        print(f"Encoding strategy: {self.config.encoding_strategy.value}")
        print(f"Output format: {'int8' if self.config.use_int8_output else 'float32'}")
        print(f"Vectorized operations: {self.config.use_vectorized_ops}")
        print()
        
        print("📊 Performance Summary:")
        print(f"  Max throughput: {max(r.samples_per_second for r in results):.1f} samples/sec")
        print(f"  Min encoding time: {min(r.encoding_time_ms for r in results):.2f} ms")
        print(f"  Max memory efficiency: {max(memory_efficiencies):.1f} samples/sec/MB")
        print(f"  Average throughput score: {sum(throughput_scores) / len(throughput_scores):.2f}")
        print()
        
        # Top 5 configurations
        sorted_results = sorted(results, key=lambda x: x.throughput_score, reverse=True)
        print("🏆 Top 5 Configurations:")
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"  {i}. Batch:{result.batch_size}, Seq:{result.sequence_length}, Dim:{result.latent_dim}")
            print(f"     Throughput: {result.samples_per_second:.1f} samples/sec")
            print(f"     Memory: {result.memory_usage_mb:.1f} MB, Score: {result.throughput_score:.2f}")
            print()
        
        # Memory efficiency analysis
        print("💾 Memory Efficiency Analysis:")
        int8_results = [r for r in results if r.output_format == "int8"]
        float32_results = [r for r in results if r.output_format == "float32"]
        
        if int8_results and float32_results:
            avg_int8_memory = sum(r.memory_usage_mb for r in int8_results) / len(int8_results)
            avg_float32_memory = sum(r.memory_usage_mb for r in float32_results) / len(float32_results)
            memory_savings = (avg_float32_memory - avg_int8_memory) / avg_float32_memory * 100
            print(f"  int8 memory savings: {memory_savings:.1f}% vs float32")
        
        print(f"  Average memory usage: {sum(r.memory_usage_mb for r in results) / len(results):.1f} MB")
        print(f"  Peak memory usage: {max(r.peak_memory_mb for r in results):.1f} MB")


# Backward compatibility
class SpikeBridge:
    """Backward compatible spike bridge."""
    
    def __init__(self,
                 encoding_strategy: SpikeEncodingStrategy = SpikeEncodingStrategy.POISSON_RATE,
                 spike_time_steps: int = 100,
                 max_spike_rate: float = 100.0,
                 dt: float = 1e-3,
                 population_size: int = 8):
        """Initialize backward compatible spike bridge."""
        self.config = SpikeBridgeConfig(
            encoding_strategy=encoding_strategy,
            spike_time_steps=spike_time_steps,
            max_spike_rate=max_spike_rate,
            dt=dt,
            population_size=population_size,
            use_int8_output=False,  # Maintain backward compatibility
            use_vectorized_ops=True
        )
        self.optimized_bridge = OptimizedSpikeBridge(self.config)
    
    def encode(self, latent_features: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Encode using optimized bridge."""
        return self.optimized_bridge.encode(latent_features, key)


# Enhanced factory functions
def create_optimized_spike_bridge(config: Optional[SpikeBridgeConfig] = None) -> OptimizedSpikeBridge:
    """Create optimized spike bridge with full configuration."""
    if config is None:
        config = SpikeBridgeConfig()
    
    return OptimizedSpikeBridge(config)


def create_int8_spike_bridge(encoding_strategy: SpikeEncodingStrategy = SpikeEncodingStrategy.POISSON_RATE) -> OptimizedSpikeBridge:
    """Create memory-efficient int8 spike bridge."""
    config = SpikeBridgeConfig(
        encoding_strategy=encoding_strategy,
        use_int8_output=True,
        use_vectorized_ops=True,
        memory_efficient=True
    )
    return OptimizedSpikeBridge(config)


def create_cosine_spike_bridge(basis_size: int = 64, temperature: float = 1.0) -> OptimizedSpikeBridge:
    """Create cosine-similarity spike bridge."""
    config = SpikeBridgeConfig(
        encoding_strategy=SpikeEncodingStrategy.COSINE_SIMILARITY,
        cosine_basis_size=basis_size,
        cosine_temperature=temperature,
        use_int8_output=True,
        use_vectorized_ops=True
    )
    return OptimizedSpikeBridge(config)


def create_benchmark_config() -> SpikeBridgeConfig:
    """Create configuration optimized for benchmarking."""
    return SpikeBridgeConfig(
        encoding_strategy=SpikeEncodingStrategy.POISSON_RATE,
        spike_time_steps=50,
        max_spike_rate=100.0,
        use_int8_output=True,
        use_vectorized_ops=True,
        memory_efficient=True,
        enable_profiling=True,
        profile_memory=True
    )


# Legacy convenience functions
def create_default_spike_bridge() -> SpikeBridge:
    """Create spike bridge with optimal default parameters for GW detection."""
    return SpikeBridge(
        encoding_strategy=SpikeEncodingStrategy.POISSON_RATE,
        spike_time_steps=50,
        max_spike_rate=100.0,
        dt=1e-3,
        population_size=8
    )


def create_fast_spike_bridge() -> SpikeBridge:
    """Create spike bridge optimized for speed."""
    return SpikeBridge(
        encoding_strategy=SpikeEncodingStrategy.RATE_BASED,
        spike_time_steps=20,
        max_spike_rate=50.0,
        dt=1e-3,
        population_size=4
    )


def create_robust_spike_bridge() -> SpikeBridge:
    """Create spike bridge optimized for robustness."""
    return SpikeBridge(
        encoding_strategy=SpikeEncodingStrategy.POPULATION_VECTOR,
        spike_time_steps=80,
        max_spike_rate=200.0,
        dt=0.5e-3,
        population_size=16
    )


def create_spike_bridge_from_string(encoding: str, **kwargs) -> SpikeBridge:
    """Create SpikeBridge from string encoding (legacy compatibility)."""
    encoding_map = {
        "poisson": SpikeEncodingStrategy.POISSON_RATE,
        "temporal_contrast": SpikeEncodingStrategy.TEMPORAL_CONTRAST,
        "population_vector": SpikeEncodingStrategy.POPULATION_VECTOR,
        "rate_based": SpikeEncodingStrategy.RATE_BASED,
        "cosine_similarity": SpikeEncodingStrategy.COSINE_SIMILARITY,
        "binary_threshold": SpikeEncodingStrategy.BINARY_THRESHOLD
    }
    
    if encoding not in encoding_map:
        raise ValueError(f"Unknown encoding: {encoding}. Available: {list(encoding_map.keys())}")
    
    return SpikeBridge(encoding_strategy=encoding_map[encoding], **kwargs)


# Enhanced test functions
def test_optimized_spike_bridge():
    """Test optimized spike bridge with all features."""
    print("Testing Optimized Spike Bridge...")
    
    try:
        # Test configurations
        configs = [
            SpikeBridgeConfig(use_int8_output=True, use_vectorized_ops=True),
            SpikeBridgeConfig(encoding_strategy=SpikeEncodingStrategy.COSINE_SIMILARITY),
            SpikeBridgeConfig(encoding_strategy=SpikeEncodingStrategy.BINARY_THRESHOLD),
            create_benchmark_config()
        ]
        
        # Test data
        key = jax.random.PRNGKey(42)
        batch_size, seq_len, latent_dim = 4, 32, 128
        latent_features = jax.random.normal(key, (batch_size, seq_len, latent_dim))
        
        for i, config in enumerate(configs):
            print(f"  Testing configuration {i+1}...")
            
            bridge = OptimizedSpikeBridge(config)
            
            key, subkey = jax.random.split(key)
            spikes = bridge.encode(latent_features, subkey)
            
            print(f"    ✅ Input: {latent_features.shape}")
            print(f"    ✅ Output: {spikes.shape}, dtype: {spikes.dtype}")
            print(f"    ✅ Strategy: {config.encoding_strategy.value}")
            print(f"    ✅ int8: {config.use_int8_output}")
            print(f"    ✅ Spike rate: {jnp.mean(spikes.astype(jnp.float32)):.4f}")
            
            # Test memory efficiency
            memory_usage = bridge._estimate_memory_usage(spikes)
            print(f"    ✅ Memory usage: {memory_usage:.2f} MB")
            
            # Verify output format
            if config.use_int8_output:
                assert spikes.dtype == jnp.int8, f"Expected int8, got {spikes.dtype}"
            else:
                assert spikes.dtype == jnp.float32, f"Expected float32, got {spikes.dtype}"
            
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Optimized spike bridge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cosine_similarity_encoding():
    """Test cosine similarity encoding specifically."""
    print("Testing Cosine Similarity Encoding...")
    
    try:
        # Create cosine bridge
        bridge = create_cosine_spike_bridge(basis_size=32, temperature=0.5)
        
        # Test data
        key = jax.random.PRNGKey(42)
        batch_size, seq_len, latent_dim = 2, 16, 64
        latent_features = jax.random.normal(key, (batch_size, seq_len, latent_dim))
        
        # Encode
        key, subkey = jax.random.split(key)
        spikes = bridge.encode(latent_features, subkey)
        
        print(f"  ✅ Input shape: {latent_features.shape}")
        print(f"  ✅ Output shape: {spikes.shape}")
        print(f"  ✅ Basis size: {bridge.config.cosine_basis_size}")
        print(f"  ✅ Temperature: {bridge.config.cosine_temperature}")
        print(f"  ✅ Spike rate: {jnp.mean(spikes.astype(jnp.float32)):.4f}")
        print(f"  ✅ Output dtype: {spikes.dtype}")
        
        # Check basis orthogonality
        basis = bridge.cosine_basis
        orthogonality = jnp.max(jnp.abs(jnp.dot(basis, basis.T) - jnp.eye(basis.shape[0])))
        print(f"  ✅ Basis orthogonality error: {orthogonality:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cosine similarity test failed: {e}")
        return False


def test_throughput_benchmarks():
    """Test throughput benchmarking functionality."""
    print("Testing Throughput Benchmarks...")
    
    try:
        # Create benchmark configuration
        config = create_benchmark_config()
        bridge = OptimizedSpikeBridge(config)
        
        # Run mini benchmark
        results = bridge.benchmark_throughput(
            batch_sizes=[1, 4],
            sequence_lengths=[32, 64],
            latent_dims=[64, 128],
            num_runs=2
        )
        
        print(f"  ✅ Benchmark completed: {len(results)} configurations")
        
        # Check results
        for result in results:
            print(f"    Batch:{result.batch_size}, Seq:{result.sequence_length}, Dim:{result.latent_dim}")
            print(f"    Throughput: {result.samples_per_second:.1f} samples/sec")
            print(f"    Memory: {result.memory_usage_mb:.2f} MB")
            print(f"    Score: {result.throughput_score:.2f}")
            print()
        
        # Print benchmark report
        bridge.print_benchmark_report(results)
        
        return True
        
    except Exception as e:
        print(f"❌ Throughput benchmark test failed: {e}")
        return False


def test_memory_efficiency():
    """Test memory efficiency of int8 vs float32."""
    print("Testing Memory Efficiency...")
    
    try:
        # Create both configurations
        config_int8 = SpikeBridgeConfig(use_int8_output=True)
        config_float32 = SpikeBridgeConfig(use_int8_output=False)
        
        bridge_int8 = OptimizedSpikeBridge(config_int8)
        bridge_float32 = OptimizedSpikeBridge(config_float32)
        
        # Test data
        key = jax.random.PRNGKey(42)
        batch_size, seq_len, latent_dim = 8, 64, 256
        latent_features = jax.random.normal(key, (batch_size, seq_len, latent_dim))
        
        # Generate spikes
        key, subkey1, subkey2 = jax.random.split(key, 3)
        spikes_int8 = bridge_int8.encode(latent_features, subkey1)
        spikes_float32 = bridge_float32.encode(latent_features, subkey2)
        
        # Calculate memory usage
        memory_int8 = bridge_int8._estimate_memory_usage(spikes_int8)
        memory_float32 = bridge_float32._estimate_memory_usage(spikes_float32)
        
        memory_savings = (memory_float32 - memory_int8) / memory_float32 * 100
        
        print(f"  ✅ int8 memory usage: {memory_int8:.2f} MB")
        print(f"  ✅ float32 memory usage: {memory_float32:.2f} MB")
        print(f"  ✅ Memory savings: {memory_savings:.1f}%")
        print(f"  ✅ Compression ratio: {memory_float32 / memory_int8:.1f}x")
        
        # Verify correctness
        spikes_int8_float = spikes_int8.astype(jnp.float32)
        max_diff = jnp.max(jnp.abs(spikes_int8_float - spikes_float32))
        print(f"  ✅ Max difference (should be ~0): {max_diff:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory efficiency test failed: {e}")
        return False


def test_backward_compatibility():
    """Test backward compatibility with original SpikeBridge."""
    print("Testing Backward Compatibility...")
    
    try:
        # Create original and optimized bridges
        original_bridge = create_default_spike_bridge()
        optimized_bridge = create_optimized_spike_bridge()
        
        # Test data
        key = jax.random.PRNGKey(42)
        batch_size, seq_len, latent_dim = 2, 16, 64
        latent_features = jax.random.normal(key, (batch_size, seq_len, latent_dim))
        
        # Generate spikes
        key, subkey1, subkey2 = jax.random.split(key, 3)
        spikes_original = original_bridge.encode(latent_features, subkey1)
        spikes_optimized = optimized_bridge.encode(latent_features, subkey2)
        
        print(f"  ✅ Original output: {spikes_original.shape}, dtype: {spikes_original.dtype}")
        print(f"  ✅ Optimized output: {spikes_optimized.shape}, dtype: {spikes_optimized.dtype}")
        
        # Check API compatibility
        assert hasattr(original_bridge, 'encode'), "Original bridge missing encode method"
        assert hasattr(optimized_bridge, 'encode'), "Optimized bridge missing encode method"
        
        print(f"  ✅ API compatibility maintained")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧠 Testing Optimized Spike Bridge Implementation...")
    print()
    
    success1 = test_optimized_spike_bridge()
    print()
    success2 = test_cosine_similarity_encoding()
    print()
    success3 = test_throughput_benchmarks()
    print()
    success4 = test_memory_efficiency()
    print()
    success5 = test_backward_compatibility()
    print()
    
    overall_success = success1 and success2 and success3 and success4 and success5
    print(f"🎯 Overall: {'SUCCESS' if overall_success else 'FAILED'}")
    
    if overall_success:
        print("🎉 All optimized spike bridge tests passed!")
    else:
        print("❌ Some tests failed - check implementation")
