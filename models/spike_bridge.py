"""
Optimized Spike Bridge: Continuous to Spike Conversion

Converts continuous latent representations to spike trains for SNN processing.
Supports multiple encoding strategies with performance optimizations.

Key features:
- Multiple encoding strategies (Poisson, temporal contrast, population vector, etc.)
- Memory optimization with int8 output support
- Vectorized operations for GPU/TPU efficiency
- Comprehensive benchmarking capabilities
"""

import jax
import jax.numpy as jnp
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
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
    samples_per_second: float
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
    Optimized spike bridge with multiple encoding strategies and performance optimizations.
    
    Features:
    - Multiple encoding strategies with JIT compilation
    - int8 output support for memory efficiency
    - Vectorized operations for GPU/TPU performance
    - Comprehensive benchmarking capabilities
    """
    
    def __init__(self, config: SpikeBridgeConfig):
        self.config = config
        
        # Pre-validate configuration
        self._validate_config()
        
        # Pre-create cosine basis if needed
        if config.encoding_strategy == SpikeEncodingStrategy.COSINE_SIMILARITY:
            self.cosine_basis = self._create_cosine_basis()
        else:
            self.cosine_basis = None
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.spike_time_steps <= 0:
            raise ValueError("spike_time_steps must be positive")
        if self.config.max_spike_rate <= 0:
            raise ValueError("max_spike_rate must be positive")
        if self.config.population_size <= 0:
            raise ValueError("population_size must be positive")
    
    def _create_cosine_basis(self) -> jnp.ndarray:
        """Create cosine basis for cosine similarity encoding."""
        # Create random basis vectors
        key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
        basis = jax.random.normal(key, (self.config.cosine_basis_size, self.config.cosine_basis_size))
        
        # Orthonormalize using QR decomposition
        q, _ = jnp.linalg.qr(basis)
        return q
    
    def encode(self, 
               latent_features: jnp.ndarray, 
               key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Encode latent features to spikes using specified strategy.
        
        Args:
            latent_features: Continuous features [batch, time, latent_dim]
            key: Random key for stochastic encoding
            
        Returns:
            Spike trains [batch, spike_time_steps, output_dim]
        """
        # Input validation
        self._validate_input(latent_features)
        
        # Choose encoding strategy
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
            raise ValueError(f"Unknown encoding strategy: {self.config.encoding_strategy}")
        
        # Convert to int8 if requested
        if self.config.use_int8_output:
            spikes = self._convert_to_int8(spikes)
        
        return spikes
    
    def _validate_input(self, latent_features: jnp.ndarray):
        """Validate input tensor."""
        if latent_features.ndim != 3:
            raise ValueError(f"Expected 3D input [batch, time, features], got {latent_features.ndim}D")
        
        batch_size, seq_len, _ = latent_features.shape
        if batch_size > self.config.batch_size_limit:
            logger.warning(f"Large batch size {batch_size} may cause memory issues")
        if seq_len > self.config.time_steps_limit:
            logger.warning(f"Long sequence {seq_len} may cause memory issues")
    
    @jax.jit
    def _poisson_rate_encoding(self, 
                              latent_features: jnp.ndarray, 
                              key: jax.random.PRNGKey) -> jnp.ndarray:
        """Poisson rate encoding with temporal interpolation."""
        batch_size, seq_len, latent_dim = latent_features.shape
        
        # Normalize features to [0, 1]
        features_norm = jax.nn.sigmoid(latent_features)
        
        # Temporal interpolation to spike time steps
        if seq_len != self.config.spike_time_steps:
            # Interpolate temporally
            time_indices = jnp.linspace(0, seq_len - 1, self.config.spike_time_steps)
            time_indices_int = jnp.floor(time_indices).astype(jnp.int32)
            time_weights = time_indices - time_indices_int
            
            # Linear interpolation
            features_interp = (
                features_norm[:, time_indices_int] * (1 - time_weights[None, :, None]) +
                features_norm[:, jnp.minimum(time_indices_int + 1, seq_len - 1)] * time_weights[None, :, None]
            )
        else:
            features_interp = features_norm
        
        # Convert to spike rates
        spike_rates = features_interp * self.config.max_spike_rate * self.config.dt
        
        # Poisson sampling
        spikes = jax.random.poisson(key, spike_rates) > 0
        
        return spikes.astype(jnp.float32)
    
    @jax.jit
    def _temporal_contrast_encoding(self, 
                                  latent_features: jnp.ndarray, 
                                  key: jax.random.PRNGKey) -> jnp.ndarray:
        """Temporal contrast encoding based on differences."""
        batch_size, seq_len, latent_dim = latent_features.shape
        
        # Compute temporal differences
        diff = jnp.diff(latent_features, axis=1, prepend=latent_features[:, :1])
        
        # Threshold for spike generation
        threshold = jnp.std(diff) * 0.5
        
        # ON and OFF spikes
        on_spikes = (diff > threshold).astype(jnp.float32)
        off_spikes = (diff < -threshold).astype(jnp.float32)
        
        # Combine ON and OFF channels
        contrast_spikes = jnp.concatenate([on_spikes, off_spikes], axis=-1)
        
        # Temporal interpolation if needed
        if seq_len != self.config.spike_time_steps:
            time_indices = jnp.linspace(0, seq_len - 1, self.config.spike_time_steps)
            time_indices_int = jnp.floor(time_indices).astype(jnp.int32)
            contrast_spikes = contrast_spikes[:, time_indices_int]
        
        return contrast_spikes
    
    @jax.jit
    def _population_vector_encoding(self, 
                                  latent_features: jnp.ndarray, 
                                  key: jax.random.PRNGKey) -> jnp.ndarray:
        """Population vector encoding with multiple neurons per feature."""
        batch_size, seq_len, latent_dim = latent_features.shape
        
        # Create preferred directions for population
        directions = jnp.linspace(-1, 1, self.config.population_size)
        
        # Expand features and directions for broadcasting
        features_expanded = latent_features[..., None]  # [batch, time, latent_dim, 1]
        directions_expanded = directions[None, None, None, :]  # [1, 1, 1, population_size]
        
        # Compute population responses (Gaussian tuning curves)
        responses = jnp.exp(-0.5 * ((features_expanded - directions_expanded) / 0.3)**2)
        
        # Reshape to [batch, time, latent_dim * population_size]
        population_responses = responses.reshape(batch_size, seq_len, -1)
        
        # Convert to spike rates and apply Poisson sampling
        spike_rates = population_responses * self.config.max_spike_rate * self.config.dt
        spikes = jax.random.poisson(key, spike_rates) > 0
        
        # Temporal interpolation if needed
        if seq_len != self.config.spike_time_steps:
            time_indices = jnp.linspace(0, seq_len - 1, self.config.spike_time_steps)
            time_indices_int = jnp.floor(time_indices).astype(jnp.int32)
            spikes = spikes[:, time_indices_int]
        
        return spikes.astype(jnp.float32)
    
    @jax.jit
    def _rate_based_encoding(self,
                            latent_features: jnp.ndarray,
                            key: jax.random.PRNGKey) -> jnp.ndarray:
        """Simple rate-based encoding without randomness."""
        batch_size, seq_len, latent_dim = latent_features.shape
        
        # Normalize to [0, 1]
        features_norm = (latent_features - jnp.min(latent_features)) / (jnp.max(latent_features) - jnp.min(latent_features) + 1e-8)
        
        # Convert to binary spikes based on threshold
        threshold = 0.5
        spikes = (features_norm > threshold).astype(jnp.float32)
        
        # Temporal interpolation if needed
        if seq_len != self.config.spike_time_steps:
            time_indices = jnp.linspace(0, seq_len - 1, self.config.spike_time_steps)
            time_indices_int = jnp.floor(time_indices).astype(jnp.int32)
            spikes = spikes[:, time_indices_int]
        
        return spikes
    
    @jax.jit
    def _cosine_similarity_encoding(self,
                                   latent_features: jnp.ndarray,
                                   key: jax.random.PRNGKey) -> jnp.ndarray:
        """Cosine similarity-based encoding using learned basis."""
        batch_size, seq_len, latent_dim = latent_features.shape
        
        # Project features onto cosine basis
        similarities = jnp.dot(latent_features, self.cosine_basis.T)  # [batch, time, basis_size]
        
        # Apply temperature scaling
        similarities_scaled = similarities / self.config.cosine_temperature
        
        # Convert to probabilities and sample
        probabilities = jax.nn.sigmoid(similarities_scaled)
        spikes = jax.random.bernoulli(key, probabilities).astype(jnp.float32)
        
        # Temporal interpolation if needed
        if seq_len != self.config.spike_time_steps:
            time_indices = jnp.linspace(0, seq_len - 1, self.config.spike_time_steps)
            time_indices_int = jnp.floor(time_indices).astype(jnp.int32)
            spikes = spikes[:, time_indices_int]
        
        return spikes
    
    @jax.jit
    def _binary_threshold_encoding(self,
                           latent_features: jnp.ndarray, 
                           key: jax.random.PRNGKey) -> jnp.ndarray:
        """Simple binary threshold encoding."""
        batch_size, seq_len, latent_dim = latent_features.shape
        
        # Compute adaptive threshold
        threshold = jnp.mean(latent_features, axis=(0, 1), keepdims=True)
        
        # Generate spikes
        spikes = (latent_features > threshold).astype(jnp.float32)
        
        # Temporal interpolation if needed
        if seq_len != self.config.spike_time_steps:
            time_indices = jnp.linspace(0, seq_len - 1, self.config.spike_time_steps)
            time_indices_int = jnp.floor(time_indices).astype(jnp.int32)
            spikes = spikes[:, time_indices_int]
        
        return spikes
    
    def _convert_to_int8(self, spikes: jnp.ndarray) -> jnp.ndarray:
        """Convert spikes to int8 format for memory efficiency."""
        # Clamp to [0, 1] and convert to int8
        spikes_clamped = jnp.clip(spikes, 0, 1)
        spikes_int8 = (spikes_clamped * 127).astype(jnp.int8)
        return spikes_int8
    
    def _estimate_memory_usage(self, spikes: jnp.ndarray) -> float:
        """Estimate memory usage in MB."""
        num_elements = spikes.size
        bytes_per_element = spikes.dtype.itemsize
        total_bytes = num_elements * bytes_per_element
        return total_bytes / (1024 * 1024)  # Convert to MB


# Backward compatibility wrapper
class SpikeBridge:
    """Backward compatible spike bridge interface."""
    
    def __init__(self,
                 encoding_strategy: SpikeEncodingStrategy = SpikeEncodingStrategy.POISSON_RATE,
                 spike_time_steps: int = 100,
                 max_spike_rate: float = 100.0,
                 dt: float = 1e-3,
                 population_size: int = 8):
        
        # Create config and optimized bridge
        self.config = SpikeBridgeConfig(
            encoding_strategy=encoding_strategy,
            spike_time_steps=spike_time_steps,
            max_spike_rate=max_spike_rate,
            dt=dt,
            population_size=population_size,
            use_int8_output=False,  # Disable for compatibility
            memory_efficient=True
        )
        self.optimized_bridge = OptimizedSpikeBridge(self.config)
    
    def encode(self, latent_features: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Encode latent features to spikes."""
        return self.optimized_bridge.encode(latent_features, key)


# Factory functions
def create_optimized_spike_bridge(config: Optional[SpikeBridgeConfig] = None) -> OptimizedSpikeBridge:
    """Create optimized spike bridge with configuration."""
    if config is None:
        config = SpikeBridgeConfig()
    return OptimizedSpikeBridge(config)


def create_int8_spike_bridge(encoding_strategy: SpikeEncodingStrategy = SpikeEncodingStrategy.POISSON_RATE) -> OptimizedSpikeBridge:
    """Create int8-optimized spike bridge for memory efficiency."""
    config = SpikeBridgeConfig(
        encoding_strategy=encoding_strategy,
        use_int8_output=True,
        memory_efficient=True
    )
    return OptimizedSpikeBridge(config)


def create_cosine_spike_bridge(basis_size: int = 64, temperature: float = 1.0) -> OptimizedSpikeBridge:
    """Create cosine similarity spike bridge."""
    config = SpikeBridgeConfig(
        encoding_strategy=SpikeEncodingStrategy.COSINE_SIMILARITY,
        cosine_basis_size=basis_size,
        cosine_temperature=temperature
    )
    return OptimizedSpikeBridge(config)


def create_benchmark_config() -> SpikeBridgeConfig:
    """Create configuration optimized for benchmarking."""
    return SpikeBridgeConfig(
        encoding_strategy=SpikeEncodingStrategy.POISSON_RATE,
        spike_time_steps=100,
        use_int8_output=True,
        use_vectorized_ops=True,
        memory_efficient=True,
        enable_profiling=True
    )


def create_default_spike_bridge() -> SpikeBridge:
    """Create spike bridge with optimal default parameters for GW detection."""
    return SpikeBridge(
        encoding_strategy=SpikeEncodingStrategy.POISSON_RATE,
        spike_time_steps=50,
        max_spike_rate=100.0,
        dt=1e-3,
        population_size=8
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
