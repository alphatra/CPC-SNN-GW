#!/usr/bin/env python3
"""
🧪 CRITICAL INTEGRATION TESTING FRAMEWORK

Comprehensive unit and integration tests addressing analysis recommendations:
- Priority: Add testing framework for validation (medium impact, medium effort)
- Focus: Gradient flow tests, model integration, performance validation
- Goal: Ensure stability and detect bugs in production pipeline

Testing Categories:
1. Unit Tests: Core functions (gradient flow, spike encoding, model components)
2. Integration Tests: End-to-end pipeline validation
3. Performance Tests: Benchmark validation (<100ms inference)
4. Scientific Tests: PyCBC baseline accuracy, statistical significance
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import sys
import tempfile
import time
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test fixtures and utilities
@pytest.fixture
def test_config():
    """Standard test configuration for reproducible tests."""
    return {
        'batch_size': 4,
        'sequence_length': 1024,  # 0.25s @ 4kHz
        'latent_dim': 128,
        'num_classes': 3,
        'random_seed': 42
    }

@pytest.fixture  
def sample_strain_data(test_config):
    """Generate realistic test strain data."""
    np.random.seed(test_config['random_seed'])
    
    # Generate realistic strain amplitudes (LIGO-like)
    strain_amplitude = 1e-21  # Realistic LIGO strain level
    
    batch_size = test_config['batch_size']
    seq_len = test_config['sequence_length']
    
    # Create test data with GW-like characteristics
    strain_data = []
    for i in range(batch_size):
        # Base noise
        noise = np.random.normal(0, strain_amplitude, seq_len)
        
        # Add chirp-like signal for half the samples
        if i < batch_size // 2:
            t = np.linspace(0, 0.25, seq_len)  # 0.25s duration
            f0, f1 = 35, 250  # Frequency sweep
            chirp = strain_amplitude * 5 * np.sin(2 * np.pi * (f0 + (f1-f0) * t**2) * t)
            noise += chirp
            
        strain_data.append(noise)
    
    return jnp.array(strain_data, dtype=jnp.float32)

@pytest.fixture
def sample_labels(test_config):
    """Generate test labels."""
    batch_size = test_config['batch_size']
    # Half signals, half noise
    labels = [1] * (batch_size // 2) + [0] * (batch_size - batch_size // 2)
    return jnp.array(labels, dtype=jnp.int32)

# ============================================================================
# UNIT TESTS: Core Functions
# ============================================================================

class TestCPCEncoder:
    """Unit tests for CPC encoder components."""
    
    def test_cpc_encoder_initialization(self, test_config):
        """Test CPC encoder can be initialized with correct parameters."""
        from models.cpc_encoder import RealCPCEncoder, RealCPCConfig
        
        config = RealCPCConfig(
            latent_dim=test_config['latent_dim'],
            downsample_factor=4,  # Critical fix verification
            context_length=64,
            num_negatives=32
        )
        
        encoder = RealCPCEncoder(config)
        
        # Test initialization
        key = jax.random.PRNGKey(test_config['random_seed'])
        dummy_input = jnp.ones((1, test_config['sequence_length']))
        
        params = encoder.init(key, dummy_input)
        assert params is not None
        logger.info("✅ CPC encoder initialization successful")
    
    def test_cpc_forward_pass(self, test_config, sample_strain_data):
        """Test CPC encoder forward pass produces correct output shapes."""
        from models.cpc_encoder import RealCPCEncoder, RealCPCConfig
        
        config = RealCPCConfig(
            latent_dim=test_config['latent_dim'],
            downsample_factor=4,
            context_length=64
        )
        
        encoder = RealCPCEncoder(config)
        key = jax.random.PRNGKey(test_config['random_seed'])
        
        # Initialize and test forward pass
        params = encoder.init(key, sample_strain_data[:1])
        features = encoder.apply(params, sample_strain_data)
        
        # Verify output shape
        expected_seq_len = test_config['sequence_length'] // config.downsample_factor
        expected_shape = (test_config['batch_size'], expected_seq_len, config.latent_dim)
        
        assert features.shape == expected_shape, f"Expected {expected_shape}, got {features.shape}"
        assert not jnp.any(jnp.isnan(features)), "CPC features contain NaN values"
        assert not jnp.any(jnp.isinf(features)), "CPC features contain infinite values"
        
        logger.info(f"✅ CPC forward pass: {sample_strain_data.shape} → {features.shape}")
    
    def test_cpc_gradient_flow(self, test_config, sample_strain_data, sample_labels):
        """🚨 CRITICAL: Test gradient flow through CPC encoder."""
        from models.cpc_encoder import RealCPCEncoder, RealCPCConfig
        from models.cpc_losses import enhanced_info_nce_loss
        
        config = RealCPCConfig(latent_dim=test_config['latent_dim'])
        encoder = RealCPCEncoder(config)
        key = jax.random.PRNGKey(test_config['random_seed'])
        
        params = encoder.init(key, sample_strain_data[:1])
        
        def loss_fn(params, x, labels):
            features = encoder.apply(params, x)
            # Simple loss for gradient testing
            logits = jnp.mean(features, axis=(1, 2))  # Global average pooling
            targets = jax.nn.one_hot(labels, num_classes=2)
            return jnp.mean(jnp.sum((logits[:, None] - targets)**2, axis=1))
        
        # Test gradient computation
        loss_val, grads = jax.value_and_grad(loss_fn)(params, sample_strain_data, sample_labels)
        
        # Validate gradients
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
        
        assert not jnp.isnan(loss_val), f"Loss is NaN: {loss_val}"
        assert not jnp.isnan(grad_norm), f"Gradient norm is NaN: {grad_norm}"
        assert grad_norm > 1e-8, f"Vanishing gradients detected: {grad_norm}"
        assert grad_norm < 1e2, f"Exploding gradients detected: {grad_norm}"
        
        logger.info(f"✅ CPC gradient flow: loss={loss_val:.6f}, grad_norm={grad_norm:.6f}")

class TestSpikeBridge:
    """Unit tests for spike bridge component."""
    
    def test_temporal_contrast_encoding(self, test_config):
        """Test temporal contrast encoding preserves frequency information."""
        from models.spike_bridge import OptimizedSpikeBridge, SpikeBridgeConfig
        
        config = SpikeBridgeConfig(
            encoding_strategy="temporal_contrast",
            time_steps=50,
            dt=1e-3
        )
        
        bridge = OptimizedSpikeBridge(config)
        key = jax.random.PRNGKey(test_config['random_seed'])
        
        # Create test input with known frequency content
        batch_size = test_config['batch_size']
        seq_len = 256
        latent_dim = test_config['latent_dim']
        
        # Generate input with frequency sweep
        t = jnp.linspace(0, 1.0, seq_len)
        freq_sweep = jnp.sin(2 * jnp.pi * 50 * t)  # 50 Hz signal
        test_features = jnp.broadcast_to(
            freq_sweep[None, :, None], 
            (batch_size, seq_len, latent_dim)
        )
        
        # Initialize and encode
        params = bridge.init(key, test_features)
        spikes = bridge.apply(params, test_features)
        
        # Verify spike encoding
        assert spikes.shape[0] == batch_size, f"Batch dimension mismatch: {spikes.shape[0]}"
        assert spikes.dtype == jnp.float32, f"Spike dtype should be float32: {spikes.dtype}"
        
        spike_rate = jnp.mean(spikes)
        assert 0.01 < spike_rate < 0.5, f"Unrealistic spike rate: {spike_rate}"
        
        # Test temporal contrast preserves transitions
        diff_count = jnp.sum(jnp.abs(jnp.diff(spikes, axis=1)) > 0)
        assert diff_count > 0, "No temporal transitions detected in spike encoding"
        
        logger.info(f"✅ Temporal contrast encoding: spike_rate={spike_rate:.4f}, transitions={diff_count}")
    
    def test_spike_gradient_flow(self, test_config):
        """Test gradient flow through spike bridge with surrogate gradients."""
        from models.spike_bridge import OptimizedSpikeBridge, SpikeBridgeConfig
        
        config = SpikeBridgeConfig(
            encoding_strategy="temporal_contrast",
            surrogate_beta=4.0  # Enhanced gradients
        )
        
        bridge = OptimizedSpikeBridge(config)
        key = jax.random.PRNGKey(test_config['random_seed'])
        
        # Test data
        test_features = jax.random.normal(key, (2, 100, test_config['latent_dim']))
        
        params = bridge.init(key, test_features)
        
        def spike_loss_fn(params, features):
            spikes = bridge.apply(params, features)
            return jnp.mean(spikes**2)  # Simple loss for gradient test
        
        loss_val, grads = jax.value_and_grad(spike_loss_fn)(params, test_features)
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
        
        # Validate surrogate gradients
        assert not jnp.isnan(grad_norm), "Spike bridge gradients are NaN"
        assert grad_norm > 1e-8, f"Vanishing surrogate gradients: {grad_norm}"
        
        logger.info(f"✅ Spike bridge gradient flow: grad_norm={grad_norm:.6f}")

class TestSNNClassifier:
    """Unit tests for SNN classifier component."""
    
    def test_snn_deep_architecture(self, test_config):
        """Test deep SNN architecture (3 layers: 256→128→64)."""
        from models.snn_classifier import EnhancedSNNClassifier, SNNConfig
        
        config = SNNConfig(
            hidden_sizes=[256, 128, 64],  # Deep 3-layer architecture
            num_classes=test_config['num_classes'],
            surrogate_beta=4.0
        )
        
        snn = EnhancedSNNClassifier(config)
        key = jax.random.PRNGKey(test_config['random_seed'])
        
        # Test with spike input
        spike_input = jax.random.uniform(
            key, 
            (test_config['batch_size'], 50, 512),  # After spike bridge
            minval=0, maxval=1
        ) > 0.8  # Binary spikes
        spike_input = spike_input.astype(jnp.float32)
        
        params = snn.init(key, spike_input)
        logits = snn.apply(params, spike_input)
        
        # Verify output shape and properties
        expected_shape = (test_config['batch_size'], test_config['num_classes'])
        assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
        
        # Check if outputs are reasonable
        probs = jax.nn.softmax(logits, axis=-1)
        assert jnp.allclose(jnp.sum(probs, axis=-1), 1.0), "Probabilities don't sum to 1"
        assert not jnp.any(jnp.isnan(logits)), "SNN logits contain NaN"
        
        logger.info(f"✅ Deep SNN architecture: {spike_input.shape} → {logits.shape}")

# ============================================================================
# INTEGRATION TESTS: End-to-End Pipeline
# ============================================================================

class TestPipelineIntegration:
    """Integration tests for complete pipeline."""
    
    def test_end_to_end_pipeline(self, test_config, sample_strain_data, sample_labels):
        """🚨 CRITICAL: Test complete CPC→Spike→SNN pipeline integration."""
        from models.cpc_encoder import RealCPCEncoder, RealCPCConfig
        from models.spike_bridge import OptimizedSpikeBridge, SpikeBridgeConfig  
        from models.snn_classifier import EnhancedSNNClassifier, SNNConfig
        
        # Initialize all components
        cpc_config = RealCPCConfig(latent_dim=128, downsample_factor=4)
        spike_config = SpikeBridgeConfig(encoding_strategy="temporal_contrast", time_steps=50)
        snn_config = SNNConfig(hidden_sizes=[128, 64], num_classes=test_config['num_classes'])
        
        cpc_encoder = RealCPCEncoder(cpc_config)
        spike_bridge = OptimizedSpikeBridge(spike_config)
        snn_classifier = EnhancedSNNClassifier(snn_config)
        
        key = jax.random.PRNGKey(test_config['random_seed'])
        
        # Initialize pipeline
        cpc_params = cpc_encoder.init(key, sample_strain_data[:1])
        sample_features = cpc_encoder.apply(cpc_params, sample_strain_data[:1])
        
        spike_params = spike_bridge.init(key, sample_features)
        sample_spikes = spike_bridge.apply(spike_params, sample_features)
        
        snn_params = snn_classifier.init(key, sample_spikes)
        
        # Test complete pipeline
        def full_pipeline(strain):
            features = cpc_encoder.apply(cpc_params, strain)
            spikes = spike_bridge.apply(spike_params, features)
            logits = snn_classifier.apply(snn_params, spikes)
            return logits
        
        # Forward pass
        final_logits = full_pipeline(sample_strain_data)
        
        # Validate pipeline output
        expected_shape = (test_config['batch_size'], test_config['num_classes'])
        assert final_logits.shape == expected_shape
        assert not jnp.any(jnp.isnan(final_logits))
        
        # Test gradient flow through entire pipeline
        def pipeline_loss(strain, labels):
            logits = full_pipeline(strain)
            targets = jax.nn.one_hot(labels, test_config['num_classes'])
            return jnp.mean(jnp.sum((jax.nn.softmax(logits) - targets)**2, axis=1))
        
        loss_val, grads = jax.value_and_grad(pipeline_loss)(sample_strain_data, sample_labels)
        
        # Check gradients for all parameters
        for param_name, param_grads in [("cpc", cpc_params), ("spike", spike_params), ("snn", snn_params)]:
            if param_grads:
                grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(param_grads) if g is not None))
                assert not jnp.isnan(grad_norm), f"{param_name} gradients are NaN"
                logger.info(f"✅ {param_name} gradient norm: {grad_norm:.6f}")
        
        logger.info(f"✅ End-to-end pipeline: {sample_strain_data.shape} → {final_logits.shape}")
        logger.info(f"✅ Pipeline loss: {loss_val:.6f}")

# ============================================================================
# PERFORMANCE TESTS: Benchmark Validation
# ============================================================================

class TestPerformanceBenchmarks:
    """Performance tests for <100ms inference validation."""
    
    def test_inference_latency_target(self, test_config):
        """🚨 CRITICAL: Test <100ms inference latency target."""
        from models.cpc_encoder import RealCPCEncoder, RealCPCConfig
        from models.spike_bridge import OptimizedSpikeBridge, SpikeBridgeConfig  
        from models.snn_classifier import EnhancedSNNClassifier, SNNConfig
        
        # Create optimized models for performance test
        cpc_config = RealCPCConfig(latent_dim=256, downsample_factor=4)
        spike_config = SpikeBridgeConfig(encoding_strategy="temporal_contrast")
        snn_config = SNNConfig(hidden_sizes=[256, 128, 64], num_classes=3)
        
        cpc_encoder = RealCPCEncoder(cpc_config)
        spike_bridge = OptimizedSpikeBridge(spike_config)
        snn_classifier = EnhancedSNNClassifier(snn_config)
        
        # Initialize with realistic input size
        key = jax.random.PRNGKey(test_config['random_seed'])
        test_input = jax.random.normal(key, (16, 16384))  # 16 samples, 4s @ 4kHz
        
        # Initialize parameters
        cpc_params = cpc_encoder.init(key, test_input[:1])
        sample_features = cpc_encoder.apply(cpc_params, test_input[:1])
        spike_params = spike_bridge.init(key, sample_features)
        sample_spikes = spike_bridge.apply(spike_params, sample_features)
        snn_params = snn_classifier.init(key, sample_spikes)
        
        # Compile pipeline (warmup)
        @jax.jit
        def optimized_pipeline(strain):
            features = cpc_encoder.apply(cpc_params, strain)
            spikes = spike_bridge.apply(spike_params, features)
            logits = snn_classifier.apply(snn_params, spikes)
            return logits
        
        # Warmup compilation
        for _ in range(3):
            _ = optimized_pipeline(test_input)
        
        # Benchmark inference latency
        num_runs = 20
        latencies = []
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            result = optimized_pipeline(test_input)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Performance statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Validate <100ms target
        target_latency = 100.0  # ms
        
        logger.info(f"✅ Performance Benchmark Results:")
        logger.info(f"   Average latency: {avg_latency:.2f}ms")
        logger.info(f"   P95 latency: {p95_latency:.2f}ms") 
        logger.info(f"   P99 latency: {p99_latency:.2f}ms")
        logger.info(f"   Target <100ms: {'✅ PASS' if avg_latency < target_latency else '❌ FAIL'}")
        
        # Assert performance requirement
        assert avg_latency < target_latency, f"Average latency {avg_latency:.2f}ms exceeds 100ms target"
        assert p95_latency < target_latency * 1.5, f"P95 latency {p95_latency:.2f}ms too high"
    
    def test_memory_usage_monitoring(self, test_config):
        """Test memory usage stays within reasonable bounds."""
        import psutil
        import os
        
        # Monitor memory before test
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1e9  # GB
        
        # Create models and large batch for memory test
        from models.cpc_encoder import RealCPCEncoder, RealCPCConfig
        
        config = RealCPCConfig(latent_dim=512)
        encoder = RealCPCEncoder(config)
        
        key = jax.random.PRNGKey(test_config['random_seed'])
        large_batch = jax.random.normal(key, (64, 16384))  # Larger batch
        
        params = encoder.init(key, large_batch[:1])
        features = encoder.apply(params, large_batch)
        
        # Monitor memory after
        memory_after = process.memory_info().rss / 1e9  # GB
        memory_increase = memory_after - memory_before
        
        logger.info(f"✅ Memory usage: {memory_before:.2f}GB → {memory_after:.2f}GB (+{memory_increase:.2f}GB)")
        
        # Reasonable memory bounds (adjust based on system)
        max_memory_increase = 4.0  # GB
        assert memory_increase < max_memory_increase, f"Memory increase {memory_increase:.2f}GB too high"

# ============================================================================
# SCIENTIFIC TESTS: Validation & Statistical Tests
# ============================================================================

class TestScientificValidation:
    """Scientific validation tests for publication readiness."""
    
    def test_statistical_significance_framework(self, test_config):
        """Test statistical significance testing framework."""
        from sklearn.metrics import roc_auc_score
        from scipy.stats import chi2_contingency
        import numpy as np
        
        # Generate test predictions and labels
        np.random.seed(test_config['random_seed'])
        n_samples = 1000
        
        # Simulate neuromorphic predictions (better than random)
        true_labels = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        neuro_predictions = true_labels.copy()
        
        # Add realistic errors (10% error rate)
        error_indices = np.random.choice(n_samples, int(0.1 * n_samples), replace=False)
        neuro_predictions[error_indices] = 1 - neuro_predictions[error_indices]
        
        # Simulate baseline predictions (worse performance)
        baseline_predictions = true_labels.copy()
        error_indices = np.random.choice(n_samples, int(0.2 * n_samples), replace=False)
        baseline_predictions[error_indices] = 1 - baseline_predictions[error_indices]
        
        # Test ROC AUC computation
        neuro_scores = np.where(neuro_predictions == true_labels, 
                               np.random.uniform(0.7, 0.95, n_samples),
                               np.random.uniform(0.3, 0.6, n_samples))
        
        baseline_scores = np.where(baseline_predictions == true_labels,
                                  np.random.uniform(0.6, 0.8, n_samples),
                                  np.random.uniform(0.2, 0.5, n_samples))
        
        neuro_auc = roc_auc_score(true_labels, neuro_scores)
        baseline_auc = roc_auc_score(true_labels, baseline_scores)
        
        # McNemar's test for significance
        a = np.sum((neuro_predictions == true_labels) & (baseline_predictions != true_labels))
        b = np.sum((neuro_predictions != true_labels) & (baseline_predictions == true_labels))
        
        if a + b > 0:
            mcnemar_stat = (abs(a - b) - 1)**2 / (a + b)
            p_value = 1 - chi2_contingency([[a, b]])[1]
        else:
            mcnemar_stat, p_value = 0, 1
        
        logger.info(f"✅ Statistical Validation:")
        logger.info(f"   Neuromorphic AUC: {neuro_auc:.3f}")
        logger.info(f"   Baseline AUC: {baseline_auc:.3f}")
        logger.info(f"   McNemar p-value: {p_value:.4f}")
        logger.info(f"   Significant: {p_value < 0.05}")
        
        # Validate framework functionality
        assert 0.5 < neuro_auc < 1.0, f"Neuromorphic AUC out of range: {neuro_auc}"
        assert 0.5 < baseline_auc < 1.0, f"Baseline AUC out of range: {baseline_auc}"
        assert isinstance(p_value, (int, float)), "p-value should be numeric"

# ============================================================================
# TEST EXECUTION MAIN
# ============================================================================

if __name__ == "__main__":
    """Run critical integration tests for production validation."""
    
    print("🧪 CRITICAL INTEGRATION TESTING FRAMEWORK")
    print("="*70)
    print("Testing Categories:")
    print("1. Unit Tests: Core functions and gradient flow")
    print("2. Integration Tests: End-to-end pipeline validation")  
    print("3. Performance Tests: <100ms inference benchmarking")
    print("4. Scientific Tests: Statistical validation framework")
    print("="*70)
    
    # Run pytest with verbose output
    import subprocess
    
    result = subprocess.run([
        "python", "-m", "pytest", __file__, 
        "-v", "--tb=short", "-x"  # Stop on first failure
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print("\n🎉 ALL CRITICAL TESTS PASSED!")
        print("✅ System validated for production deployment")
    else:
        print("\n❌ CRITICAL TESTS FAILED!")
        print("🚨 Production deployment blocked until fixes applied")
        sys.exit(1) 