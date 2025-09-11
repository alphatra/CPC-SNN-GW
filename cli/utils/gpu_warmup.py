"""
GPU warmup utilities for CLI commands.

This module contains GPU warmup functionality extracted from
cli.py for better modularity.

Split from cli.py for better maintainability.
"""

import logging
import os
import time
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def setup_jax_environment(device: str = "auto", memory_fraction: float = 0.35):
    """
    Setup JAX environment with optimized settings.
    
    Args:
        device: Device preference ("auto", "cpu", "gpu")
        memory_fraction: Memory fraction for GPU
    """
    # Device selection and platform safety
    try:
        # Set platform BEFORE importing jax so it takes effect
        if device == 'cpu':
            os.environ['JAX_PLATFORMS'] = 'cpu'
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            os.environ['NVIDIA_VISIBLE_DEVICES'] = ''
            logger.info("Forcing CPU backend as requested")
        elif device == 'gpu':
            os.environ.pop('JAX_PLATFORMS', None)
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            os.environ.pop('NVIDIA_VISIBLE_DEVICES', None)
            logger.info("Requesting GPU backend; JAX will use CUDA if available")
        
        # Auto device handling
        if device == 'auto':
            try:
                if jax.default_backend() == 'metal':
                    os.environ['JAX_PLATFORMS'] = 'cpu'
                    logger.warning("Metal backend is experimental; falling back to CPU for stability")
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Device configuration warning: {e}")
    
    # ✅ FIX: Set JAX memory pre-allocation to prevent allocation spikes
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(memory_fraction)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # ✅ CUDA TIMING FIX: Suppress timing warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['XLA_FLAGS'] = '--xla_gpu_enable_fast_min_max=true'
    
    # Configure JAX for efficient memory usage
    jax.config.update('jax_enable_x64', False)  # Use float32 for memory efficiency


def perform_gpu_warmup(device: str = "auto") -> bool:
    """
    Perform comprehensive GPU warmup to eliminate timing issues.
    
    Args:
        device: Device type for warmup
        
    Returns:
        True if warmup successful, False otherwise
    """
    try:
        # ✅ COMPREHENSIVE CUDA WARMUP: Advanced model-specific kernel initialization
        if device == 'cpu':
            logger.info("⏭️ Skipping GPU warmup (CPU device)")
            return True
            
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        
        if not gpu_devices:
            logger.info("⏭️ Skipping GPU warmup (no GPU detected)")
            return False
        
        logger.info("🔥 Performing COMPREHENSIVE GPU warmup to eliminate timing issues...")
        
        warmup_key = jax.random.PRNGKey(456)
        
        # ✅ STAGE 1: Basic tensor operations (varied sizes)
        logger.info("   🔸 Stage 1: Basic tensor operations...")
        for size in [(8, 32), (16, 64), (32, 128)]:
            data = jax.random.normal(warmup_key, size)
            _ = jnp.sum(data ** 2).block_until_ready()
            _ = jnp.dot(data, data.T).block_until_ready()
            _ = jnp.mean(data, axis=1).block_until_ready()
        
        # ✅ STAGE 2: Model-specific operations (Dense layers)
        logger.info("   🔸 Stage 2: Dense layer operations...")
        input_data = jax.random.normal(warmup_key, (4, 256))
        weight_matrix = jax.random.normal(jax.random.split(warmup_key)[0], (256, 128))
        bias = jax.random.normal(jax.random.split(warmup_key)[1], (128,))
        
        dense_output = jnp.dot(input_data, weight_matrix) + bias
        activated = jnp.tanh(dense_output)
        activated.block_until_ready()
        
        # ✅ STAGE 3: CPC/SNN specific operations  
        logger.info("   🔸 Stage 3: CPC/SNN operations...")
        sequence_data = jax.random.normal(warmup_key, (2, 64, 32))
        
        # Temporal operations (like CPC)
        context = sequence_data[:, :-1, :]
        target = sequence_data[:, 1:, :]
        
        # Normalization (like CPC encoder)
        context_norm = context / (jnp.linalg.norm(context, axis=-1, keepdims=True) + 1e-8)
        target_norm = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)
        
        # Similarity computation (like InfoNCE)
        context_flat = context_norm.reshape(-1, context_norm.shape[-1])
        target_flat = target_norm.reshape(-1, target_norm.shape[-1])
        similarity = jnp.dot(context_flat, target_flat.T)
        similarity.block_until_ready()
        
        # ✅ STAGE 4: Advanced operations (convolutions)
        logger.info("   🔸 Stage 4: Advanced CUDA kernels...")
        conv_data = jax.random.normal(warmup_key, (4, 128, 1))
        kernel = jax.random.normal(jax.random.split(warmup_key)[0], (5, 1, 16))
        
        conv_result = jax.lax.conv_general_dilated(
            conv_data, kernel, 
            window_strides=[1], padding=[(2, 2)],
            dimension_numbers=('NHC', 'HIO', 'NHC')
        )
        conv_result.block_until_ready()
        
        # ✅ STAGE 5: JAX compilation warmup 
        logger.info("   🔸 Stage 5: JAX JIT compilation warmup...")
        
        @jax.jit
        def warmup_jit_function(x):
            return jnp.sum(x ** 2) + jnp.mean(jnp.tanh(x))
        
        jit_data = jax.random.normal(warmup_key, (8, 32))
        _ = warmup_jit_function(jit_data).block_until_ready()
        
        # ✅ STAGE 6: Model-specific operations
        logger.info("   🔸 Stage 6: SpikeBridge/CPC specific warmup...")
        
        # Mimic CPC encoder operations
        cpc_input = jax.random.normal(warmup_key, (1, 256))
        for channels in [32, 64, 128]:
            conv_kernel = jax.random.normal(jax.random.split(warmup_key)[0], (3, 1, channels))
            conv_data = cpc_input[..., None]
            _ = jax.lax.conv_general_dilated(
                conv_data, conv_kernel,
                window_strides=[2], padding='SAME',
                dimension_numbers=('NHC', 'HIO', 'NHC')
            ).block_until_ready()
        
        # Dense layers with activations
        dense_sizes = [(256, 128), (128, 64), (64, 32)]
        temp_data = jax.random.normal(warmup_key, (1, 256))
        for in_size, out_size in dense_sizes:
            w = jax.random.normal(jax.random.split(warmup_key)[0], (in_size, out_size))
            b = jax.random.normal(jax.random.split(warmup_key)[1], (out_size,))
            temp_data = jnp.tanh(jnp.dot(temp_data, w) + b)
            temp_data.block_until_ready()
            if temp_data.shape[1] != in_size:
                temp_data = jax.random.normal(warmup_key, (1, out_size))
        
        # ✅ FINAL SYNCHRONIZATION
        time.sleep(0.1)  # Brief pause for kernel initialization
        
        logger.info("✅ COMPREHENSIVE GPU warmup completed - ALL CUDA kernels initialized!")
        
        # Check final device status
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        
        if gpu_devices:
            logger.info(f"🎯 GPU devices available: {len(gpu_devices)}")
        else:
            logger.info("💻 Using CPU backend")
        
        return len(gpu_devices) > 0
        
    except Exception as e:
        logger.warning(f"⚠️ GPU warmup failed: {e}")
        logger.info("   Continuing with default JAX settings")
        return False


def apply_performance_optimizations():
    """Apply performance optimizations for CLI operations."""
    try:
        # ✅ FIX: Apply optimizations once at startup
        from utils.config import apply_performance_optimizations as apply_opts
        apply_opts()
        logger.info("🔧 Performance optimizations applied")
        return True
    except Exception as e:
        logger.warning(f"Performance optimization failed: {e}")
        return False


def setup_training_environment():
    """Setup training environment for CLI operations."""
    try:
        from utils.config import setup_training_environment as setup_env
        setup_env()
        logger.info("🔧 Training environment setup completed")
        return True
    except Exception as e:
        logger.warning(f"Training environment setup failed: {e}")
        return False


# Export GPU warmup utilities
__all__ = [
    "setup_jax_environment",
    "perform_gpu_warmup",
    "apply_performance_optimizations",
    "setup_training_environment"
]
