#!/usr/bin/env python3
"""
Smart Device Auto-Detection for CPC-SNN-GW Pipeline

Automatically detects and configures optimal device (GPU/CPU) with appropriate settings.
Handles seamless switching between CPU-only and GPU environments without code changes.
"""

import os
import logging
import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DeviceConfig:
    """Device configuration settings"""
    platform: str  # 'gpu', 'cpu', 'metal' 
    memory_fraction: float
    use_preallocate: bool
    xla_flags: str
    recommended_batch_size: int
    recommended_epochs: int
    expected_speedup: float
    
def detect_available_devices() -> Dict[str, Any]:
    """Detect all available computational devices"""
    device_info = {
        'gpu_available': False,
        'gpu_count': 0,
        'gpu_memory_gb': 0.0,
        'cpu_cores': 0,
        'total_memory_gb': 0.0,
        'platform_detected': 'cpu'
    }
    
    # Check for GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            device_info['gpu_available'] = True
            device_info['gpu_count'] = torch.cuda.device_count()
            if device_info['gpu_count'] > 0:
                device_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                device_info['platform_detected'] = 'gpu'
        logger.info(f"PyTorch CUDA detection: {device_info['gpu_available']}")
    except ImportError:
        logger.info("PyTorch not available for GPU detection")
    
    # Check JAX devices
    try:
        jax_devices = jax.devices()
        for device in jax_devices:
            device_str = str(device).lower()
            if 'gpu' in device_str:
                device_info['gpu_available'] = True
                device_info['platform_detected'] = 'gpu'
                break
        logger.info(f"JAX devices detected: {jax_devices}")
    except Exception as e:
        logger.warning(f"JAX device detection failed: {e}")
    
    # System info
    try:
        import psutil
        device_info['cpu_cores'] = psutil.cpu_count()
        device_info['total_memory_gb'] = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass
    
    return device_info

def create_optimal_device_config(device_info: Dict[str, Any]) -> DeviceConfig:
    """Create optimal device configuration based on detected hardware"""
    
    # If GPU detected via JAX but memory unknown (no torch), assume 8GB class
    if device_info['gpu_available'] and device_info['gpu_memory_gb'] == 0.0:
        device_info['gpu_memory_gb'] = 8.0

    if device_info['gpu_available'] and device_info['gpu_memory_gb'] > 4.0:
        # 🚀 GPU Configuration (T4, V100, A100, etc.)
        logger.info(f"🎮 GPU DETECTED: {device_info['gpu_memory_gb']:.1f}GB VRAM")
        
        if device_info['gpu_memory_gb'] >= 15.0:  # T4 (16GB), V100 (16GB+)
            return DeviceConfig(
                platform='gpu',
                memory_fraction=0.85,  # Use most of GPU VRAM
                use_preallocate=False,  # Dynamic allocation
                xla_flags='--xla_gpu_cuda_data_dir=/usr/local/cuda',
                recommended_batch_size=1,  # ✅ MEMORY FIX: Ultra-small batch for GPU memory constraints
                recommended_epochs=100,  # Full training
                expected_speedup=25.0   # 25x faster than CPU
            )
        else:  # Smaller GPU (8-12GB)
            return DeviceConfig(
                platform='gpu',
                memory_fraction=0.35,  # More conservative to avoid OOM on 8-12GB
                use_preallocate=False,
                xla_flags='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_gpu_autotune_level=0',
                recommended_batch_size=1,  # ✅ MEMORY FIX: Ultra-small batch for GPU memory constraints
                recommended_epochs=100,
                expected_speedup=15.0
            )
    
    elif device_info['total_memory_gb'] > 30.0:
        # 🖥️ High-end CPU Configuration (32GB+ RAM)
        logger.info(f"💻 HIGH-END CPU: {device_info['cpu_cores']} cores, {device_info['total_memory_gb']:.1f}GB RAM")
        return DeviceConfig(
            platform='cpu',
            memory_fraction=0.6,  # More aggressive on high-end CPU
            use_preallocate=False,
            xla_flags='--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=true',
            recommended_batch_size=1,  # ✅ MEMORY FIX: Conservative batch even for high-end CPU
            recommended_epochs=50,   # Reduced for CPU
            expected_speedup=1.0
        )
    
    else:
        # 🔋 Standard CPU Configuration (16GB or less)
        logger.info(f"🔋 STANDARD CPU: {device_info['cpu_cores']} cores, {device_info['total_memory_gb']:.1f}GB RAM")
        return DeviceConfig(
            platform='cpu',
            memory_fraction=0.4,  # Conservative memory usage
            use_preallocate=False,
            xla_flags='--xla_force_host_platform_device_count=1',
            recommended_batch_size=1,   # ✅ MEMORY FIX: Ultra-small batch for limited memory
            recommended_epochs=20,   # Quick testing
            expected_speedup=1.0
        )

def apply_device_configuration(config: DeviceConfig) -> None:
    """Apply device configuration to JAX environment"""
    
    logger.info("🔧 Applying optimal device configuration...")
    logger.info(f"   Platform: {config.platform}")
    logger.info(f"   Memory fraction: {config.memory_fraction}")
    logger.info(f"   Recommended batch size: {config.recommended_batch_size}")
    logger.info(f"   Expected speedup: {config.expected_speedup:.1f}x")
    
    # Set JAX platform
    if config.platform != 'auto':
        os.environ['JAX_PLATFORM_NAME'] = config.platform
    
    # Memory configuration
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(config.memory_fraction)
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = str(config.use_preallocate).lower()
    os.environ['JAX_THREEFRY_PARTITIONABLE'] = 'true'
    
    # XLA flags
    os.environ['XLA_FLAGS'] = config.xla_flags
    
    # JAX configuration
    jax.config.update('jax_enable_x64', False)  # float32 for speed
    
    # Verify configuration
    try:
        devices = jax.devices()
        platform = jax.lib.xla_bridge.get_backend().platform
        logger.info(f"✅ JAX configured successfully:")
        logger.info(f"   Platform: {platform}")
        logger.info(f"   Devices: {devices}")
        
        if platform == 'gpu' and len(devices) > 0:
            logger.info(f"🚀 GPU ACCELERATION ACTIVE - Expected {config.expected_speedup:.1f}x speedup!")
        elif platform == 'cpu':
            logger.info(f"💻 CPU mode active - Consider GPU for {config.expected_speedup:.1f}x speedup")
            
    except Exception as e:
        logger.error(f"❌ JAX configuration verification failed: {e}")

def get_optimal_training_config(device_config: DeviceConfig) -> Dict[str, Any]:
    """Get optimal training configuration for detected device"""
    
    training_config = {
        'batch_size': device_config.recommended_batch_size,
        'num_epochs': device_config.recommended_epochs,
        'learning_rate': 1e-4,  # Base learning rate
        'memory_efficient': device_config.platform == 'cpu',
        'use_mixed_precision': device_config.platform == 'gpu',
        'gradient_accumulation_steps': 1 if device_config.platform == 'gpu' else 2,
        'val_frequency': 5 if device_config.platform == 'gpu' else 10,
        'checkpoint_frequency': 10 if device_config.platform == 'gpu' else 20,
    }
    
    # Adjust learning rate based on platform
    if device_config.platform == 'gpu':
        training_config['learning_rate'] = 2e-4  # Higher LR for GPU (larger batches)
    
    return training_config

def setup_auto_device_optimization() -> Tuple[DeviceConfig, Dict[str, Any]]:
    """
    🚀 MAIN FUNCTION: Auto-detect and configure optimal device setup
    
    Returns:
        Tuple of (DeviceConfig, TrainingConfig) optimized for detected hardware
    """
    logger.info("🔍 Starting intelligent device detection...")
    
    # Step 1: Detect available devices
    device_info = detect_available_devices()
    
    # Step 2: Create optimal configuration
    device_config = create_optimal_device_config(device_info)
    
    # Step 3: Apply configuration
    apply_device_configuration(device_config)
    
    # Step 4: Get training configuration
    training_config = get_optimal_training_config(device_config)
    
    logger.info("✅ Device auto-detection and optimization complete!")
    logger.info(f"💡 TIP: Your system is optimized for {device_config.platform.upper()} training")
    
    return device_config, training_config

if __name__ == "__main__":
    # Test auto-detection
    device_config, training_config = setup_auto_device_optimization()
    print(f"\n🎯 OPTIMAL CONFIGURATION DETECTED:")
    print(f"   Platform: {device_config.platform}")
    print(f"   Batch Size: {training_config['batch_size']}")
    print(f"   Epochs: {training_config['num_epochs']}")
    print(f"   Expected Speedup: {device_config.expected_speedup:.1f}x") 