"""
Training environment initialization and setup.

This module contains training setup functionality extracted from
train.py for better modularity.

Split from cli/commands/train.py for better maintainability.
"""

import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)


def setup_training_environment(args) -> Dict[str, Any]:
    """
    Setup complete training environment with optimized settings.
    
    Args:
        args: CLI arguments
        
    Returns:
        Environment setup results
    """
    logger.info("🔧 Setting up training environment...")
    
    # Import utilities
    from cli.utils.gpu_warmup import setup_jax_environment, perform_gpu_warmup, apply_performance_optimizations, setup_training_environment as setup_env
    
    setup_results = {}
    
    try:
        # ✅ JAX Environment setup
        setup_jax_environment(args.device)
        setup_results['jax_environment'] = True
        
        # ✅ GPU warmup if needed
        if args.device != 'cpu':
            warmup_success = perform_gpu_warmup(args.device)
            setup_results['gpu_warmup'] = warmup_success
            
            if warmup_success:
                logger.info("✅ GPU warmup completed successfully")
            else:
                logger.warning("⚠️ GPU warmup failed - continuing with CPU")
        else:
            setup_results['gpu_warmup'] = False
            logger.info("⏭️ Skipping GPU warmup (CPU mode)")
        
        # ✅ Performance optimizations
        opt_success = apply_performance_optimizations()
        setup_results['optimizations'] = opt_success
        
        # ✅ Training environment
        env_success = setup_env()
        setup_results['training_env'] = env_success
        
        # ✅ Device auto-detection
        try:
            from utils.device_auto_detection import setup_auto_device_optimization
            device_config, optimal_config = setup_auto_device_optimization()
            
            setup_results['device_config'] = {
                'platform': device_config.platform,
                'expected_speedup': device_config.expected_speedup
            }
            setup_results['optimal_config'] = optimal_config
            
            logger.info(f"🎮 Platform detected: {device_config.platform.upper()}")
            logger.info(f"⚡ Expected speedup: {device_config.expected_speedup:.1f}x")
            
        except ImportError:
            logger.warning("Auto-detection not available, using default settings")
            setup_results['device_config'] = None
            setup_results['optimal_config'] = {}
        
        # ✅ Memory settings validation
        memory_fraction = float(os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.35'))
        setup_results['memory_fraction'] = memory_fraction
        
        if memory_fraction > 0.8:
            logger.warning(f"High memory fraction: {memory_fraction} - consider reducing")
        
        logger.info("✅ Training environment setup completed successfully")
        setup_results['success'] = True
        
        return setup_results
        
    except Exception as e:
        logger.error(f"❌ Training environment setup failed: {e}")
        setup_results['success'] = False
        setup_results['error'] = str(e)
        return setup_results


def validate_training_setup(setup_results: Dict[str, Any]) -> bool:
    """
    Validate that training setup was successful.
    
    Args:
        setup_results: Results from setup_training_environment
        
    Returns:
        True if setup is valid for training
    """
    if not setup_results.get('success', False):
        logger.error("❌ Training setup validation failed - setup was unsuccessful")
        return False
    
    # Check critical components
    required_components = ['jax_environment', 'training_env']
    
    for component in required_components:
        if not setup_results.get(component, False):
            logger.error(f"❌ Required component not setup: {component}")
            return False
    
    # Validate memory settings
    memory_fraction = setup_results.get('memory_fraction', 0.0)
    if memory_fraction <= 0 or memory_fraction > 1.0:
        logger.error(f"❌ Invalid memory fraction: {memory_fraction}")
        return False
    
    logger.info("✅ Training setup validation passed")
    return True


def get_recommended_training_config(args) -> Dict[str, Any]:
    """
    Get recommended training configuration based on detected hardware.
    
    Args:
        args: CLI arguments
        
    Returns:
        Recommended configuration dictionary
    """
    # Base configuration
    config = {
        'batch_size': args.batch_size,
        'learning_rate': getattr(args, 'learning_rate', 1e-3),
        'num_epochs': args.epochs,
        'optimizer': 'adamw',
        'scheduler': 'cosine'
    }
    
    # Hardware-specific optimizations
    device_config = getattr(args, 'device_config', None)
    
    if device_config and hasattr(device_config, 'platform'):
        platform = device_config.platform.lower()
        
        if 'gpu' in platform:
            # GPU optimizations
            config.update({
                'mixed_precision': True,
                'gradient_accumulation_steps': max(1, 16 // args.batch_size),
                'memory_efficient': True
            })
        elif 'cpu' in platform:
            # CPU optimizations
            config.update({
                'mixed_precision': False,
                'batch_size': min(args.batch_size, 2),  # Smaller batch for CPU
                'gradient_accumulation_steps': 1,
                'memory_efficient': True
            })
    
    # Mode-specific adjustments
    if hasattr(args, 'mode'):
        if args.mode == 'complete_enhanced':
            config.update({
                'batch_size': min(config['batch_size'], 4),  # Conservative for enhanced
                'learning_rate': config['learning_rate'] * 0.5,  # Lower LR for stability
                'gradient_clipping': True
            })
        elif args.mode == 'standard':
            config.update({
                'optimizer': 'sgd',  # More memory efficient
                'learning_rate': config['learning_rate'] * 0.1  # Very conservative
            })
    
    return config


# Export training initialization functions
__all__ = [
    "setup_training_environment",
    "validate_training_setup",
    "get_recommended_training_config"
]
