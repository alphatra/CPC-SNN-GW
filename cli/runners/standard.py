"""
Standard training runner implementation.

Extracted from cli.py for better modularity.
"""

import logging
import time
from typing import Dict, Any
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def run_standard_training(config: Dict, args) -> Dict[str, Any]:
    """
    Run real CPC+SNN training using CPCSNNTrainer.
    
    Args:
        config: Training configuration dictionary
        args: Command line arguments
        
    Returns:
        Training results dictionary
    """
    logger.info("🚀 Starting STANDARD CPC+SNN Training")
    
    try:
        # Import training modules
        from training.base.trainer import CPCSNNTrainer
        from training.base.config import TrainingConfig
        # Use unified data router to respect --use-mlgwsc and config data_dir
        from cli.commands.training.data_loader import load_training_data
        
        # Create training config
        training_config = TrainingConfig(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_classes=config.get('model', {}).get('num_classes', 2),
            spike_time_steps=args.spike_time_steps,
            spike_threshold=args.spike_threshold,
            spike_learnable=args.spike_learnable,
            spike_threshold_levels=args.spike_threshold_levels,
            spike_surrogate_type=args.spike_surrogate_type,
            spike_surrogate_beta=args.spike_surrogate_beta
        )
        
        logger.info(f"   📊 Config: {args.epochs} epochs, batch={args.batch_size}")
        logger.info(f"   🧠 Model: {training_config.num_classes} classes")
        logger.info(f"   ⚡ Spikes: {args.spike_time_steps} steps, threshold={args.spike_threshold}")
        
        # Create trainer
        trainer = CPCSNNTrainer(training_config)
        
        # Load dataset via router (MLGWSC / synthetic / real)
        logger.info("📊 Loading training dataset...")
        train_signals, train_labels, test_signals, test_labels = load_training_data(args)
        # Ensure CPC input shape [B, T, F]
        if train_signals.ndim == 2:
            train_signals = train_signals[..., None]
        if test_signals.ndim == 2:
            test_signals = test_signals[..., None]
        # Reduce channels → single feature for CPC if multi-channel
        if train_signals.shape[-1] > 1:
            train_signals = jnp.mean(train_signals, axis=-1, keepdims=True)
        if test_signals.shape[-1] > 1:
            test_signals = jnp.mean(test_signals, axis=-1, keepdims=True)
        # Downsample long sequences to stabilize attention memory (target T≈512)
        def _downsample(x, target_t: int = 512):
            t = x.shape[1]
            if t <= target_t:
                return x
            factor = max(1, t // target_t)
            return x[:, ::factor, :]
        train_signals = _downsample(train_signals, 512)
        test_signals = _downsample(test_signals, 512)
        logger.info(f"   ⏬ Downsampled T: train={train_signals.shape[1]}, test={test_signals.shape[1]}, F={train_signals.shape[-1]}")
        logger.info(f"   📊 Train: {len(train_signals)} samples")
        logger.info(f"   📊 Test: {len(test_signals)} samples")
        
        # Run training
        start_time = time.time()
        training_results = trainer.train(
            train_signals=train_signals,
            train_labels=train_labels,
            test_signals=test_signals,
            test_labels=test_labels
        )
        training_time = time.time() - start_time
        
        logger.info(f"✅ Standard training completed in {training_time:.1f}s")
        logger.info(f"   📊 Final accuracy: {training_results.get('test_accuracy', 0.0):.3f}")
        
        # Save results
        results_dir = args.output_dir / "standard_training"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        results_file = results_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        logger.info(f"📊 Results saved to: {results_file}")
        
        return training_results
        
    except Exception as e:
        logger.error(f"❌ Standard training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
