"""
Standard training runner implementation.

Extracted from cli.py for better modularity.
"""

import logging
import time
from typing import Dict, Any

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
        from ...training.base_trainer import CPCSNNTrainer, TrainingConfig
        from ...data.gw_dataset_builder import create_evaluation_dataset
        
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
            spike_surrogate_beta=args.spike_surrogate_beta,
            random_seed=args.random_seed
        )
        
        logger.info(f"   📊 Config: {args.epochs} epochs, batch={args.batch_size}")
        logger.info(f"   🧠 Model: {training_config.num_classes} classes")
        logger.info(f"   ⚡ Spikes: {args.spike_time_steps} steps, threshold={args.spike_threshold}")
        
        # Create trainer
        trainer = CPCSNNTrainer(training_config)
        
        # Create or load dataset
        logger.info("📊 Loading training dataset...")
        dataset = create_evaluation_dataset(
            num_samples=1000,  # Default for standard training
            sequence_length=512,
            sample_rate=2048,
            random_key=42
        )
        
        train_signals = dataset['data'][:800]  # 80% for training
        train_labels = dataset['labels'][:800]
        test_signals = dataset['data'][800:]   # 20% for testing
        test_labels = dataset['labels'][800:]
        
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
