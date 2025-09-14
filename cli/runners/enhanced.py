"""
Enhanced training runner implementation.

Extracted from cli.py for better modularity.
"""

import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)


def run_enhanced_training(config: Dict, args) -> Dict[str, Any]:
    """
    Run enhanced training with mixed continuous+binary dataset.
    
    Args:
        config: Training configuration dictionary
        args: Command line arguments
        
    Returns:
        Training results dictionary
    """
    logger.info("🚀 Starting ENHANCED CPC+SNN Training")
    
    try:
        # Import enhanced training modules
        from ...training.enhanced_gw_training import EnhancedGWTrainer, EnhancedGWConfig
        
        # Create enhanced config
        enhanced_config = EnhancedGWConfig(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_classes=config.get('model', {}).get('num_classes', 3),  # Enhanced: 3 classes
            
            # Enhanced features
            use_temporal_transformer=True,
            transformer_num_layers=args.cpc_layers,
            use_learnable_thresholds=args.spike_learnable,
            num_threshold_levels=args.spike_threshold_levels,
            threshold_adaptation_rate=0.01,
            
            # Spike parameters
            spike_time_steps=args.spike_time_steps,
            spike_threshold=args.spike_threshold,
            spike_surrogate_type=args.spike_surrogate_type,
            spike_surrogate_beta=args.spike_surrogate_beta,
            
            random_seed=args.random_seed
        )
        
        logger.info("🌟 Enhanced Training Features:")
        logger.info("   1. 🧠 Adaptive Multi-Scale Surrogate Gradients")
        logger.info("   2. 🔄 Temporal Transformer with Multi-Scale Convolution")
        logger.info("   3. 🎯 Learnable Multi-Threshold Spike Encoding")
        logger.info("   4. 💾 Enhanced LIF with Memory and Refractory Period")
        logger.info("   5. 🚀 Momentum-based InfoNCE with Hard Negative Mining")
        
        # Create enhanced trainer
        trainer = EnhancedGWTrainer(enhanced_config)
        
        # Create enhanced dataset
        logger.info("📊 Loading enhanced training dataset...")
        
        # Use enhanced dataset builder
        from ...data.gw_dataset_builder import create_evaluation_dataset
        
        dataset = create_evaluation_dataset(
            num_samples=2000,  # Larger dataset for enhanced training
            sequence_length=512,
            sample_rate=2048,
            random_key=42
        )
        
        # Enhanced data split (70/15/15)
        total_samples = len(dataset['data'])
        train_end = int(0.7 * total_samples)
        val_end = int(0.85 * total_samples)
        
        train_signals = dataset['data'][:train_end]
        train_labels = dataset['labels'][:train_end]
        val_signals = dataset['data'][train_end:val_end]
        val_labels = dataset['labels'][train_end:val_end]
        test_signals = dataset['data'][val_end:]
        test_labels = dataset['labels'][val_end:]
        
        logger.info(f"   📊 Train: {len(train_signals)} samples")
        logger.info(f"   📊 Val: {len(val_signals)} samples")
        logger.info(f"   📊 Test: {len(test_signals)} samples")
        
        # Run enhanced training
        start_time = time.time()
        training_results = trainer.run_full_training_pipeline()
        training_time = time.time() - start_time
        
        logger.info(f"✅ Enhanced training completed in {training_time:.1f}s")
        logger.info(f"   📊 Final accuracy: {training_results.get('test_accuracy', 0.0):.3f}")
        logger.info(f"   📊 Best F1 score: {training_results.get('best_f1', 0.0):.3f}")
        
        # Save enhanced results
        results_dir = args.output_dir / "enhanced_training"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        results_file = results_dir / "enhanced_results.json"
        with open(results_file, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        logger.info(f"📊 Enhanced results saved to: {results_file}")
        
        return training_results
        
    except Exception as e:
        logger.error(f"❌ Enhanced training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
