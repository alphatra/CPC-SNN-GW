#!/usr/bin/env python3
"""
Training script with all fixes applied

✅ COMPLETE: Integrates all improvements from audit
- End-to-end gradient flow
- 3-layer SNN with LayerNorm
- PSD whitening with aLIGO
- MLGWSC-1 dataset (100k+ samples)
- Real evaluation metrics
- Reproducible seeds
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import logging
from pathlib import Path
import time
import numpy as np

# Import fixed components
from data.mlgwsc_dataset_loader import load_mlgwsc_for_training
from evaluation.real_metrics_evaluator import create_evaluator
from training.unified_trainer import UnifiedTrainer
from utils.config import TrainingConfig
from models.snn_classifier import SNNConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_enhanced_config():
    """
    Create configuration with all fixes applied.
    """
    config = TrainingConfig(
        # Dataset - use MLGWSC-1 with 100k+ samples
        dataset_type="mlgwsc",
        num_samples=100000,  # ✅ 2778x more than before (36 → 100k)
        
        # Architecture - 3 layers with LayerNorm
        snn_config=SNNConfig(
            hidden_sizes=(256, 128, 64),  # ✅ 3 deep layers
            num_layers=3,
            surrogate_beta=10.0,  # ✅ Increased for better gradients
            use_enhanced_lif=True,
            use_learnable_dynamics=True,
            dropout_rate=0.1
        ),
        
        # Training parameters
        batch_size=32,
        num_epochs=100,
        learning_rate=5e-4,
        
        # Enable all fixes
        enable_cpc_finetuning_stage2=True,  # ✅ End-to-end gradient flow
        apply_whitening=True,  # ✅ PSD whitening
        
        # Reproducibility
        seed=42,  # ✅ Fixed seed (not time.time())
        
        # Evaluation
        evaluate_every=5,
        save_checkpoints=True,
        
        # Output
        output_dir=Path("outputs/fixed_training"),
        experiment_name="cpc-snn-gw-fixed",
        
        # Logging
        use_wandb=False,  # Set to True if you have W&B configured
        verbose=True
    )
    
    return config


def main():
    """
    Main training pipeline with all fixes.
    """
    logger.info("=" * 70)
    logger.info("🚀 CPC-SNN-GW Training with All Fixes Applied")
    logger.info("=" * 70)
    
    # 1. Load configuration
    config = create_enhanced_config()
    logger.info("✅ Configuration created with all fixes")
    
    # 2. Load MLGWSC-1 dataset (100k+ samples)
    logger.info("\n📊 Loading MLGWSC-1 dataset...")
    dataset = load_mlgwsc_for_training(num_samples=config.num_samples)
    
    train_data, train_labels = dataset['train']
    val_data, val_labels = dataset['val']
    test_data, test_labels = dataset['test']
    
    logger.info(f"✅ Dataset loaded:")
    logger.info(f"   Train: {train_data.shape}")
    logger.info(f"   Val: {val_data.shape}")
    logger.info(f"   Test: {test_data.shape}")
    
    # 3. Create unified trainer with fixes
    logger.info("\n🔧 Creating trainer with enhanced architecture...")
    trainer = UnifiedTrainer(config)
    
    # 4. Initialize model
    logger.info("\n🧠 Initializing model...")
    sample_input = train_data[:1]  # Single sample for initialization
    trainer.initialize(sample_input)
    
    # 5. Training loop
    logger.info("\n🏋️ Starting training...")
    logger.info("=" * 70)
    
    best_val_accuracy = 0.0
    evaluator = create_evaluator()
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # Training step
        train_metrics = trainer.train_epoch(train_data, train_labels)
        
        # Validation
        if (epoch + 1) % config.evaluate_every == 0:
            val_metrics = trainer.evaluate(val_data, val_labels)
            
            # Real evaluation metrics
            val_scores = trainer.predict_proba(val_data)
            eval_metrics = evaluator.evaluate(
                val_labels.numpy() if hasattr(val_labels, 'numpy') else val_labels,
                val_scores
            )
            
            # Log progress
            epoch_time = time.time() - epoch_start
            logger.info(f"\nEpoch {epoch+1}/{config.num_epochs} ({epoch_time:.1f}s)")
            logger.info(f"  Train Loss: {train_metrics.get('loss', 0):.4f}")
            logger.info(f"  Train Acc: {train_metrics.get('accuracy', 0):.3f}")
            logger.info(f"  Val Acc: {val_metrics.get('accuracy', 0):.3f}")
            logger.info(f"  ROC-AUC: {eval_metrics.roc_auc:.3f} "
                       f"[{eval_metrics.roc_auc_ci[0]:.3f}, {eval_metrics.roc_auc_ci[1]:.3f}]")
            logger.info(f"  TPR@FAR=1/30d: {eval_metrics.tpr_at_far:.3f} "
                       f"[{eval_metrics.tpr_at_far_ci[0]:.3f}, {eval_metrics.tpr_at_far_ci[1]:.3f}]")
            
            # Check for model collapse
            if eval_metrics.model_collapse:
                logger.warning(f"⚠️ Model collapse detected to class {eval_metrics.collapsed_class}")
            
            # Save best model
            if val_metrics.get('accuracy', 0) > best_val_accuracy:
                best_val_accuracy = val_metrics.get('accuracy', 0)
                trainer.save_checkpoint(epoch, is_best=True)
                logger.info(f"  ✅ New best model saved (Val Acc: {best_val_accuracy:.3f})")
    
    # 6. Final test evaluation
    logger.info("\n" + "=" * 70)
    logger.info("📊 Final Test Evaluation")
    logger.info("=" * 70)
    
    test_scores = trainer.predict_proba(test_data)
    test_metrics = evaluator.evaluate(
        test_labels.numpy() if hasattr(test_labels, 'numpy') else test_labels,
        test_scores
    )
    
    logger.info(f"\n🎯 Final Test Results:")
    logger.info(f"  Accuracy: {test_metrics.accuracy:.3f}")
    logger.info(f"  ROC-AUC: {test_metrics.roc_auc:.3f} "
               f"[{test_metrics.roc_auc_ci[0]:.3f}, {test_metrics.roc_auc_ci[1]:.3f}]")
    logger.info(f"  TPR@FAR=1/30d: {test_metrics.tpr_at_far:.3f} "
               f"[{test_metrics.tpr_at_far_ci[0]:.3f}, {test_metrics.tpr_at_far_ci[1]:.3f}]")
    logger.info(f"  F1 Score: {test_metrics.f1:.3f} "
               f"[{test_metrics.f1_ci[0]:.3f}, {test_metrics.f1_ci[1]:.3f}]")
    logger.info(f"  Precision: {test_metrics.precision:.3f}")
    logger.info(f"  Recall: {test_metrics.recall:.3f}")
    logger.info(f"  Specificity: {test_metrics.specificity:.3f}")
    
    # Plot results
    if test_metrics.roc_curve:
        save_path = config.output_dir / "test_metrics.png"
        evaluator.plot_metrics(test_metrics, save_path=save_path)
    
    # 7. Summary
    logger.info("\n" + "=" * 70)
    logger.info("✅ Training Complete!")
    logger.info("=" * 70)
    logger.info(f"\nImprovements achieved:")
    logger.info(f"  • End-to-end gradient flow enabled")
    logger.info(f"  • 3-layer SNN with LayerNorm")
    logger.info(f"  • PSD whitening with aLIGO")
    logger.info(f"  • {config.num_samples:,} training samples (vs 36 before)")
    logger.info(f"  • Real evaluation metrics with CI")
    logger.info(f"  • Reproducible with fixed seeds")
    
    logger.info(f"\nExpected vs Actual:")
    logger.info(f"  Expected ROC-AUC: 0.75-0.80")
    logger.info(f"  Actual ROC-AUC: {test_metrics.roc_auc:.3f}")
    logger.info(f"  Expected TPR@FAR: +20-30%")
    logger.info(f"  Actual TPR@FAR: {test_metrics.tpr_at_far:.3f}")
    
    return test_metrics


if __name__ == "__main__":
    # Check JAX backend
    logger.info(f"JAX backend: {jax.default_backend()}")
    logger.info(f"JAX devices: {jax.devices()}")
    
    # Run training
    try:
        metrics = main()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
