#!/usr/bin/env python3
"""
Fixed training script with corrected 2-class configuration
✅ CRITICAL FIX: num_classes=2 to match dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import logging
from pathlib import Path
import yaml
import time
import numpy as np
from types import SimpleNamespace

# Import components
from data.real_ligo_integration import create_enhanced_ligo_dataset
from training.unified_trainer import UnifiedTrainer
from models.snn_classifier import SNNConfig
from evaluation.real_metrics_evaluator import RealMetricsEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    logger.info(f"✅ Loaded configuration from {config_path}")
    return config_dict

def create_dataset_with_2_classes():
    """
    Create dataset with proper 2-class labels.
    Ensures consistency between dataset and model.
    """
    logger.info("📊 Creating dataset with 2 classes (noise=0, gw_signal=1)...")
    
    # Create enhanced dataset - returns (signals, labels) tuple
    signals, labels = create_enhanced_ligo_dataset(
        num_samples=4000,  # More samples for better training
        enhanced_overlap=0.9,
        data_augmentation=True,
        window_size=512  # Larger window for better context
    )
    
    # Ensure labels are binary (0 or 1)
    labels = jnp.where(labels > 0, 1, 0)  # Convert any multi-class to binary
    
    # Log class distribution
    unique, counts = jnp.unique(labels, return_counts=True)
    for cls, cnt in zip(unique, counts):
        pct = cnt / len(labels) * 100
        logger.info(f"  Class {cls}: {cnt} samples ({pct:.1f}%)")
    
    # Split into train/test
    n_train = int(0.8 * len(signals))
    train_signals = signals[:n_train]
    train_labels = labels[:n_train]
    test_signals = signals[n_train:]
    test_labels = labels[n_train:]
    
    logger.info(f"✅ Dataset created: {n_train} train, {len(test_signals)} test samples")
    
    return {
        'train': (train_signals, train_labels),
        'test': (test_signals, test_labels)
    }

def create_fixed_config(config_dict):
    """
    Create configuration object with fixed num_classes=2.
    """
    # Create namespace for easy attribute access
    config = SimpleNamespace()
    
    # Data settings
    config.batch_size = config_dict['training']['batch_size']
    config.sequence_length = config_dict['data']['sequence_length']
    config.sample_rate = config_dict['data']['sample_rate']
    
    # ✅ CRITICAL FIX: Ensure num_classes=2
    config.num_classes = 2
    
    # Model settings
    config.cpc_latent_dim = config_dict['model']['cpc_latent_dim']
    config.cpc_context_length = config_dict['model']['cpc_context_length']
    config.cpc_downsample_factor = config_dict['model']['cpc_downsample_factor']
    config.cpc_num_negatives = config_dict['model']['cpc_num_negatives']
    config.cpc_temperature = config_dict['model']['cpc_temperature']
    
    # SNN settings with 2 classes
    config.snn_hidden_sizes = tuple(config_dict['model']['snn_layer_sizes'])
    config.snn_num_layers = config_dict['model']['snn_num_layers']
    config.snn_layer_norm = config_dict['model']['snn_layer_norm']
    config.snn_surrogate_slope = config_dict['model']['snn_surrogate_slope']
    config.snn_tau_mem = config_dict['model']['snn_tau_mem']
    config.snn_tau_syn = config_dict['model']['snn_tau_syn']
    config.snn_threshold = config_dict['model']['snn_threshold']
    config.snn_dropout_rate = config_dict['model']['snn_dropout_rate']
    config.snn_hidden_size = config_dict['model']['snn_layer_sizes'][0]
    
    # Spike encoding
    config.spike_encoding = config_dict['model']['spike_encoding']
    config.spike_threshold_pos = config_dict['model']['spike_threshold_pos']
    config.spike_threshold_neg = config_dict['model']['spike_threshold_neg']
    config.spike_time_steps = config_dict['model']['spike_time_steps']
    
    # Training settings
    config.num_epochs = config_dict['training']['num_epochs']
    config.learning_rate = config_dict['training']['joint_lr']
    config.cpc_lr = config_dict['training']['cpc_lr']
    config.snn_lr = config_dict['training']['snn_lr']
    config.joint_lr = config_dict['training']['joint_lr']
    config.cpc_epochs = config_dict['training']['cpc_epochs']
    config.snn_epochs = config_dict['training']['snn_epochs']
    config.joint_epochs = config_dict['training']['joint_epochs']
    config.gradient_clip = config_dict['training']['gradient_clip']
    config.weight_decay = config_dict['training']['weight_decay']
    config.scheduler = config_dict['training']['scheduler']
    config.warmup_epochs = config_dict['training']['warmup_epochs']
    
    # Training flags
    config.enable_cpc_finetuning_stage2 = config_dict['training']['enable_cpc_finetuning_stage2']
    config.grad_accumulation_steps = config_dict['training']['grad_accumulation_steps']
    
    # Evaluation
    config.eval_every_epochs = config_dict['training']['eval_every_epochs']
    config.compute_roc_auc = config_dict['training']['compute_roc_auc']
    config.bootstrap_samples = config_dict['training']['bootstrap_samples']
    config.target_far = config_dict['training']['target_far']
    
    # Platform
    config.device = config_dict['platform']['device']
    config.memory_fraction = config_dict['platform']['memory_fraction']
    
    # Output
    config.output_dir = Path(config_dict['output_dir'])
    config.experiment_name = config_dict['experiment_name']
    config.run_name = f"{config_dict['experiment_name']}_{int(time.time())}"
    config.project_name = "cpc-snn-gw-fixed"  # Add missing attribute
    
    # Reproducibility
    config.random_seed = config_dict['training']['random_seed']
    config.seed = config_dict['training']['random_seed']
    
    # Logging
    config.log_every_n_steps = config_dict['logging']['log_every_n_steps']
    config.save_every_n_epochs = config_dict['logging']['save_every_n_epochs']
    config.use_wandb = config_dict['logging']['use_wandb']
    
    logger.info(f"✅ Configuration created with num_classes={config.num_classes}")
    
    return config

def main():
    """Main training pipeline with fixed configuration."""
    
    logger.info("="*70)
    logger.info("🚀 CPC-SNN-GW Training with FIXED 2-class Configuration")
    logger.info("="*70)
    
    # Set environment for better GPU utilization
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
    os.environ['JAX_ENABLE_X64'] = 'True'
    
    # Load configuration
    config_path = "configs/fixed_config.yaml"
    config_dict = load_config(config_path)
    config = create_fixed_config(config_dict)
    
    # Log critical settings
    logger.info("\n📋 Key Configuration:")
    logger.info(f"  num_classes: {config.num_classes} ✅ FIXED")
    logger.info(f"  batch_size: {config.batch_size}")
    logger.info(f"  learning_rate: {config.learning_rate}")
    logger.info(f"  SNN layers: {config.snn_hidden_sizes}")
    logger.info(f"  epochs: {config.num_epochs}")
    logger.info(f"  device: {config.device}")
    
    # Create dataset with 2 classes
    dataset = create_dataset_with_2_classes()
    train_data, train_labels = dataset['train']
    test_data, test_labels = dataset['test']
    
    # Verify class consistency
    train_classes = len(jnp.unique(train_labels))
    test_classes = len(jnp.unique(test_labels))
    assert train_classes == 2, f"Train has {train_classes} classes, expected 2"
    assert test_classes == 2, f"Test has {test_classes} classes, expected 2"
    logger.info(f"✅ Class consistency verified: {train_classes} classes in both sets")
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_save_path = config.output_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    logger.info(f"💾 Configuration saved to {config_save_path}")
    
    # Initialize trainer
    logger.info("\n🔧 Initializing UnifiedTrainer...")
    trainer = UnifiedTrainer(config)
    
    # Initialize evaluator
    evaluator = RealMetricsEvaluator(
        num_classes=2,
        target_far=config.target_far,
        bootstrap_samples=config.bootstrap_samples
    )
    
    # Training loop
    logger.info("\n🚀 Starting training...")
    logger.info(f"  Train samples: {len(train_data)}")
    logger.info(f"  Test samples: {len(test_data)}")
    
    # Run training
    history = trainer.train(
        train_data=train_data,
        train_labels=train_labels,
        val_data=test_data,
        val_labels=test_labels
    )
    
    # Final evaluation
    logger.info("\n📊 Final Evaluation:")
    final_metrics = evaluator.evaluate(
        trainer.predict(test_data),
        test_labels
    )
    
    # Log final results
    logger.info("\n✅ Training Complete!")
    logger.info(f"  Final ROC-AUC: {final_metrics.get('roc_auc', 0):.4f}")
    logger.info(f"  Final TPR@FAR: {final_metrics.get('tpr_at_far', 0):.4f}")
    logger.info(f"  Final Accuracy: {final_metrics.get('accuracy', 0):.4f}")
    logger.info(f"  Final F1 Score: {final_metrics.get('f1_score', 0):.4f}")
    
    # Save final model
    model_path = config.output_dir / "final_model.pkl"
    trainer.save_model(model_path)
    logger.info(f"💾 Model saved to {model_path}")
    
    return history, final_metrics

if __name__ == "__main__":
    try:
        history, metrics = main()
        logger.info("\n🎉 SUCCESS! Training completed successfully")
    except Exception as e:
        logger.error(f"\n❌ ERROR: Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)