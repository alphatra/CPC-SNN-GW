"""
Standard CPC+SNN training implementation.

This module contains standard training functionality extracted from
train.py for better modularity.

Split from cli/commands/train.py for better maintainability.
"""

import logging
import time
from typing import Dict, Any
from pathlib import Path

import jax.numpy as jnp

from .data_loader import load_training_data
from .initializer import setup_training_environment, validate_training_setup, get_recommended_training_config

logger = logging.getLogger(__name__)


def run_standard_training(config: Dict, args) -> Dict[str, Any]:
    """
    Run real CPC+SNN training using CPCSNNTrainer.
    
    Args:
        config: Training configuration
        args: CLI arguments
        
    Returns:
        Training results dictionary
    """
    try:
        # Setup training environment
        setup_results = setup_training_environment(args)
        if not validate_training_setup(setup_results):
            raise RuntimeError("Training environment setup failed")
        
        # Import training components
        from training.base_trainer import CPCSNNTrainer, TrainingConfig
        
        # Create output directory
        training_dir = args.output_dir / f"standard_training_{config['training']['batch_size']}bs"
        training_dir.mkdir(parents=True, exist_ok=True)
        
        # Get recommended configuration
        recommended_config = get_recommended_training_config(args)
        
        # Create training configuration
        trainer_config = TrainingConfig(
            model_name="cpc_snn_gw",
            learning_rate=recommended_config['learning_rate'],
            batch_size=recommended_config['batch_size'],
            num_epochs=recommended_config['num_epochs'],
            output_dir=str(training_dir),
            use_wandb=getattr(args, 'wandb', False),
            optimizer=recommended_config['optimizer'],
            scheduler=recommended_config['scheduler']
        )
        
        logger.info("🔧 Standard CPC+SNN training pipeline:")
        logger.info(f"   - Batch Size: {trainer_config.batch_size}")
        logger.info(f"   - Learning Rate: {trainer_config.learning_rate}")
        logger.info(f"   - Epochs: {trainer_config.num_epochs}")
        logger.info(f"   - Optimizer: {trainer_config.optimizer}")
        
        # Create trainer and model
        trainer = CPCSNNTrainer(trainer_config)
        model = trainer.create_model()
        
        # Load training data
        logger.info("📊 Loading training data...")
        signals, labels, test_signals, test_labels = load_training_data(args)
        
        # Initialize training state
        sample_input = signals[:1]
        trainer.train_state = trainer.create_train_state(model, sample_input)
        
        # Setup checkpointing
        checkpoint_managers = _setup_checkpointing(args, training_dir)
        
        # Run training loop
        logger.info("⏳ Starting standard training loop...")
        training_results = _execute_training_loop(
            trainer, trainer_config, signals, labels, training_dir, checkpoint_managers
        )
        
        # Final evaluation
        final_results = _run_final_evaluation(
            trainer, test_signals, test_labels, signals, training_results
        )
        
        logger.info("🎉 Standard training completed successfully!")
        return final_results
        
    except Exception as e:
        logger.error(f"❌ Standard training failed: {e}")
        return {'success': False, 'error': str(e)}


def _setup_checkpointing(args, training_dir):
    """Setup checkpoint managers for training."""
    checkpoint_managers = {'best': None, 'latest': None}
    
    # Skip Orbax in quick-mode to reduce overhead
    if getattr(args, 'quick_mode', False):
        logger.info("⚡ Quick-mode: disabling Orbax checkpoint managers")
        return checkpoint_managers
    
    try:
        import orbax.checkpoint as ocp
        
        ckpt_root = (training_dir / "ckpts").resolve()
        (ckpt_root / "best").mkdir(parents=True, exist_ok=True)
        (ckpt_root / "latest").mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint managers
        handler = ocp.PyTreeCheckpointHandler()
        
        checkpoint_managers['best'] = ocp.CheckpointManager(
            directory=str((ckpt_root / "best").resolve()),
            checkpointers={"train_state": ocp.Checkpointer(handler)},
            options=ocp.CheckpointManagerOptions(max_to_keep=3)
        )
        
        checkpoint_managers['latest'] = ocp.CheckpointManager(
            directory=str((ckpt_root / "latest").resolve()),
            checkpointers={"train_state": ocp.Checkpointer(handler)},
            options=ocp.CheckpointManagerOptions(max_to_keep=1)
        )
        
        logger.info("✅ Checkpoint managers initialized")
        
    except Exception as e:
        logger.warning(f"Orbax managers unavailable: {e}")
    
    return checkpoint_managers


def _execute_training_loop(trainer, config, signals, labels, training_dir, checkpoint_managers):
    """Execute the main training loop."""
    epoch_results = []
    
    for epoch in range(config.num_epochs):
        logger.info(f"🔥 Epoch {epoch+1}/{config.num_epochs}")
        
        # Calculate batches
        num_samples = len(signals)
        full_batches = (num_samples + config.batch_size - 1) // config.batch_size
        num_batches = min(full_batches, 120)  # Cap for quick feedback
        
        epoch_losses = []
        epoch_accuracies = []
        
        # Training batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * config.batch_size
            end_idx = min(start_idx + config.batch_size, num_samples)
            
            batch_signals = signals[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            batch = (batch_signals, batch_labels)
            
            # Execute training step
            trainer.train_state, metrics = trainer.train_step(trainer.train_state, batch)
            
            epoch_losses.append(metrics.loss)
            epoch_accuracies.append(metrics.accuracy)
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"      Step {batch_idx+1}/{num_batches} "
                           f"loss={metrics.loss:.4f} acc={metrics.accuracy:.3f}")
        
        # Epoch summary
        import numpy as np
        avg_loss = float(np.mean(np.array(epoch_losses)))
        avg_accuracy = float(np.mean(np.array(epoch_accuracies)))
        
        logger.info(f"      Epoch Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        
        # Save checkpoint
        _save_checkpoint(checkpoint_managers, trainer, epoch, avg_loss, avg_accuracy, training_dir)
        
        epoch_results.append({
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': avg_accuracy
        })
    
    # Return training results
    final_epoch = epoch_results[-1] if epoch_results else {'loss': 0.0, 'accuracy': 0.0}
    
    return {
        'final_loss': final_epoch['loss'],
        'accuracy': final_epoch['accuracy'],
        'epochs_completed': config.num_epochs,
        'epoch_results': epoch_results
    }


def _save_checkpoint(checkpoint_managers, trainer, epoch, loss, accuracy, training_dir):
    """Save training checkpoint."""
    try:
        # Save latest checkpoint
        if checkpoint_managers['latest'] is not None:
            checkpoint_managers['latest'].save(
                epoch + 1,
                {'train_state': trainer.train_state},
                metrics={'epoch': epoch+1, 'loss': loss, 'accuracy': accuracy}
            )
        
        # Save best checkpoint logic here (would compare with previous best)
        # For now, save every 5 epochs as "best"
        if (epoch + 1) % 5 == 0 and checkpoint_managers['best'] is not None:
            checkpoint_managers['best'].save(
                epoch + 1,
                {'train_state': trainer.train_state},
                metrics={'epoch': epoch+1, 'loss': loss, 'accuracy': accuracy}
            )
            logger.info(f"      💾 Saved checkpoint at epoch {epoch+1}")
        
    except Exception as e:
        logger.warning(f"Checkpoint save failed: {e}")


def _run_final_evaluation(trainer, test_signals, test_labels, train_signals, training_results):
    """Run final evaluation on test set."""
    logger.info("🔧 Running final evaluation...")
    
    try:
        from training.test_evaluation import evaluate_on_test_set, create_test_evaluation_summary
        
        test_results = evaluate_on_test_set(
            trainer.train_state,
            test_signals,
            test_labels,
            train_signals=train_signals,
            verbose=True
        )
        
        # Create evaluation summary
        test_summary = create_test_evaluation_summary(
            train_accuracy=training_results['accuracy'],
            test_results=test_results,
            data_source="Real LIGO GW150914",
            num_epochs=training_results['epochs_completed']
        )
        
        logger.info(test_summary)
        
        # Comprehensive results
        return {
            'success': True,
            'metrics': {
                'final_train_loss': training_results['final_loss'],
                'final_train_accuracy': training_results['accuracy'],
                'final_test_accuracy': test_results['test_accuracy'],
                'total_epochs': training_results['epochs_completed'],
                'has_proper_test_set': test_results['has_proper_test_set'],
                'model_collapse': test_results.get('model_collapse', False),
                'test_analysis': test_results
            },
            'training_time': training_results.get('training_time', 0.0)
        }
        
    except Exception as e:
        logger.error(f"❌ Final evaluation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'partial_metrics': training_results
        }


# Export standard training functions
__all__ = [
    "run_standard_training"
]
