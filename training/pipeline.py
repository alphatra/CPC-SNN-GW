"""
Training pipeline orchestration for CPC+SNN unified training.

This module contains the high-level training pipeline logic that coordinates
the multi-stage training process and manages the flow between stages.

Split from unified_trainer.py for better modularity and maintainability.
"""

import time
import logging
from typing import Dict, Any, Optional

import jax

# Import stage functions
from .stages import train_stage
from .training_utils import format_training_time

logger = logging.getLogger(__name__)


def train_unified_pipeline(trainer, train_dataloader, val_dataloader=None) -> Dict[str, Any]:
    """
    Execute complete unified training pipeline.
    
    Coordinates all three training stages:
    1. CPC pretraining (self-supervised)  
    2. SNN training (with optional CPC fine-tuning)
    3. Joint fine-tuning
    
    Args:
        trainer: UnifiedTrainer instance
        train_dataloader: Training data loader
        val_dataloader: Optional validation data loader
        
    Returns:
        Dictionary containing complete training results
    """
    pipeline_start_time = time.time()
    
    logger.info("🚀 Starting Unified CPC+SNN Training Pipeline")
    logger.info(f"Configuration: {trainer.config}")
    
    all_results = {
        'stage_results': {},
        'validation_results': {},
        'pipeline_metadata': {
            'total_stages': 3,
            'config': trainer.config.__dict__,
        }
    }
    
    try:
        # ===== STAGE 1: CPC PRETRAINING =====
        logger.info("=" * 60)
        logger.info("STAGE 1: CPC Pretraining (Self-Supervised)")
        logger.info("=" * 60)
        
        stage1_results = train_stage(
            trainer=trainer,
            stage=1,
            dataloader=train_dataloader,
            num_epochs=trainer.config.cpc_epochs
        )
        all_results['stage_results']['stage_1'] = stage1_results
        
        # Validation after Stage 1 (if available)
        if val_dataloader:
            val_metrics_1 = _compute_real_evaluation_metrics(trainer, val_dataloader)
            all_results['validation_results']['stage_1'] = val_metrics_1
            logger.info(f"Stage 1 Validation - Loss: {val_metrics_1.get('loss', 'N/A'):.4f}")
        
        # ===== STAGE 2: SNN TRAINING =====
        logger.info("=" * 60)
        logger.info("STAGE 2: SNN Training (Classification)")
        logger.info("=" * 60)
        
        stage2_results = train_stage(
            trainer=trainer,
            stage=2,
            dataloader=train_dataloader,
            num_epochs=trainer.config.snn_epochs
        )
        all_results['stage_results']['stage_2'] = stage2_results
        
        # Validation after Stage 2 (if available)
        if val_dataloader:
            val_metrics_2 = _compute_real_evaluation_metrics(trainer, val_dataloader)
            all_results['validation_results']['stage_2'] = val_metrics_2
            logger.info(f"Stage 2 Validation - Accuracy: {val_metrics_2.get('accuracy', 'N/A'):.4f}")
        
        # ===== STAGE 3: JOINT FINE-TUNING =====
        logger.info("=" * 60)
        logger.info("STAGE 3: Joint Fine-tuning (CPC + SNN)")
        logger.info("=" * 60)
        
        stage3_results = train_stage(
            trainer=trainer,
            stage=3,
            dataloader=train_dataloader,
            num_epochs=trainer.config.joint_epochs
        )
        all_results['stage_results']['stage_3'] = stage3_results
        
        # Final validation (if available)
        if val_dataloader:
            final_val_metrics = _compute_real_evaluation_metrics(trainer, val_dataloader)
            all_results['validation_results']['final'] = final_val_metrics
            logger.info(f"Final Validation - Accuracy: {final_val_metrics.get('accuracy', 'N/A'):.4f}")
        
        # ===== PIPELINE SUMMARY =====
        total_training_time = time.time() - pipeline_start_time
        all_results['pipeline_metadata']['total_training_time'] = total_training_time
        all_results['pipeline_metadata']['success'] = True
        
        logger.info("=" * 60)
        logger.info("🎉 UNIFIED PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Total training time: {format_training_time(0, total_training_time)}")
        logger.info("=" * 60)
        
        # Log stage summaries
        for stage_num, stage_data in all_results['stage_results'].items():
            stage_time = stage_data['training_time']
            stage_loss = stage_data['final_loss']
            logger.info(f"{stage_data['stage_name']}: {format_training_time(0, stage_time)}, "
                       f"final_loss={stage_loss:.4f}")
        
        # ✅ Save final model parameters
        trainer.final_params = trainer.train_state.params
        
        return all_results
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        all_results['pipeline_metadata']['success'] = False
        all_results['pipeline_metadata']['error'] = str(e)
        all_results['pipeline_metadata']['total_training_time'] = time.time() - pipeline_start_time
        raise


def _compute_real_evaluation_metrics(trainer, val_dataloader) -> Dict[str, Any]:
    """
    Compute real evaluation metrics on validation set.
    
    ✅ FIXED: Real metrics computation (not mock values)
    
    Args:
        trainer: UnifiedTrainer instance
        val_dataloader: Validation data loader
        
    Returns:
        Dictionary containing validation metrics
    """
    if not trainer.train_state:
        logger.warning("No trained model available for evaluation")
        return {'error': 'no_model'}
    
    logger.info("Computing validation metrics...")
    eval_start_time = time.time()
    
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0
    
    # Evaluation loop
    for batch in val_dataloader:
        x, y = batch
        batch_size = x.shape[0]
        
        # Forward pass (deterministic)
        eval_metrics = trainer.eval_step(trainer.train_state, batch)
        
        # Accumulate metrics
        total_loss += eval_metrics.loss * batch_size
        total_accuracy += eval_metrics.accuracy * batch_size
        total_samples += batch_size
    
    # ✅ FIXED: Real average computation
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0.0
    
    eval_time = time.time() - eval_start_time
    
    results = {
        'loss': float(avg_loss),
        'accuracy': float(avg_accuracy),
        'num_samples': total_samples,
        'eval_time': eval_time,
        'stage': trainer.current_stage,
        'epoch': trainer.current_epoch
    }
    
    logger.info(f"Validation complete: loss={avg_loss:.4f}, accuracy={avg_accuracy:.4f} "
               f"({total_samples} samples, {eval_time:.2f}s)")
    
    return results


# Export functions for use by UnifiedTrainer
__all__ = [
    "train_unified_pipeline",
    "_compute_real_evaluation_metrics"
]

