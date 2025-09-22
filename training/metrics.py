"""
Training metrics computation and evaluation for CPC+SNN unified training.

This module contains metric computation logic extracted from unified_trainer.py
for better modularity. Includes evaluation steps and metric aggregation.

Split from unified_trainer.py for better maintainability.
"""

import logging
from typing import Dict, Any, Optional

import jax
import jax.numpy as jnp
import optax

# Import models and utilities
from models.cpc.losses import enhanced_info_nce_loss
from .monitoring.core import create_training_metrics

logger = logging.getLogger(__name__)


def eval_step(trainer, train_state, batch):
    """
    Evaluation step for all training stages.
    
    ✅ FIXED: Real evaluation (not mock values)
    Computes appropriate metrics based on current training stage.
    
    Args:
        trainer: UnifiedTrainer instance
        train_state: Current training state
        batch: Evaluation batch (x, y)
        
    Returns:
        TrainingMetrics with evaluation results
    """
    x, y = batch
    
    # Route evaluation by current stage
    if trainer.current_stage == 1:
        # CPC evaluation (self-supervised)
        latents = trainer.cpc_encoder.apply(
            train_state.params, 
            x,
            rngs={'dropout': trainer._get_deterministic_key("cpc_eval")}
        )
        
        # InfoNCE loss for CPC
        cpc_loss = enhanced_info_nce_loss(
            latents[:, :-1], 
            latents[:, 1:], 
            temperature=getattr(trainer.config, 'cpc_temperature', 0.1)
        )
        
        return create_training_metrics(
            step=train_state.step,
            epoch=trainer.current_epoch,
            loss=float(cpc_loss),
            cpc_loss=float(cpc_loss)
        )
    
    elif trainer.current_stage == 2:
        # SNN evaluation
        if trainer.config.enable_cpc_finetuning_stage2:
            # Joint CPC+SNN evaluation
            logits, latents = trainer._snn_with_cpc_apply_fn(train_state.params, x, training=False)
            clf_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            cpc_loss = enhanced_info_nce_loss(
                latents[:, :-1],
                latents[:, 1:],
                temperature=getattr(trainer.config, 'cpc_temperature', 0.1)
            )
            total_loss = clf_loss + trainer.config.cpc_loss_weight * cpc_loss
        else:
            # Frozen CPC evaluation
            latents = trainer.cpc_encoder.apply(trainer.stage1_cpc_params, x)
            logits = trainer._snn_frozen_apply_fn(train_state.params, latents, training=False)
            clf_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            total_loss = clf_loss
            cpc_loss = 0.0
        
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        
        return create_training_metrics(
            step=train_state.step,
            epoch=trainer.current_epoch,
            loss=float(total_loss),
            accuracy=float(accuracy),
            snn_loss=float(clf_loss),
            cpc_loss=float(cpc_loss)
        )
    
    elif trainer.current_stage == 3:
        # Joint evaluation (CPC + SNN)
        logits, latents = trainer._joint_apply_fn(train_state.params, x, training=False)
        
        clf_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        cpc_loss = enhanced_info_nce_loss(
            latents[:, :-1],
            latents[:, 1:],
            temperature=getattr(trainer.config, 'cpc_temperature', 0.1)
        )
        
        total_loss = (trainer.config.snn_loss_weight * clf_loss + 
                     trainer.config.cpc_loss_weight * cpc_loss)
        
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        
        return create_training_metrics(
            step=train_state.step,
            epoch=trainer.current_epoch,
            loss=float(total_loss),
            accuracy=float(accuracy),
            cpc_loss=float(cpc_loss),
            snn_loss=float(clf_loss)
        )
    
    else:
        raise ValueError(f"Invalid evaluation stage: {trainer.current_stage}")


def compute_comprehensive_metrics(trainer, dataloader, stage_name: str = "evaluation") -> Dict[str, Any]:
    """
    Compute comprehensive metrics across entire dataset.
    
    Args:
        trainer: UnifiedTrainer instance
        dataloader: Data loader for evaluation
        stage_name: Name of evaluation stage for logging
        
    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    if not trainer.train_state:
        logger.warning(f"No trained model available for {stage_name}")
        return {'error': 'no_model'}
    
    logger.info(f"Computing comprehensive metrics for {stage_name}...")
    
    # Accumulate metrics
    total_loss = 0.0
    total_accuracy = 0.0
    total_cpc_loss = 0.0
    total_snn_loss = 0.0
    total_samples = 0
    
    all_predictions = []
    all_labels = []
    
    # Evaluation loop
    for batch in dataloader:
        x, y = batch
        batch_size = x.shape[0]
        
        # Get batch metrics
        batch_metrics = eval_step(trainer, trainer.train_state, batch)
        
        # Accumulate weighted metrics
        total_loss += batch_metrics.loss * batch_size
        total_accuracy += batch_metrics.accuracy * batch_size
        total_cpc_loss += getattr(batch_metrics, 'cpc_loss', 0.0) * batch_size
        total_snn_loss += getattr(batch_metrics, 'snn_loss', 0.0) * batch_size
        total_samples += batch_size
        
        # Collect predictions for additional metrics
        if trainer.current_stage in [2, 3]:  # Classification stages
            if trainer.current_stage == 2:
                if trainer.config.enable_cpc_finetuning_stage2:
                    logits, _ = trainer._snn_with_cpc_apply_fn(trainer.train_state.params, x, training=False)
                else:
                    latents = trainer.cpc_encoder.apply(trainer.stage1_cpc_params, x)
                    logits = trainer._snn_frozen_apply_fn(trainer.train_state.params, latents, training=False)
            else:  # stage == 3
                logits, _ = trainer._joint_apply_fn(trainer.train_state.params, x, training=False)
            
            predictions = jnp.argmax(logits, axis=-1)
            all_predictions.append(predictions)
            all_labels.append(y)
    
    # Compute averages
    avg_metrics = {}
    if total_samples > 0:
        avg_metrics['loss'] = float(total_loss / total_samples)
        avg_metrics['accuracy'] = float(total_accuracy / total_samples)
        avg_metrics['cpc_loss'] = float(total_cpc_loss / total_samples)
        avg_metrics['snn_loss'] = float(total_snn_loss / total_samples)
        avg_metrics['num_samples'] = total_samples
    
    # Additional classification metrics (for stages 2 and 3)
    if all_predictions and trainer.current_stage in [2, 3]:
        all_preds = jnp.concatenate(all_predictions)
        all_true = jnp.concatenate(all_labels)
        
        # Class-wise accuracy
        unique_classes = jnp.unique(all_true)
        class_accuracies = {}
        for cls in unique_classes:
            cls_mask = all_true == cls
            if jnp.sum(cls_mask) > 0:
                cls_acc = jnp.mean(all_preds[cls_mask] == all_true[cls_mask])
                class_accuracies[f'class_{int(cls)}_accuracy'] = float(cls_acc)
        
        avg_metrics.update(class_accuracies)
        
        # Confusion matrix elements (for binary classification)
        if len(unique_classes) == 2:
            tp = jnp.sum((all_preds == 1) & (all_true == 1))
            tn = jnp.sum((all_preds == 0) & (all_true == 0))
            fp = jnp.sum((all_preds == 1) & (all_true == 0))
            fn = jnp.sum((all_preds == 0) & (all_true == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            avg_metrics.update({
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn)
            })
    
    # Add metadata
    avg_metrics.update({
        'stage': trainer.current_stage,
        'epoch': trainer.current_epoch,
        'stage_name': stage_name
    })
    
    logger.info(f"{stage_name} metrics: loss={avg_metrics.get('loss', 0):.4f}, "
               f"accuracy={avg_metrics.get('accuracy', 0):.4f}")
    
    return avg_metrics


def validate_and_log_step(trainer, metrics, step_type: str = "train"):
    """
    Validate and log step metrics with quality checks.
    
    Args:
        trainer: UnifiedTrainer instance  
        metrics: TrainingMetrics from step
        step_type: Type of step ("train", "eval", etc.)
    """
    # Basic validation
    if metrics.loss < 0:
        logger.warning(f"Negative loss detected in {step_type}: {metrics.loss}")
    
    if hasattr(metrics, 'accuracy') and (metrics.accuracy < 0 or metrics.accuracy > 1):
        logger.warning(f"Invalid accuracy in {step_type}: {metrics.accuracy}")
    
    # Log metrics based on stage and verbosity
    if trainer.current_stage == 1:  # CPC stage
        logger.debug(f"{step_type} | CPC Loss: {metrics.loss:.6f}")
    elif trainer.current_stage in [2, 3]:  # Classification stages
        logger.debug(f"{step_type} | Loss: {metrics.loss:.4f}, "
                    f"Accuracy: {getattr(metrics, 'accuracy', 0):.4f}")
    
    # Check for suspicious patterns
    if hasattr(metrics, 'accuracy'):
        if metrics.accuracy == 0.5:  # Potential random guessing
            logger.debug(f"Random-level accuracy detected in {step_type}: {metrics.accuracy}")
        elif metrics.accuracy > 0.95:  # Suspiciously high
            logger.debug(f"High accuracy in {step_type}: {metrics.accuracy}")


# Export functions for use by UnifiedTrainer
__all__ = [
    "eval_step",
    "compute_comprehensive_metrics", 
    "validate_and_log_step"
]

