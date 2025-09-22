"""
Training stages for unified CPC+SNN pipeline.

This module contains the core training step implementations for each stage
of the multi-stage training process:
- Stage 1: CPC pretraining (self-supervised)
- Stage 2: SNN training (with optional CPC fine-tuning)  
- Stage 3: Joint fine-tuning

Split from unified_trainer.py for better modularity and maintainability.
"""

import time
import logging
from typing import Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

# Import models and utilities
from models.cpc.losses import enhanced_info_nce_loss
from .monitoring.core import create_training_metrics
from .utils import ProgressTracker, format_training_time

logger = logging.getLogger(__name__)


def _cpc_train_step(trainer, train_state, batch):
    """CPC pretraining step with InfoNCE loss."""
    x, _ = batch  # Ignore labels for self-supervised learning
    
    def loss_fn(params):
        # ✅ FIXED: Deterministic key for reproducibility
        latents = trainer.cpc_encoder.apply(params, x, rngs={'dropout': trainer._get_deterministic_key("cpc_train")})
        
        # ✅ FIXED: Working InfoNCE loss (not zero!)
        loss = enhanced_info_nce_loss(
            latents[:, :-1],  # context
            latents[:, 1:],   # targets
            temperature=getattr(trainer.config, 'cpc_temperature', 0.1)
        )
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    
    # ✅ FIXED: Real epoch tracking (not hardcoded epoch=0)
    metrics = create_training_metrics(
        step=train_state.step,
        epoch=trainer.current_epoch,
        loss=float(loss),
        cpc_loss=float(loss)
    )
    
    return train_state, metrics


def _snn_train_step(trainer, train_state, batch):
    """SNN training step with optional CPC fine-tuning."""
    x, y = batch
    
    def loss_fn(params):
        if trainer.config.enable_cpc_finetuning_stage2:
            # ✅ FIXED: Removed stop_gradient to allow CPC fine-tuning
            logits, latents = trainer._snn_with_cpc_apply_fn(params, x, training=True)
            clf_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            cpc_reg = enhanced_info_nce_loss(
                latents[:, :-1],
                latents[:, 1:],
                temperature=getattr(trainer.config, 'cpc_temperature', 0.1)
            )
            total_loss = clf_loss + trainer.config.cpc_loss_weight * cpc_reg
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
            return total_loss, (clf_loss, cpc_reg, accuracy)
        else:
            # Frozen CPC: use stored params from Stage 1
            latents = trainer.cpc_encoder.apply(trainer.stage1_cpc_params, x)
            logits = trainer._snn_frozen_apply_fn(params, latents, training=True)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
            return loss, accuracy
    
    if trainer.config.enable_cpc_finetuning_stage2:
        (total_loss, (clf_loss, cpc_reg, accuracy)), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        
        metrics = create_training_metrics(
            step=train_state.step,
            epoch=trainer.current_epoch,
            loss=float(total_loss),
            accuracy=float(accuracy),
            snn_loss=float(clf_loss),
            cpc_loss=float(cpc_reg)
        )
    else:
        (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        
        metrics = create_training_metrics(
            step=train_state.step,
            epoch=trainer.current_epoch,
            loss=float(loss),
            accuracy=float(accuracy),
            snn_loss=float(loss)
        )
    
    return train_state, metrics


def _joint_train_step(trainer, train_state, batch):
    """Joint training step with both CPC and classification losses."""
    x, y = batch
    
    def loss_fn(params):
        logits, latents = trainer._joint_apply_fn(params, x, training=True)
        
        clf_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        cpc_loss = enhanced_info_nce_loss(
            latents[:, :-1],
            latents[:, 1:],
            temperature=getattr(trainer.config, 'cpc_temperature', 0.1)
        )
        
        total_loss = (trainer.config.snn_loss_weight * clf_loss + 
                     trainer.config.cpc_loss_weight * cpc_loss)
        
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        return total_loss, (clf_loss, cpc_loss, accuracy)
    
    (total_loss, (clf_loss, cpc_loss, accuracy)), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    
    metrics = create_training_metrics(
        step=train_state.step,
        epoch=trainer.current_epoch,
        loss=float(total_loss),
        accuracy=float(accuracy),
        cpc_loss=float(cpc_loss),
        snn_loss=float(clf_loss)
    )
    
    return train_state, metrics


def train_stage(trainer, stage: int, dataloader, num_epochs: int) -> Dict[str, Any]:
    """Train single stage with real epoch tracking."""
    trainer.current_stage = stage
    stage_start_time = time.time()
    
    stage_names = {1: "CPC Pretraining", 2: "SNN Training", 3: "Joint Fine-tuning"}
    logger.info(f"Starting Stage {stage}: {stage_names[stage]} ({num_epochs} epochs)")
    
    # Create model if needed
    if not trainer.cpc_encoder:
        trainer.create_model()
    
    # Initialize training state
    sample_batch = next(iter(dataloader))
    sample_input = sample_batch[0]
    trainer.train_state = trainer.create_train_state(None, sample_input)
    
    # Progress tracking
    total_steps = num_epochs * len(list(dataloader))  # Estimate
    progress = ProgressTracker(total_steps, log_interval=50)
    
    # ✅ FIXED: Real epoch tracking
    for epoch in range(num_epochs):
        trainer.current_epoch = epoch  # ✅ Real epoch counter
        epoch_metrics = []
        
        for step, batch in enumerate(dataloader):
            # ✅ FIXED: Training step routing by stage
            if stage == 1:
                trainer.train_state, metrics = _cpc_train_step(trainer, trainer.train_state, batch)
            elif stage == 2:  
                trainer.train_state, metrics = _snn_train_step(trainer, trainer.train_state, batch)
            elif stage == 3:
                trainer.train_state, metrics = _joint_train_step(trainer, trainer.train_state, batch)
            else:
                raise ValueError(f"Invalid training stage: {stage}")
            
            epoch_metrics.append(metrics)
            
            # Log metrics
            if step % 50 == 0:
                trainer.validate_and_log_step(metrics, f"stage_{stage}_train")
            
            # Update progress
            progress.update(epoch * 100 + step, metrics.to_dict())
        
        # Epoch summary
        avg_loss = sum(m.loss for m in epoch_metrics) / len(epoch_metrics)
        logger.info(f"Stage {stage} Epoch {epoch+1}/{num_epochs}: avg_loss={avg_loss:.4f}")
    
    # Save stage results
    stage_time = time.time() - stage_start_time
    stage_results = {
        'stage': stage,
        'stage_name': stage_names[stage],
        'num_epochs': num_epochs,
        'final_loss': avg_loss,
        'training_time': stage_time,
        'params': trainer.train_state.params
    }
    
    # ✅ Store CPC params for Stage 2 (if needed)
    if stage == 1:
        trainer.stage1_cpc_params = trainer.train_state.params
    
    logger.info(f"Stage {stage} completed in {format_training_time(0, stage_time)}")
    return stage_results


# Export functions for use by UnifiedTrainer
__all__ = [
    "_cpc_train_step",
    "_snn_train_step", 
    "_joint_train_step",
    "train_stage"
]

