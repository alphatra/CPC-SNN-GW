"""
Unified Trainer: Simplified Multi-Stage Training

Streamlined implementation of CPC+SNN multi-stage training:
- Stage 1: CPC pretraining (self-supervised)
- Stage 2: SNN training (frozen CPC)
- Stage 3: Joint fine-tuning
- Optimized for production use with minimal complexity
"""

import time
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

# Import base trainer and utilities
from .base_trainer import TrainerBase, TrainingConfig
from .training_utils import ProgressTracker, format_training_time
from .training_metrics import create_training_metrics

# Import models
from ..models.cpc_encoder import CPCEncoder, enhanced_info_nce_loss  
from ..models.snn_classifier import SNNClassifier
from ..models.spike_bridge import SpikeBridge

logger = logging.getLogger(__name__)


@dataclass
class UnifiedTrainingConfig(TrainingConfig):
    """Configuration for unified multi-stage training."""
    
    # Multi-stage training
    cpc_epochs: int = 50
    snn_epochs: int = 30  
    joint_epochs: int = 20
    
    # Model architecture
    cpc_latent_dim: int = 256
    snn_hidden_size: int = 128
    num_classes: int = 3
    
    # Data parameters
    sequence_length: int = 1024
    
    # Loss weights for joint training
    cpc_loss_weight: float = 1.0
    snn_loss_weight: float = 1.0


class UnifiedTrainer(TrainerBase):
    """
    Unified trainer for multi-stage CPC+SNN training.
    
    Implements progressive training strategy:
    1. CPC pretraining for representation learning
    2. SNN training with frozen CPC encoder  
    3. Joint fine-tuning of full pipeline
    """
    
    def __init__(self, config: UnifiedTrainingConfig):
        super().__init__(config)
        self.config: UnifiedTrainingConfig = config
        
        # Stage tracking
        self.current_stage = 1
        self.stage_start_time = None
        
        # Model components
        self.cpc_encoder = None
        self.snn_classifier = None  
        self.spike_bridge = None
        
        logger.info("Initialized UnifiedTrainer for multi-stage training")
    
    def create_model(self):
        """Create individual model components."""
        self.cpc_encoder = CPCEncoder(latent_dim=self.config.cpc_latent_dim)
        self.spike_bridge = SpikeBridge()
        self.snn_classifier = SNNClassifier(
            hidden_size=self.config.snn_hidden_size,
            num_classes=self.config.num_classes
        )
        
        logger.info("Created model components: CPC, SpikeBridge, SNN")
    
    def create_train_state(self, model, sample_input):
        """Create training state for current stage."""
        key = jax.random.PRNGKey(42)
        
        if self.current_stage == 1:
            # CPC pretraining - only CPC encoder
            params = self.cpc_encoder.init(key, sample_input)
            apply_fn = self.cpc_encoder.apply
        elif self.current_stage == 2:
            # SNN training - spike bridge + SNN classifier
            latent_input = jnp.ones((sample_input.shape[0], sample_input.shape[1] // 16, self.config.cpc_latent_dim))
            spike_params = self.spike_bridge.init(key, latent_input, key)
            snn_input = jnp.ones((sample_input.shape[0], 50, self.config.cpc_latent_dim))  # dummy spike input
            snn_params = self.snn_classifier.init(key, snn_input)
            
            params = {'spike_bridge': spike_params, 'snn': snn_params}
            apply_fn = self._snn_apply_fn
        else:
            # Joint training - full pipeline
            cpc_params = self.cpc_encoder.init(key, sample_input)
            latent_input = jnp.ones((sample_input.shape[0], sample_input.shape[1] // 16, self.config.cpc_latent_dim))
            spike_params = self.spike_bridge.init(key, latent_input, key)
            snn_input = jnp.ones((sample_input.shape[0], 50, self.config.cpc_latent_dim))
            snn_params = self.snn_classifier.init(key, snn_input)
            
            params = {'cpc': cpc_params, 'spike_bridge': spike_params, 'snn': snn_params}
            apply_fn = self._joint_apply_fn
        
        optimizer = self.create_optimizer()
        
        return train_state.TrainState.create(
            apply_fn=apply_fn,
            params=params,
            tx=optimizer
        )
    
    def _snn_apply_fn(self, params, x_latent, key):
        """Apply function for SNN training stage."""
        spikes = self.spike_bridge.apply(params['spike_bridge'], x_latent, key)
        logits = self.snn_classifier.apply(params['snn'], spikes)
        return logits
    
    def _joint_apply_fn(self, params, x, key):
        """Apply function for joint training stage."""
        latents = self.cpc_encoder.apply(params['cpc'], x)
        spikes = self.spike_bridge.apply(params['spike_bridge'], latents, key)
        logits = self.snn_classifier.apply(params['snn'], spikes)
        return logits, latents
    
    def train_step(self, train_state, batch):
        """Training step for current stage."""
        if self.current_stage == 1:
            return self._cpc_train_step(train_state, batch)
        elif self.current_stage == 2:
            return self._snn_train_step(train_state, batch)
        else:
            return self._joint_train_step(train_state, batch)
    
    def _cpc_train_step(self, train_state, batch):
        """CPC pretraining step with InfoNCE loss."""
        x, _ = batch  # Ignore labels for self-supervised learning
        
        def loss_fn(params):
            # Forward pass through CPC encoder
            latents = train_state.apply_fn(params, x)
            
            # InfoNCE contrastive loss
            loss = enhanced_info_nce_loss(
                latents[:, :-1],  # context
                latents[:, 1:],   # targets
                temperature=0.1
            )
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        
        metrics = create_training_metrics(
            step=train_state.step,
            epoch=0,
            loss=float(loss),
            cpc_loss=float(loss)
        )
        
        return train_state, metrics
    
    def _snn_train_step(self, train_state, batch):
        """SNN training step with frozen CPC features."""
        x, y = batch
        
        def loss_fn(params):
            # Get frozen CPC features (no gradients)
            latents = jax.lax.stop_gradient(
                self.cpc_encoder.apply(self.frozen_cpc_params, x)
            )
            
            # Generate spike train and classify
            key = jax.random.PRNGKey(int(time.time()))
            logits = train_state.apply_fn(params, latents, key)
            
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
            return loss, accuracy
        
        (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        
        metrics = create_training_metrics(
            step=train_state.step,
            epoch=0,
            loss=float(loss),
            accuracy=float(accuracy),
            snn_loss=float(loss)
        )
        
        return train_state, metrics
    
    def _joint_train_step(self, train_state, batch):
        """Joint training step with both CPC and classification losses."""
        x, y = batch
        
        def loss_fn(params):
            key = jax.random.PRNGKey(int(time.time()))
            logits, latents = train_state.apply_fn(params, x, key)
            
            # Classification loss
            clf_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            
            # CPC contrastive loss
            cpc_loss = enhanced_info_nce_loss(
                latents[:, :-1],
                latents[:, 1:],
                temperature=0.1
            )
            
            # Combined loss
            total_loss = (self.config.snn_loss_weight * clf_loss + 
                         self.config.cpc_loss_weight * cpc_loss)
            
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
            return total_loss, (clf_loss, cpc_loss, accuracy)
        
        (total_loss, (clf_loss, cpc_loss, accuracy)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(train_state.params)
        
        train_state = train_state.apply_gradients(grads=grads)
        
        metrics = create_training_metrics(
            step=train_state.step,
            epoch=0,
            loss=float(total_loss),
            accuracy=float(accuracy),
            cpc_loss=float(cpc_loss),
            snn_loss=float(clf_loss)
        )
        
        return train_state, metrics
    
    def eval_step(self, train_state, batch):
        """Evaluation step for current stage."""
        x, y = batch
        
        if self.current_stage == 1:
            # CPC evaluation - use reconstruction quality
            latents = train_state.apply_fn(train_state.params, x)
            loss = enhanced_info_nce_loss(latents[:, :-1], latents[:, 1:])
            
            metrics = create_training_metrics(
                step=train_state.step,
                epoch=0,
                loss=float(loss)
            )
        else:
            # Classification evaluation
            if self.current_stage == 2:
                latents = jax.lax.stop_gradient(
                    self.cpc_encoder.apply(self.frozen_cpc_params, x)
                )
                key = jax.random.PRNGKey(42)
                logits = train_state.apply_fn(train_state.params, latents, key)
            else:
                key = jax.random.PRNGKey(42)
                logits, _ = train_state.apply_fn(train_state.params, x, key)
            
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
            
            metrics = create_training_metrics(
                step=train_state.step,
                epoch=0,
                loss=float(loss),
                accuracy=float(accuracy)
            )
        
        return metrics
    
    def train_stage(self, stage: int, dataloader, num_epochs: int) -> Dict[str, Any]:
        """Train single stage."""
        self.current_stage = stage
        self.stage_start_time = time.time()
        
        stage_names = {1: "CPC Pretraining", 2: "SNN Training", 3: "Joint Fine-tuning"}
        logger.info(f"Starting Stage {stage}: {stage_names[stage]} ({num_epochs} epochs)")
        
        # Create model if needed
        if not self.cpc_encoder:
            self.create_model()
        
        # Initialize training state
        sample_batch = next(iter(dataloader))
        sample_input = sample_batch[0]
        self.train_state = self.create_train_state(None, sample_input)
        
        # Progress tracking
        total_steps = num_epochs * len(list(dataloader))  # Estimate
        progress = ProgressTracker(total_steps, log_interval=50)
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_metrics = []
            
            for step, batch in enumerate(dataloader):
                # Training step
                self.train_state, metrics = self.train_step(self.train_state, batch)
                epoch_metrics.append(metrics)
                
                # Log metrics
                if step % 50 == 0:
                    self.validate_and_log_step(metrics, f"stage_{stage}_train")
                
                # Update progress
                progress.update(epoch * 100 + step, metrics.to_dict())
            
            # Epoch summary
            avg_loss = sum(m.loss for m in epoch_metrics) / len(epoch_metrics)
            logger.info(f"Stage {stage} Epoch {epoch+1}/{num_epochs}: avg_loss={avg_loss:.4f}")
        
        # Save stage results
        stage_time = time.time() - self.stage_start_time
        stage_results = {
            'stage': stage,
            'stage_name': stage_names[stage],
            'num_epochs': num_epochs,
            'final_loss': avg_loss,
            'training_time': stage_time,
            'params': self.train_state.params
        }
        
        # Store CPC params for stage 2
        if stage == 1:
            self.frozen_cpc_params = self.train_state.params
        
        logger.info(f"Stage {stage} completed in {format_training_time(0, stage_time)}")
        return stage_results
    
    def train_unified_pipeline(self, train_dataloader, val_dataloader=None) -> Dict[str, Any]:
        """Execute complete multi-stage training pipeline."""
        logger.info("Starting unified multi-stage training pipeline")
        
        results = {}
        
        # Stage 1: CPC Pretraining
        results['stage_1'] = self.train_stage(1, train_dataloader, self.config.cpc_epochs)
        
        # Stage 2: SNN Training
        results['stage_2'] = self.train_stage(2, train_dataloader, self.config.snn_epochs)
        
        # Stage 3: Joint Fine-tuning
        results['stage_3'] = self.train_stage(3, train_dataloader, self.config.joint_epochs)
        
        # Final evaluation
        if val_dataloader:
            logger.info("Running final evaluation...")
            final_metrics = []
            for batch in val_dataloader:
                metrics = self.eval_step(self.train_state, batch)
                final_metrics.append(metrics)
            
            avg_accuracy = sum(m.accuracy for m in final_metrics if m.accuracy) / len(final_metrics)
            results['final_accuracy'] = avg_accuracy
            logger.info(f"Final validation accuracy: {avg_accuracy:.4f}")
        
        # Training summary
        total_time = sum(r['training_time'] for r in results.values() if 'training_time' in r)
        results['total_training_time'] = total_time
        
        logger.info(f"Unified training completed in {format_training_time(0, total_time)}")
        return results


def create_unified_trainer(config: Optional[UnifiedTrainingConfig] = None) -> UnifiedTrainer:
    """Factory function to create unified trainer."""
    if config is None:
        config = UnifiedTrainingConfig()
    
    return UnifiedTrainer(config) 