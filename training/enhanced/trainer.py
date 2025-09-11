"""
Complete enhanced trainer implementation.

This module contains the CompleteEnhancedTrainer class extracted from
complete_enhanced_training.py for better modularity.

Split from complete_enhanced_training.py for better maintainability.
"""

import time
import logging
from typing import Dict, Any, Optional
import jax
import jax.numpy as jnp
import optax

from .config import CompleteEnhancedConfig, TrainStateWithBatchStats
from .model import CompleteEnhancedModel
from ..base_trainer import TrainerBase
from ..training_metrics import create_training_metrics

# Import enhanced models
from models.cpc_losses import (
    MomentumHardNegativeMiner,
    AdaptiveTemperatureController
)

logger = logging.getLogger(__name__)


class CompleteEnhancedTrainer(TrainerBase):
    """
    Complete enhanced trainer using ALL 5 revolutionary improvements.
    
    This trainer demonstrates the full potential of our neuromorphic system
    with all enhancements working together synergistically.
    """
    
    def __init__(self, config: CompleteEnhancedConfig):
        super().__init__(config)
        self.config: CompleteEnhancedConfig = config
        
        # 🚀 Enhancement 5: Initialize Momentum Hard Negative Miner
        if getattr(config, 'use_momentum_negatives', False):
            self.negative_miner = MomentumHardNegativeMiner(
                momentum=getattr(config, 'negative_momentum', 0.999),
                hard_negative_ratio=getattr(config, 'hard_negative_ratio', 0.3),
                memory_bank_size=2048
            )
        else:
            self.negative_miner = None
        
        # Training progress tracking for adaptive components
        self.training_progress = 0.0
        self.total_training_steps = 0
        
        # 🌡️ MATHEMATICAL FRAMEWORK: Adaptive Temperature Controller
        if getattr(config, 'use_adaptive_temperature', False):
            self.temp_controller = AdaptiveTemperatureController(
                initial_temperature=getattr(config, 'initial_temperature', 0.1),
                min_temperature=0.01,
                max_temperature=1.0,
                adaptation_rate=0.01,
                target_accuracy=0.7
            )
            logger.info(f"🌡️ Adaptive Temperature: τ_0={getattr(config, 'initial_temperature', 0.1):.3f}")
        else:
            self.temp_controller = None
        
        logger.info("🧮 MATHEMATICAL FRAMEWORK Enhanced Trainer initialized:")
        logger.info("🚀 Original 5 Enhancements:")
        logger.info(f"   1. Adaptive Surrogate: {getattr(config, 'surrogate_gradient_type', 'adaptive')}")
        logger.info(f"   2. Temporal Transformer: {getattr(config, 'use_temporal_transformer', True)}")
        logger.info(f"   3. Learnable Thresholds: {getattr(config, 'use_learnable_thresholds', True)}")
        logger.info(f"   4. Enhanced LIF: {getattr(config, 'use_enhanced_lif', True)}")
        logger.info(f"   5. Momentum InfoNCE: {getattr(config, 'use_momentum_negatives', True)}")
        
        logger.info("🧮 Mathematical Framework Enhancements:")
        logger.info(f"   📐 Temporal InfoNCE: {getattr(config, 'use_temporal_infonce', True)}")
        logger.info(f"   🌡️ Adaptive Temperature: {getattr(config, 'use_adaptive_temperature', False)}")
        logger.info(f"   🌊 Phase-Preserving: {getattr(config, 'use_phase_preserving_encoding', True)}")
    
    def create_model(self):
        """Create complete enhanced model with all improvements."""
        return CompleteEnhancedModel(config=self.config)
    
    def create_train_state(self, model, sample_input: jnp.ndarray) -> TrainStateWithBatchStats:
        """Create training state with model parameters and batch_stats."""
        key = jax.random.PRNGKey(42)
        
        # Initialize model parameters with mutable batch_stats
        logger.info("🔧 Initializing model parameters...")
        init_start_time = time.time()
        variables = model.init(key, sample_input, training=False)
        init_time = time.time() - init_start_time
        logger.info(f"✅ Model.init() completed in {init_time:.1f}s")
        
        params = variables['params']
        batch_stats = variables.get('batch_stats', {})
        
        # Create optimizer
        logger.info("🔧 Creating optimizer...")
        opt_start_time = time.time()
        optimizer = self.create_optimizer()
        opt_time = time.time() - opt_start_time
        logger.info(f"✅ Optimizer created in {opt_time:.1f}s")
        
        # Initialize optimizer state  
        logger.info("🔧 Initializing optimizer state...")
        opt_state_start_time = time.time()
        opt_state = optimizer.init(params)
        opt_state_time = time.time() - opt_state_start_time
        logger.info(f"✅ Optimizer state initialized in {opt_state_time:.1f}s")
        
        # Create custom train state with batch_stats
        return TrainStateWithBatchStats(
            step=0,
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
            opt_state=opt_state,
            batch_stats=batch_stats
        )
    
    def create_optimizer(self):
        """Create ENHANCED optimizer with configurable schedule and regularization."""
        
        # Calculate training steps if not set yet
        if self.total_training_steps == 0:
            # Estimate based on config - will be updated later
            estimated_steps_per_epoch = max(100, 1000 // max(1, self.config.batch_size))
            self.total_training_steps = max(1, self.config.num_epochs) * estimated_steps_per_epoch
        
        # Ensure total_training_steps is always positive for scheduler compatibility
        self.total_training_steps = max(1000, self.total_training_steps)
        
        # 🔧 ENHANCED LEARNING RATE SCHEDULING
        lr_schedule = getattr(self.config, 'learning_rate_schedule', 'cosine')
        
        if lr_schedule == "cosine_with_warmup":
            # Cosine decay with warmup
            warmup_epochs = getattr(self.config, 'warmup_epochs', 5)
            warmup_steps = warmup_epochs * (self.total_training_steps // max(1, self.config.num_epochs))
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.config.learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=self.total_training_steps,
                end_value=self.config.learning_rate * 0.01
            )
        elif lr_schedule == "cosine":
            # Standard cosine decay
            schedule = optax.cosine_decay_schedule(
                init_value=self.config.learning_rate,
                decay_steps=self.total_training_steps,
                alpha=0.01
            )
        elif lr_schedule == "exponential":
            schedule = optax.exponential_decay(
                init_value=self.config.learning_rate,
                transition_steps=self.total_training_steps // 4,
                decay_rate=0.8
            )
        else:  # constant
            schedule = self.config.learning_rate
        
        # 🔧 ENHANCED OPTIMIZER CHAIN
        optimizer_chain = []
        
        # Add gradient clipping if enabled
        if getattr(self.config, 'gradient_clipping', True):
            max_grad_norm = getattr(self.config, 'max_gradient_norm', 1.0)
            optimizer_chain.append(optax.clip_by_global_norm(max_grad_norm))
        
        # Add AdamW with weight decay
        optimizer_chain.append(
            optax.adamw(
                learning_rate=schedule, 
                weight_decay=self.config.weight_decay,
                b1=0.9,
                b2=0.999,
                eps=1e-8
            )
        )
        
        optimizer = optax.chain(*optimizer_chain)
        
        # 🔧 MIXED PRECISION ENHANCEMENT
        if getattr(self.config, 'use_mixed_precision', True):
            optimizer = optax.apply_if_finite(optimizer, max_consecutive_errors=3)
        
        return optimizer
    
    def enhanced_loss_fn(self, train_state, params, batch, rng_key):
        """
        Enhanced loss function using framework mathematical components.
        🧮 MATHEMATICAL FRAMEWORK: Temporal InfoNCE + Adaptive Temperature Control
        🚀 Enhancement 5: Superior contrastive learning
        """
        signals, labels = batch
        
        # Forward pass with all enhancements and mutable batch_stats
        output = train_state.apply_fn(
            {'params': params, 'batch_stats': train_state.batch_stats},
            signals,
            training=True,
            training_progress=self.training_progress,
            return_intermediates=True,
            rngs={'dropout': rng_key}
        )
        
        logits = output['logits']
        cpc_features = output['cpc_features']
        
        # 🔧 CLASSIFICATION LOSS: Standard cross-entropy
        classification_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        
        # 🧮 MATHEMATICAL FRAMEWORK: Temporal InfoNCE Loss (Equation 1)
        temporal_infonce_loss = 0.0
        if self.config.use_temporal_infonce and cpc_features is not None:
            from models.cpc_losses import temporal_info_nce_loss
            
            # Dynamic temperature from adaptive controller
            if self.temp_controller is not None:
                current_accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
                current_temp = self.temp_controller.update_temperature(
                    float(current_accuracy),
                    float(classification_loss),
                    int(train_state.step)
                )
            else:
                current_temp = getattr(self.config, 'initial_temperature', 0.1)
            
            temporal_infonce_loss = temporal_info_nce_loss(
                cpc_features,
                temperature=current_temp,
                max_prediction_steps=self.config.max_prediction_steps
            )
        
        # 🚀 Enhancement 5: Momentum InfoNCE with Hard Negatives
        momentum_loss = 0.0
        if (self.config.use_momentum_infonce and 
            self.negative_miner is not None and 
            cpc_features is not None):
            
            try:
                from models.cpc_losses import advanced_info_nce_loss_with_momentum
                
                # Use advanced momentum-based InfoNCE
                momentum_loss, mining_stats = advanced_info_nce_loss_with_momentum(
                    z_context=cpc_features[:, :-1, :].reshape(-1, cpc_features.shape[-1]),
                    z_target=cpc_features[:, 1:, :].reshape(-1, cpc_features.shape[-1]),
                    momentum_miner=self.negative_miner,
                    temperature=current_temp if 'current_temp' in locals() else 0.1,
                    use_hard_negatives=True
                )
                
                logger.debug(f"🚀 Momentum InfoNCE: {float(momentum_loss):.4f}, "
                           f"hard_negatives_used: {mining_stats.get('hard_negatives_used', False)}")
                
            except Exception as e:
                logger.warning(f"Momentum InfoNCE failed: {e}")
                momentum_loss = 0.0
        
        # 🔧 TOTAL LOSS: Combine all loss components
        total_loss = (
            classification_loss + 
            self.config.temporal_infonce_weight * temporal_infonce_loss +
            getattr(self.config, 'momentum_infonce_weight', 0.2) * momentum_loss
        )
        
        # 📊 METRICS: Compute accuracy
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
        
        return total_loss, {
            'classification_loss': classification_loss,
            'temporal_infonce_loss': temporal_infonce_loss,
            'momentum_loss': momentum_loss,
            'total_loss': total_loss,
            'accuracy': accuracy
        }
    
    def train_step(self, train_state: TrainStateWithBatchStats, batch) -> Tuple[TrainStateWithBatchStats, Dict[str, Any]]:
        """Enhanced training step with all improvements."""
        
        # Update training progress for adaptive components
        if self.total_training_steps > 0:
            self.training_progress = float(train_state.step) / self.total_training_steps
        
        # Create RNG key for this step
        step_key = jax.random.PRNGKey(train_state.step)
        
        def loss_fn(params):
            return self.enhanced_loss_fn(train_state, params, batch, step_key)
        
        # Compute loss and gradients
        (total_loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        
        # Apply gradients
        new_train_state = train_state.apply_gradients(grads=grads)
        
        # Create metrics
        metrics = create_training_metrics(
            step=new_train_state.step,
            epoch=getattr(self, 'current_epoch', 0),
            loss=float(total_loss),
            accuracy=float(loss_dict['accuracy'])
        )
        
        # Add detailed loss breakdown
        detailed_metrics = {
            'total_loss': float(total_loss),
            'classification_loss': float(loss_dict['classification_loss']),
            'temporal_infonce_loss': float(loss_dict['temporal_infonce_loss']),
            'momentum_loss': float(loss_dict['momentum_loss']),
            'accuracy': float(loss_dict['accuracy']),
            'training_progress': self.training_progress
        }
        
        return new_train_state, detailed_metrics
    
    def eval_step(self, train_state: TrainStateWithBatchStats, batch) -> Dict[str, Any]:
        """Execute single evaluation step."""
        signals, labels = batch
        
        # Forward pass without gradients using current batch_stats
        output = train_state.apply_fn(
            {'params': train_state.params, 'batch_stats': train_state.batch_stats},
            signals, 
            training=False,
            training_progress=self.training_progress,
            return_intermediates=False
        )
        
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output
        
        # Compute evaluation metrics
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == labels)
        
        # Compute loss for monitoring
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        
        return create_training_metrics(
            step=train_state.step,
            epoch=getattr(self, 'current_epoch', 0),
            loss=float(loss),
            accuracy=float(accuracy)
        )
    
    def update_training_progress(self, current_step: int, total_steps: int):
        """Update training progress for adaptive components."""
        self.training_progress = float(current_step) / max(total_steps, 1)
        
        # Update temperature controller if available
        if self.temp_controller is not None:
            # This will be updated during loss computation
            pass
    
    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get enhanced training metrics and statistics."""
        base_metrics = super().get_training_metrics() if hasattr(super(), 'get_training_metrics') else {}
        
        enhanced_metrics = {
            'training_progress': self.training_progress,
            'total_training_steps': self.total_training_steps,
            'enhancements_active': {
                'temporal_transformer': self.config.use_temporal_transformer,
                'learnable_thresholds': getattr(self.config, 'use_learnable_thresholds', True),
                'enhanced_lif': getattr(self.config, 'use_enhanced_lif', True),
                'momentum_infonce': getattr(self.config, 'use_momentum_negatives', False),
                'adaptive_temperature': self.temp_controller is not None
            }
        }
        
        # Add temperature controller stats if available
        if self.temp_controller is not None:
            enhanced_metrics['temperature_stats'] = self.temp_controller.get_temperature_stats()
        
        # Add negative mining stats if available
        if self.negative_miner is not None:
            # Would add mining statistics here
            enhanced_metrics['negative_mining_active'] = True
        
        return {**base_metrics, **enhanced_metrics}


# Export trainer class
__all__ = [
    "CompleteEnhancedTrainer"
]

