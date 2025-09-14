"""
Unified Trainer: FIXED Multi-Stage Training

✅ CRITICAL FIXES APPLIED (2025-01-27):
- Removed hardcoded epoch=0, implemented proper epoch tracking
- Fixed stop_gradient in Stage 2 to allow CPC fine-tuning  
- Deterministic random keys for reproducibility
- Real evaluation metrics (not mock)

Streamlined implementation of CPC+SNN multi-stage training:
- Stage 1: CPC pretraining (self-supervised)
- Stage 2: SNN training (with CPC fine-tuning)
- Stage 3: Joint fine-tuning
- Optimized for production use with real learning
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
from .utils import ProgressTracker, format_training_time
from .training_metrics import create_training_metrics

# Import new modular components
from .stages import _cpc_train_step, _snn_train_step, _joint_train_step, train_stage
from .pipeline import train_unified_pipeline, _compute_real_evaluation_metrics
from .metrics import eval_step, validate_and_log_step, compute_comprehensive_metrics

# Import models
from models.cpc.core import CPCEncoder
from models.cpc.losses import enhanced_info_nce_loss
from models.snn.core import SNNClassifier
from models.bridge.core import ValidatedSpikeBridge

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
    
    # ✅ NEW: Reproducibility and evaluation
    random_seed: int = 42
    enable_cpc_finetuning_stage2: bool = True  # ✅ Allow CPC learning in Stage 2


class UnifiedTrainer(TrainerBase):
    """
    ✅ FIXED: Unified trainer for multi-stage CPC+SNN training.
    
    Implements progressive training strategy with REAL LEARNING:
    1. CPC pretraining for representation learning
    2. SNN training with OPTIONAL CPC fine-tuning (not frozen!)
    3. Joint fine-tuning of full pipeline
    
    CRITICAL FIXES:
    - Real epoch tracking (not hardcoded 0)
    - Optional CPC fine-tuning in Stage 2 
    - Deterministic random keys
    - Real evaluation metrics
    """
    
    def __init__(self, config: UnifiedTrainingConfig):
        super().__init__(config)
        self.config: UnifiedTrainingConfig = config
        
        # Stage tracking ✅ FIXED: Real epoch tracking
        self.current_stage = 1
        self.current_epoch = 0  # ✅ NEW: Real epoch counter
        self.stage_start_time = None
        
        # ✅ NEW: Deterministic random key management
        self.master_key = jax.random.PRNGKey(self.config.random_seed)
        self.subkeys = {}
        
        # Model components
        self.cpc_encoder = None
        self.snn_classifier = None  
        self.spike_bridge = None
        
        # ✅ NEW: Stage 1 CPC parameters (for optional Stage 2 fine-tuning)
        self.stage1_cpc_params = None
        
        logger.info("✅ Initialized FIXED UnifiedTrainer for real multi-stage training")
    
    def _get_deterministic_key(self, name: str) -> jax.random.PRNGKey:
        """✅ NEW: Get deterministic random key for reproducibility."""
        if name not in self.subkeys:
            self.master_key, subkey = jax.random.split(self.master_key)
            self.subkeys[name] = subkey
        return self.subkeys[name]
    
    def create_model(self):
        """Create individual model components."""
        self.cpc_encoder = CPCEncoder(latent_dim=self.config.cpc_latent_dim)
        self.spike_bridge = ValidatedSpikeBridge()
        self.snn_classifier = SNNClassifier(
            hidden_size=self.config.snn_hidden_size,
            num_classes=self.config.num_classes
        )
        
        logger.info("Created model components: CPC, SpikeBridge, SNN")
    
    def create_train_state(self, model, sample_input):
        """Create training state for current stage."""
        key = self._get_deterministic_key(f"init_stage_{self.current_stage}")
        
        if self.current_stage == 1:
            # CPC pretraining - only CPC encoder
            params = self.cpc_encoder.init(key, sample_input)
            apply_fn = self.cpc_encoder.apply
        elif self.current_stage == 2:
            # ✅ FIXED: SNN training with OPTIONAL CPC fine-tuning
            if self.config.enable_cpc_finetuning_stage2:
                # ✅ SOLUTION: Include CPC for fine-tuning (not frozen)
                cpc_params = self.stage1_cpc_params  # Start from Stage 1 weights
                latent_input = jnp.ones((sample_input.shape[0], sample_input.shape[1] // 16, self.config.cpc_latent_dim))
                spike_params = self.spike_bridge.init(key, latent_input, key)
                snn_input = jnp.ones((sample_input.shape[0], 50, self.config.cpc_latent_dim))
                snn_params = self.snn_classifier.init(key, snn_input)
                
                params = {'cpc': cpc_params, 'spike_bridge': spike_params, 'snn': snn_params}
                apply_fn = self._snn_with_cpc_apply_fn
            else:
                # Legacy frozen CPC approach
                latent_input = jnp.ones((sample_input.shape[0], sample_input.shape[1] // 16, self.config.cpc_latent_dim))
                spike_params = self.spike_bridge.init(key, latent_input, key)
                snn_input = jnp.ones((sample_input.shape[0], 50, self.config.cpc_latent_dim))
                snn_params = self.snn_classifier.init(key, snn_input)
                
                params = {'spike_bridge': spike_params, 'snn': snn_params}
                apply_fn = self._snn_frozen_apply_fn
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
    
    def _snn_frozen_apply_fn(self, params, x_latent, training=True):
        """Apply function for SNN training stage (legacy frozen CPC)."""
        # ✅ CRITICAL FIX: Use training parameter, not key
        spikes = self.spike_bridge.apply(params['spike_bridge'], x_latent, training=training)
        logits = self.snn_classifier.apply(params['snn'], spikes)
        return logits
        
    def _snn_with_cpc_apply_fn(self, params, x, training=True):
        """✅ NEW: Apply function for SNN training with CPC fine-tuning."""
        latents = self.cpc_encoder.apply(params['cpc'], x)
        # ✅ CRITICAL FIX: Use training parameter, not key
        spikes = self.spike_bridge.apply(params['spike_bridge'], latents, training=training)
        logits = self.snn_classifier.apply(params['snn'], spikes)
        return logits, latents
    
    def _joint_apply_fn(self, params, x, training=True):
        """Apply function for joint training stage."""
        latents = self.cpc_encoder.apply(params['cpc'], x)
        # ✅ CRITICAL FIX: Use training parameter, not key
        spikes = self.spike_bridge.apply(params['spike_bridge'], latents, training=training)
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
        """CPC pretraining step with InfoNCE loss. Delegated to stages module."""
        return _cpc_train_step(self, train_state, batch)
    
    def _snn_train_step(self, train_state, batch):
        """SNN training step with optional CPC fine-tuning. Delegated to stages module."""
        return _snn_train_step(self, train_state, batch)
    
    def _joint_train_step(self, train_state, batch):
        """Joint training step with both CPC and classification losses. Delegated to stages module."""
        return _joint_train_step(self, train_state, batch)
    
    def eval_step(self, train_state, batch):
        """Real evaluation step for current stage. Delegated to metrics module."""
        return eval_step(self, train_state, batch)
    
    def train_stage(self, stage: int, dataloader, num_epochs: int) -> Dict[str, Any]:
        """Train single stage with real epoch tracking. Delegated to stages module."""
        return train_stage(self, stage, dataloader, num_epochs)
    
    def train_unified_pipeline(self, train_dataloader, val_dataloader=None) -> Dict[str, Any]:
        """Execute complete multi-stage training pipeline. Delegated to pipeline module."""
        return train_unified_pipeline(self, train_dataloader, val_dataloader)
    
    def _compute_real_evaluation_metrics(self, val_dataloader) -> Dict[str, Any]:
        """Compute real evaluation metrics including ROC-AUC. Delegated to pipeline module."""
        return _compute_real_evaluation_metrics(self, val_dataloader)
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, classification_report,
            confusion_matrix
        )
        
        all_predictions = []
        all_true_labels = []
        all_losses = []
        
        # Collect predictions from all batches
        for batch in val_dataloader:
            metrics = self.eval_step(self.train_state, batch)
            all_losses.append(metrics.loss)
            
            if 'predictions' in metrics.custom_metrics:
                all_predictions.append(np.array(metrics.custom_metrics['predictions']))
                all_true_labels.append(np.array(metrics.custom_metrics['true_labels']))
        
        if not all_predictions:
            logger.warning("No predictions collected for evaluation")
            return {'avg_loss': float(np.mean(all_losses))}
        
        # Concatenate all predictions
        predictions = np.concatenate(all_predictions, axis=0)
        true_labels = np.concatenate(all_true_labels, axis=0)
        predicted_labels = np.argmax(predictions, axis=1)
        
        # ✅ SOLUTION: Real metrics computation (not mock!)
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        
        # Real ROC AUC (multi-class)
        roc_auc = roc_auc_score(true_labels, predictions, multi_class='ovr')
        
        # Average precision
        avg_precision = average_precision_score(true_labels, predictions, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Classification report
        class_names = ['continuous_gw', 'binary_merger', 'noise_only']
        class_report = classification_report(
            true_labels, predicted_labels, 
            target_names=class_names,
            output_dict=True
        )
        
        evaluation_results = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),  # ✅ REAL ROC-AUC!
            "average_precision": float(avg_precision),
            "avg_loss": float(np.mean(all_losses)),
            "confusion_matrix": cm.tolist(),
            "classification_report": class_report,
            "num_samples": len(true_labels),
            "class_names": class_names
        }
        
        logger.info(f"✅ REAL Evaluation Results:")
        logger.info(f"   - Accuracy: {accuracy:.4f}")
        logger.info(f"   - ROC-AUC: {roc_auc:.4f}")
        logger.info(f"   - F1-Score: {f1:.4f}")
        
        return evaluation_results


def create_unified_trainer(config: Optional[UnifiedTrainingConfig] = None) -> UnifiedTrainer:
    """Factory function to create FIXED unified trainer."""
    if config is None:
        config = UnifiedTrainingConfig()
    
    logger.info("✅ Creating FIXED UnifiedTrainer with real learning capabilities")
    return UnifiedTrainer(config) 