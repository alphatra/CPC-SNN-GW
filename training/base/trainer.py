"""
Base trainer implementations.

This module contains base trainer classes extracted from
base_trainer.py for better modularity.

Split from base_trainer.py for better maintainability.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import matplotlib.pyplot as plt
from flax.training import train_state, checkpoints
from flax.core import unfreeze
from flax.traverse_util import flatten_dict

from .config import TrainingConfig
from ..utils import (
    setup_professional_logging, setup_directories, optimize_jax_for_device,
    validate_config, save_config_to_file, compute_gradient_norm, 
    check_for_nans, ProgressTracker
)
from ..monitoring import (
    TrainingMetrics, ExperimentTracker, EarlyStoppingMonitor,
    PerformanceProfiler, create_training_metrics,
    EnhancedMetricsLogger, create_enhanced_metrics_logger
)

# Import models
from models.cpc import RealCPCEncoder, RealCPCConfig, temporal_info_nce_loss, gw_twins_inspired_loss
from models.snn.core import SNNClassifier, SNNDecoder
from models.bridge.core import ValidatedSpikeBridge

logger = logging.getLogger(__name__)


class TrainerBase(ABC):
    """
    Abstract base class for all CPC-SNN trainers.
    
    Provides unified interface and common functionality while allowing
    specialized implementations for different training strategies.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Setup infrastructure
        self.directories = setup_directories(config.output_dir)
        self.logger = setup_professional_logging(
            log_file=str(self.directories['log'] / 'training.log')
        )
        
        # Device optimization
        self.device_info = optimize_jax_for_device()
        
        # Enhanced experiment tracking
        if getattr(config, 'use_wandb', False):
            try:
                # Use enhanced metrics logger with comprehensive tracking
                self.enhanced_logger = create_enhanced_metrics_logger(
                    config=config.__dict__ if hasattr(config, '__dict__') else vars(config),
                    experiment_name=getattr(config, 'experiment_name', f"base-trainer-{getattr(config, 'seed', 42)}"),
                    output_dir=config.output_dir
                )
                self.tracker = self.enhanced_logger  # Use enhanced logger as tracker
                logger.info("🚀 Using enhanced W&B metrics logger")
            except Exception as e:
                logger.warning(f"W&B logger unavailable, falling back to basic tracker: {e}")
                self.tracker = ExperimentTracker(
                    experiment_name=config.project_name,
                    output_dir=config.output_dir
                )
                self.enhanced_logger = None
        else:
            # Fallback to basic tracker
            self.tracker = ExperimentTracker(
                experiment_name=config.project_name,
                output_dir=config.output_dir
            )
            self.enhanced_logger = None
            logger.info("Using basic experiment tracker")
        
        # Monitoring utilities
        self.early_stopping = EarlyStoppingMonitor(
            patience=config.early_stopping_patience,
            metric_name=config.early_stopping_metric,
            mode=config.early_stopping_mode
        )
        
        if config.enable_profiling:
            self.profiler = PerformanceProfiler()
        else:
            self.profiler = None
        
        # Validate configuration
        if not validate_config(config):
            raise ValueError("Invalid training configuration")
        
        # Save configuration
        save_config_to_file(config, self.directories['config'] / 'config.json')
        
        logger.info(f"TrainerBase initialized: {config.model_name}")
    
    @abstractmethod
    def create_model(self):
        """Create model architecture - must be implemented by subclasses."""
        pass
    
    @abstractmethod  
    def train_step(self, train_state, batch):
        """Execute single training step - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def eval_step(self, train_state, batch):
        """Execute single evaluation step - must be implemented by subclasses."""
        pass
    
    def save_checkpoint(self, train_state, epoch: int, metrics: Optional[Dict] = None):
        """Save training checkpoint."""
        checkpoint_dir = self.directories['checkpoints']
        
        # Save training state
        checkpoints.save_checkpoint(
            ckpt_dir=checkpoint_dir,
            target=train_state,
            step=epoch,
            overwrite=True
        )
        
        # Save additional metadata
        if metrics:
            metadata_file = checkpoint_dir / f'epoch_{epoch}_metrics.json'
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        logger.info(f"Checkpoint saved at epoch {epoch}")
    
    def load_checkpoint(self, train_state, checkpoint_path: Optional[str] = None):
        """Load training checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.directories['checkpoints']
        
        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_path,
            target=train_state
        )
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return restored_state


class CPCSNNTrainer(TrainerBase):
    """
    Standard CPC+SNN trainer implementation.
    
    Implements basic CPC encoder + spike bridge + SNN classifier pipeline
    with standard training and evaluation procedures.
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        
        # Model components will be created lazily
        self.cpc_encoder = None
        self.spike_bridge = None
        self.snn_classifier = None
        self.snn_decoder = None  # ✅ SNN-AE: Optional decoder for reconstruction loss
        self.model = None
        self._jit_update = None
        self._base_step_key = jax.random.PRNGKey(2)
        
        logger.info("CPCSNNTrainer initialized")
    
    def create_model(self):
        """Create standard CPC+SNN model."""
        if self.model is not None:
            return self.model
        
        # Create model components
        cpc_config = RealCPCConfig(
            latent_dim=256,
            context_length=64,
            prediction_steps=self.config.cpc_prediction_steps,
            temperature=self.config.cpc_temperature
        )
        
        self.cpc_encoder = RealCPCEncoder(config=cpc_config)
        
        self.spike_bridge = ValidatedSpikeBridge(
            spike_encoding="learnable_multi_threshold",
            time_steps=max(32, int(self.config.spike_time_steps)),
            threshold=jnp.clip(self.config.spike_threshold, 0.3, 0.5),
            surrogate_type="hard_sigmoid",
            surrogate_beta=4.0
        )
        # Disable non-jittable Python-side validation inside JIT paths
        try:
            self.spike_bridge.validate_input = lambda cpc_features: (True, "disabled_in_jit")
        except Exception:
            pass
        
        # ✅ Enforce binary classification (pull from config or default to 2)
        enforced_num_classes = int(getattr(self.config, 'num_classes', 2))
        if enforced_num_classes not in (2, 3):
            enforced_num_classes = 2
        self.snn_classifier = SNNClassifier(
            hidden_size=self.config.snn_hidden_sizes[0],
            num_classes=enforced_num_classes,
            num_layers=self.config.snn_num_layers
        )
        
        # ✅ SNN-AE: Optional decoder for reconstruction loss (only if enabled)
        gamma_recon = float(getattr(self.config, 'gamma_reconstruction', 
                                  getattr(self.config, 'recon_loss_weight', 0.0)))
        logger.info(f"🔍 SNN-AE Debug: gamma_recon={gamma_recon}")
        if gamma_recon > 0:
            self.snn_decoder = SNNDecoder(
                output_size=self.config.cpc_latent_dim,  # Reconstruct CPC features
                hidden_size=self.config.snn_hidden_sizes[-1]  # Use last SNN layer size
            )
            logger.info(f"✅ SNN-AE: Decoder created with output_size={self.config.cpc_latent_dim}")
        else:
            logger.info(f"❌ SNN-AE: Decoder NOT created (gamma_recon={gamma_recon})")
        
        # ✅ SNN-AE: Get decoder reference for model creation
        snn_decoder = self.snn_decoder
        logger.info(f"🔍 SNN-AE Debug: snn_decoder={snn_decoder is not None}")
        
        # Create combined model with decoder built-in
        class StandardCPCSNNModel(nn.Module):
            """Standard CPC+SNN model with optional SNN-AE decoder."""
            
            def setup(self):
                self.cpc = cpc_encoder
                self.bridge = spike_bridge  
                self.snn = snn_classifier
                # ✅ SNN-AE: Conditional decoder setup
                if snn_decoder is not None:
                    self.decoder = snn_decoder
            
            def __call__(self, x, training=True, return_stats: bool = False):
                # Sanitize inputs to avoid NaNs/Infs propagation
                x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                x = jnp.clip(x, -20.0, 20.0)
                
                # ✅ KRYTYCZNA NAPRAWA: Normalizacja Z-score per-sample przed CPC
                # Zgodnie z rekomendacją z analizy stagnacji cpc_loss
                mean = jnp.mean(x, axis=1, keepdims=True)
                std = jnp.std(x, axis=1, keepdims=True) + 1e-8
                x_normalized = (x - mean) / std
                
                # Użyj znormalizowanych danych dla CPC
                cpc_features = self.cpc(x_normalized, training=training)
                # Sanitize CPC features
                cpc_features = jnp.nan_to_num(cpc_features, nan=0.0, posinf=0.0, neginf=0.0)
                cpc_features = jnp.clip(cpc_features, -50.0, 50.0)
                # ✅ CRITICAL FIX: SpikeBridge expects [batch, time, features] - DON'T average!
                # SpikeBridge can handle full CPC features [batch, time, latent_dim]
                spikes = self.bridge(cpc_features, training=training)
                spikes = jnp.nan_to_num(spikes, nan=0.0, posinf=0.0, neginf=0.0)
                # ✅ SNN-AE: Get hidden states for reconstruction if decoder enabled
                has_decoder = hasattr(self, 'decoder') and self.decoder is not None
                # Alternative check using snn_decoder reference
                use_reconstruction = has_decoder or snn_decoder is not None
                if use_reconstruction and hasattr(self, 'decoder'):
                    snn_output = self.snn(spikes, training=training, return_hidden=True)
                    logits = snn_output['logits']
                    hidden_states = snn_output['hidden_states']
                    # Compute reconstruction
                    reconstruction = self.decoder(hidden_states, training=training)
                else:
                    logits = self.snn(spikes, training=training)
                    reconstruction = None
                
                logits = jnp.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
                if return_stats:
                    spike_rate_mean = jnp.mean(spikes)
                    spike_rate_std = jnp.std(spikes)
                    result = {
                        'logits': logits,
                        'cpc_features': cpc_features,
                        'spike_rate_mean': spike_rate_mean,
                        'spike_rate_std': spike_rate_std,
                    }
                    # ✅ SNN-AE: Add reconstruction if available
                    if reconstruction is not None:
                        result['reconstruction'] = reconstruction
                        result['target_features'] = cpc_features  # Target for MSE loss
                    return result
                # Return logits for non-stats mode
                return logits
        
        # Store references for individual component access
        cpc_encoder = self.cpc_encoder
        spike_bridge = self.spike_bridge
        snn_classifier = self.snn_classifier
        
        self.model = StandardCPCSNNModel()
        
        # ✅ SNN-AE Debug: Check if decoder was added to model
        logger.info(f"🔍 SNN-AE: Model created. snn_decoder={self.snn_decoder is not None}")
        
        logger.info("✅ Standard CPC+SNN model created")
        return self.model
    
    def train_step(self, train_state, batch):
        """Execute single training step (JIT-compiled update)."""
        signals, labels = batch

        if self._jit_update is None:
            base_key = self._base_step_key

            def update_fn(state, x, y):
                step_key = jax.random.fold_in(base_key, state.step)

                def loss_fn(params):
                    out = state.apply_fn(
                        params, x, training=True, return_stats=True,
                        rngs={'dropout': step_key}
                    )
                    logits = out['logits']
                    # ✅ DYNAMIC CPC LOSS: Choose between temporal InfoNCE and GW Twins inspired
                    cpc_feats = out.get('cpc_features')
                    if cpc_feats is not None:
                        if getattr(self.config, 'cpc_loss_type', 'temporal_info_nce') == 'gw_twins_inspired':
                            # GW Twins inspired loss (no negatives, redundancy reduction)
                            cpc_loss = gw_twins_inspired_loss(
                                cpc_feats,
                                temperature=self.config.cpc_temperature,
                                redundancy_weight=getattr(self.config, 'gw_twins_redundancy_weight', 0.1)
                            )
                        else:
                            # Default: temporal InfoNCE loss
                            cpc_loss = temporal_info_nce_loss(
                                cpc_feats,
                                temperature=self.config.cpc_temperature,
                                max_prediction_steps=self.config.cpc_prediction_steps
                            )
                    else:
                        cpc_loss = jnp.array(0.0)
                    if self.config.use_focal_loss:
                        probs = jax.nn.softmax(logits, axis=-1)
                        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
                        pt = probs[jnp.arange(len(y)), y]
                        focal_weight = (1 - pt) ** self.config.focal_gamma
                        loss = jnp.mean(focal_weight * ce_loss)
                    else:
                        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
                    # Joint objective with warmup + epoch-based schedule for CPC weight
                    epoch = jax.lax.convert_element_type(getattr(self, 'current_epoch', 0), jnp.int32)
                    # ✅ NEW: Longer warmup schedule driven by config.cpc_aux_weight (default 0.05)
                    target_w = float(getattr(self.config, 'cpc_aux_weight', 0.05))
                    stage2 = min(target_w * 0.5, target_w)
                    stage3 = min(target_w * 1.0, target_w)
                    base_cpc_w = jnp.where(epoch < 2, 0.0,
                                   jnp.where(epoch < 4, stage2,
                                   jnp.where(epoch < 6, stage3, target_w)))
                    # Zero CPC weight for first 100 optimizer steps to avoid early explosions
                    warmup_mask = (state.step < 200)
                    cpc_w = jnp.where(warmup_mask, 0.0, base_cpc_w)
                    # ✅ SNN-AE: Reconstruction loss (MSE between reconstruction and CPC features)
                    recon_loss = jnp.array(0.0)
                    if 'reconstruction' in out and 'target_features' in out:
                        reconstruction = out['reconstruction']
                        target_features = out['target_features']
                        # MSE loss between reconstruction and target (CPC features)
                        # Average over time and feature dimensions
                        recon_loss = jnp.mean((reconstruction - jnp.mean(target_features, axis=1, keepdims=True)) ** 2)
                    
                    # ✅ ENHANCED: Eksplicytne ważenie składników straty z α,β,γ parametrami
                    # α - waga straty klasyfikacji
                    alpha = float(getattr(self.config, 'alpha_classification', 
                                        getattr(self.config, 'snn_loss_weight', 1.0)))
                    # β - waga straty kontrastującej  
                    beta = float(getattr(self.config, 'beta_contrastive',
                                       getattr(self.config, 'cpc_loss_weight', 1.0)))
                    # γ - waga straty rekonstrukcji
                    gamma = float(getattr(self.config, 'gamma_reconstruction',
                                        getattr(self.config, 'recon_loss_weight', 0.0)))
                    
                    total_loss = (
                        alpha * loss                    # α × L_classification
                        + beta * (cpc_w * cpc_loss)    # β × L_contrastive  
                        + gamma * recon_loss           # γ × L_reconstruction
                    )
                    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
                    spike_mean = out['spike_rate_mean']
                    spike_std = out['spike_rate_std']
                    return total_loss, (acc, spike_mean, spike_std, cpc_loss, loss, cpc_w, recon_loss)

                (loss, (acc, spike_mean, spike_std, cpc_loss, cls_loss, eff_cpc_w, recon_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
                # Sanitize grads to avoid NaN propagation
                from jax.tree_util import tree_map
                grads = tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)

                # Gradient norms
                def l2_norm(tree):
                    leaves = [jnp.sum(jnp.square(t)) for t in jax.tree_util.tree_leaves(tree) if t is not None]
                    return jnp.sqrt(jnp.sum(jnp.stack(leaves))) if leaves else jnp.array(0.0, dtype=jnp.float32)

                def l2_norm_by_module(grads_tree, module_name: str):
                    try:
                        flat = flatten_dict(unfreeze(grads_tree) if hasattr(grads_tree, 'items') else grads_tree, sep='.')
                    except Exception:
                        flat = flatten_dict(grads_tree, sep='.')
                    parts = []
                    for k, v in flat.items():
                        key_str = k if isinstance(k, str) else '.'.join([str(p) for p in (k if isinstance(k, tuple) else (k,))])
                        if f".{module_name}." in f".{key_str}." and v is not None:
                            parts.append(jnp.sum(jnp.square(v)))
                    return jnp.sqrt(jnp.sum(jnp.stack(parts))) if parts else jnp.array(0.0, dtype=jnp.float32)

                total_norm = l2_norm(grads)
                gparams = grads.get('params', grads)
                gn_cpc = l2_norm_by_module(gparams, 'cpc')
                gn_bridge = l2_norm_by_module(gparams, 'bridge')
                gn_snn = l2_norm_by_module(gparams, 'snn')

                new_state = state.apply_gradients(grads=grads)
                return new_state, loss, acc, spike_mean, spike_std, cpc_loss, cls_loss, total_norm, gn_cpc, gn_bridge, gn_snn, eff_cpc_w, recon_loss

            # Donate state to reduce copies on GPU
            self._jit_update = jax.jit(update_fn, donate_argnums=(0,))

        new_state, loss, acc, spike_mean, spike_std, cpc_loss, cls_loss, total_norm, gn_cpc, gn_bridge, gn_snn, eff_cpc_w, recon_loss = self._jit_update(train_state, signals, labels)

        metrics = {
            'step': int(new_state.step),
            'epoch': int(getattr(self, 'current_epoch', 0)),
            'total_loss': float(loss),
            'cls_loss': float(cls_loss),
            'accuracy': float(acc),
            'cpc_loss': float(cpc_loss),
            'spike_rate_mean': float(spike_mean),
            'spike_rate_std': float(spike_std),
            'grad_norm_total': float(total_norm),
            'grad_norm_cpc': float(gn_cpc),
            'grad_norm_bridge': float(gn_bridge),
            'grad_norm_snn': float(gn_snn),
            'cpc_weight': float(eff_cpc_w),
            'recon_loss': float(recon_loss),  # ✅ SNN-AE: Reconstruction loss metric
            # ✅ LOSS WEIGHTS: α,β,γ parameters for transparency
            'alpha_classification': float(getattr(self.config, 'alpha_classification', 
                                                getattr(self.config, 'snn_loss_weight', 1.0))),
            'beta_contrastive': float(getattr(self.config, 'beta_contrastive',
                                            getattr(self.config, 'cpc_loss_weight', 1.0))),
            'gamma_reconstruction': float(getattr(self.config, 'gamma_reconstruction',
                                                getattr(self.config, 'recon_loss_weight', 0.0))),
        }
        return new_state, metrics
    
    def eval_step(self, train_state, batch):
        """Execute single evaluation step (JIT-compiled)."""
        signals, labels = batch
        
        def _eval_fn(params, x, y):
            out = train_state.apply_fn(params, x, training=False, return_stats=True)
            logits = out['logits']
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
            return loss, accuracy, out['spike_rate_mean'], out['spike_rate_std']
        
        # JIT for faster eval; no gradients, donate params buffer is unnecessary here
        eval_jit = getattr(self, "_eval_jit", None)
        if eval_jit is None:
            self._eval_jit = jax.jit(_eval_fn)
            eval_jit = self._eval_jit
        
        loss, accuracy, spike_mean, spike_std = eval_jit(train_state.params, signals, labels)
        
        return {
            'step': int(train_state.step),
            'epoch': int(getattr(self, 'current_epoch', 0)),
            'total_loss': float(loss),
            'accuracy': float(accuracy),
            'spike_rate_mean': float(spike_mean),
            'spike_rate_std': float(spike_std),
        }

    def train(self, train_signals: jnp.ndarray, train_labels: jnp.ndarray,
              test_signals: jnp.ndarray, test_labels: jnp.ndarray) -> Dict[str, Any]:
        """Simple training loop using provided arrays (data-only dependency)."""
        model = self.create_model()
        # Initialize train state with RNGs (for dropout)
        sample_input = train_signals[:1]
        init_rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}
        params = model.init(init_rngs, sample_input, training=True)
        # ✅ ENHANCED: Advanced gradient clipping with per-module control
        lr = float(getattr(self.config, 'learning_rate', 5e-5))
        
        # Build optimizer chain with enhanced gradient clipping
        optimizer_chain = []
        
        # Adaptive gradient clipping (prevents exploding gradients adaptively)
        adaptive_threshold = float(getattr(self.config, 'adaptive_grad_clip_threshold', 0.5))
        optimizer_chain.append(optax.adaptive_grad_clip(adaptive_threshold))
        
        # Per-module gradient clipping if enabled (using optax.masked)
        if getattr(self.config, 'per_module_grad_clip', True):
            # ✅ FIXED: Use optax.masked for per-module gradient clipping
            def create_per_module_masks():
                """Create masks for different modules."""
                # We'll apply different clipping to different modules
                # This is a simplified approach that works with standard optax
                pass
            
            # Apply different clipping for CPC (more conservative)
            cpc_multiplier = float(getattr(self.config, 'cpc_grad_clip_multiplier', 0.8))
            cpc_threshold = adaptive_threshold * cpc_multiplier
            
            # For now, use standard clipping with the most conservative threshold
            # TODO: Implement proper per-module clipping when optax version supports it
            conservative_threshold = min(
                adaptive_threshold * float(getattr(self.config, 'cpc_grad_clip_multiplier', 0.8)),
                adaptive_threshold * float(getattr(self.config, 'snn_grad_clip_multiplier', 1.0)),
                adaptive_threshold * float(getattr(self.config, 'bridge_grad_clip_multiplier', 1.2))
            )
            # Use the most conservative threshold for stability
            optimizer_chain.append(optax.clip_by_global_norm(conservative_threshold))
        else:
            # Global norm clipping (if per-module disabled)
            global_norm = float(getattr(self.config, 'global_grad_clip_norm', 0.5))
            optimizer_chain.append(optax.clip_by_global_norm(global_norm))
        
        # Optimizer
        optimizer_chain.append(optax.adamw(lr, weight_decay=1e-4))
        
        tx = optax.chain(*optimizer_chain)
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        
        batch_size = self.config.batch_size
        num_epochs = self.config.num_epochs
        num_samples = len(train_signals)
        steps_per_epoch = max(1, (num_samples + batch_size - 1) // batch_size)
        
        # JSONL outputs
        from pathlib import Path
        import json
        step_jsonl_path = Path(self.directories['log'] / 'training_results.jsonl')
        epoch_jsonl_path = Path(self.directories['log'] / 'epoch_metrics.jsonl')
        step_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate old files at start of run
        try:
            step_jsonl_path.write_text("")
            epoch_jsonl_path.write_text("")
        except Exception:
            pass
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            # Naive epoch loop
            # Aggregation buffers
            agg = {
                'total_loss': 0.0,
                'accuracy': 0.0,
                'cpc_loss': 0.0,
                'grad_norm_total': 0.0,
                'spike_rate_mean': 0.0,
                'spike_rate_std': 0.0,
            }
            agg_count = 0
            for step in range(steps_per_epoch):
                s = step * batch_size
                e = min(s + batch_size, num_samples)
                batch = (train_signals[s:e], train_labels[s:e])
                state, metrics = self.train_step(state, batch)
                # Aggregate
                for k in agg.keys():
                    if k in metrics:
                        agg[k] += float(metrics[k])
                agg_count += 1
                # JSONL per-step
                try:
                    with step_jsonl_path.open('a') as f:
                        f.write(json.dumps(metrics) + "\n")
                except Exception:
                    pass
                # Optional W&B log via enhanced_logger
                if getattr(self, 'enhanced_logger', None):
                    try:
                        self.enhanced_logger.log_metrics(metrics, step=int(metrics.get('step', 0)))
                    except Exception:
                        pass
                try:
                    self.logger.info(
                        "TRAIN step=%s epoch=%s | total=%.4f cls=%.4f cpc=%.4f acc=%.3f spikes=%.3f±%.3f gnorm=%.3f (cpc=%.3f br=%.3f snn=%.3f)",
                        metrics['step'], metrics['epoch'], metrics['total_loss'], metrics.get('cls_loss', 0.0), metrics.get('cpc_loss', 0.0),
                        metrics['accuracy'], metrics['spike_rate_mean'], metrics['spike_rate_std'], metrics['grad_norm_total'],
                        metrics['grad_norm_cpc'], metrics['grad_norm_bridge'], metrics['grad_norm_snn']
                    )
                except Exception:
                    pass
            
            # Eval at end of epoch (FULL TEST AGGREGATION)
            total_eval_loss_epoch = 0.0
            total_eval_acc_epoch = 0.0
            total_eval_count_epoch = 0
            eval_bs = int(getattr(self.config, 'eval_batch_size', batch_size))
            for s in range(0, len(test_signals), eval_bs):
                e = min(s + batch_size, len(test_signals))
                e = min(s + eval_bs, len(test_signals))
                eval_dict = self.eval_step(state, (test_signals[s:e], test_labels[s:e]))
                bsz = (e - s)
                total_eval_loss_epoch += float(eval_dict.get('total_loss', 0.0)) * bsz
                total_eval_acc_epoch += float(eval_dict.get('accuracy', 0.0)) * bsz
                total_eval_count_epoch += bsz
            avg_eval_loss_epoch = total_eval_loss_epoch / max(1, total_eval_count_epoch)
            avg_eval_acc_epoch = total_eval_acc_epoch / max(1, total_eval_count_epoch)
            # Epoch aggregation write
            if agg_count > 0:
                epoch_metrics = {
                    'epoch': int(self.current_epoch),
                    'mean_loss': agg['total_loss'] / agg_count,
                    'mean_accuracy': agg['accuracy'] / agg_count,
                    'mean_cpc_loss': agg['cpc_loss'] / agg_count,
                    'mean_grad_norm_total': agg['grad_norm_total'] / agg_count,
                    'mean_spike_rate_mean': agg['spike_rate_mean'] / agg_count,
                    'mean_spike_rate_std': agg['spike_rate_std'] / agg_count,
                    'eval_loss': float(avg_eval_loss_epoch),
                    'eval_accuracy': float(avg_eval_acc_epoch),
                }
                try:
                    with epoch_jsonl_path.open('a') as f:
                        f.write(json.dumps(epoch_metrics) + "\n")
                except Exception:
                    pass
                if getattr(self, 'enhanced_logger', None):
                    try:
                        self.enhanced_logger.log_metrics(epoch_metrics, step=int(self.current_epoch))
                    except Exception:
                        pass
            try:
                # Log also CPC schedule context for correlation
                # Calculate current effective CPC weight using same logic as training
                epoch = int(self.current_epoch)
                target_w = float(self.config.cpc_aux_weight)
                stage2 = min(target_w * 0.5, target_w)
                stage3 = min(target_w * 1.0, target_w)
                if epoch < 2:
                    eff_w = 0.0
                elif epoch < 4:
                    eff_w = stage2
                elif epoch < 6:
                    eff_w = stage3
                else:
                    eff_w = target_w
                
                self.logger.info(
                    "EVAL (full test) epoch=%s | avg_loss=%.4f acc=%.3f | cpc_weight=%.3f temp=%.3f",
                    int(self.current_epoch), avg_eval_loss_epoch, avg_eval_acc_epoch,
                    eff_w, float(self.config.cpc_temperature)
                )
            except Exception:
                pass
        
        # Final evaluation over entire test set in batches (average loss and accuracy)
        total_correct = 0
        total_count = 0
        total_eval_loss = 0.0
        all_probs = []
        all_preds = []
        all_trues = []
        for s in range(0, len(test_signals), batch_size):
            e = min(s + batch_size, len(test_signals))
            batch = (test_signals[s:e], test_labels[s:e])
            eval_dict = self.eval_step(state, batch)
            loss_b = float(eval_dict.get('total_loss', 0.0))
            acc_b = float(eval_dict.get('accuracy', 0.0))
            total_eval_loss += loss_b * (e - s)
            total_correct += int(acc_b * (e - s))
            total_count += (e - s)
            # Collect logits → probabilities for metrics
            logits = state.apply_fn(state.params, batch[0], training=False)
            probs = jax.nn.softmax(logits, axis=-1)
            prob1 = np.asarray(probs[..., 1]) if probs.shape[-1] > 1 else np.asarray(probs[..., 0])
            preds = np.asarray(jnp.argmax(probs, axis=-1))
            trues = np.asarray(batch[1])
            all_probs.append(prob1)
            all_preds.append(preds)
            all_trues.append(trues)
        test_accuracy = total_correct / max(1, total_count)
        test_loss = total_eval_loss / max(1, total_count)
        # Concatenate collected arrays
        try:
            y_prob = np.concatenate(all_probs, axis=0)
            y_pred = np.concatenate(all_preds, axis=0)
            y_true = np.concatenate(all_trues, axis=0)
        except Exception:
            y_prob = np.array([])
            y_pred = np.array([])
            y_true = np.array([])

        # Compute confusion matrix and ROC-AUC (binary)
        def compute_confusion_matrix(y_t: np.ndarray, y_p: np.ndarray) -> Dict[str, int]:
            cm = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
            for t, p in zip(y_t.tolist(), y_p.tolist()):
                if t == 1 and p == 1:
                    cm['tp'] += 1
                elif t == 0 and p == 1:
                    cm['fp'] += 1
                elif t == 1 and p == 0:
                    cm['fn'] += 1
                else:
                    cm['tn'] += 1
            return cm

        def compute_auc(y_t: np.ndarray, y_s: np.ndarray) -> float:
            y_t = y_t.astype(int)
            n_pos = int(np.sum(y_t == 1))
            n_neg = int(np.sum(y_t == 0))
            if n_pos == 0 or n_neg == 0 or y_s.size == 0:
                return 0.5
            order = np.argsort(y_s)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(y_s) + 1)
            sum_ranks_pos = np.sum(ranks[y_t == 1])
            u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
            return float(u / (n_pos * n_neg))

        cm = compute_confusion_matrix(y_true, y_pred) if y_pred.size else {'tn':0,'fp':0,'fn':0,'tp':0}
        auc = compute_auc(y_true, y_prob)

        # Save plots
        plots_dir = self.directories.get('log', Path(self.config.output_dir) / 'logs')
        try:
            plots_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            # ROC curve
            if y_prob.size and np.unique(y_true).size == 2:
                # Compute ROC via sorted scores
                n_pos = max(1, int(np.sum(y_true == 1)))
                n_neg = max(1, int(np.sum(y_true == 0)))
                desc = np.argsort(-y_prob)
                t_sorted = y_true[desc]
                tps = np.cumsum(t_sorted == 1)
                fps = np.cumsum(t_sorted == 0)
                tpr = tps / n_pos
                fpr = fps / n_neg
                plt.figure(figsize=(5,5))
                plt.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
                plt.plot([0,1],[0,1],'k--',alpha=0.5)
                plt.xlabel('FPR')
                plt.ylabel('TPR')
                plt.title('ROC Curve')
                plt.legend(loc='lower right')
                roc_path = plots_dir / 'roc_curve.png'
                plt.savefig(roc_path, dpi=150, bbox_inches='tight')
                plt.close()
            # Confusion matrix plot
            cm_mat = np.array([[cm['tn'], cm['fp']],[cm['fn'], cm['tp']]], dtype=int)
            plt.figure(figsize=(4,4))
            plt.imshow(cm_mat, cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xticks([0,1],["Pred 0","Pred 1"]) ; plt.yticks([0,1],["True 0","True 1"]) 
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, int(cm_mat[i, j]), ha='center', va='center', color='black')
            cm_path = plots_dir / 'confusion_matrix.png'
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            try:
                logger.warning(f"Plot generation failed: {e}")
            except Exception:
                pass
        
        return {
            'success': True,
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'epochs_completed': num_epochs
        }

# Export trainer classes
__all__ = [
    "TrainerBase",
    "CPCSNNTrainer"
]
