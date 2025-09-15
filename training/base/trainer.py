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
from flax.training import train_state, checkpoints
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
from models.cpc import RealCPCEncoder, RealCPCConfig, temporal_info_nce_loss
from models.snn.core import SNNClassifier
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
        if hasattr(config, 'wandb_config') and getattr(config, 'wandb_config', None):
            # Use enhanced metrics logger with comprehensive tracking
            self.enhanced_logger = create_enhanced_metrics_logger(
                config=config.__dict__ if hasattr(config, '__dict__') else vars(config),
                experiment_name=getattr(config, 'experiment_name', f"base-trainer-{getattr(config, 'seed', 42)}"),
                output_dir=config.output_dir
            )
            self.tracker = self.enhanced_logger  # Use enhanced logger as tracker
            logger.info("🚀 Using enhanced W&B metrics logger")
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
        
        self.snn_classifier = SNNClassifier(
            hidden_size=self.config.snn_hidden_sizes[0],
            num_classes=self.config.num_classes,
            num_layers=self.config.snn_num_layers
        )
        
        # Create combined model
        class StandardCPCSNNModel(nn.Module):
            """Standard CPC+SNN model."""
            
            def setup(self):
                self.cpc = cpc_encoder
                self.bridge = spike_bridge  
                self.snn = snn_classifier
            
            def __call__(self, x, training=True, return_stats: bool = False):
                cpc_features = self.cpc(x, training=training)
                # Reduce feature dimension for SpikeBridge (expects [batch, time])
                spike_in = jnp.mean(cpc_features, axis=-1)
                # Per-sample normalization to stabilize spikes
                _mean = jnp.mean(spike_in, axis=1, keepdims=True)
                _std = jnp.std(spike_in, axis=1, keepdims=True) + 1e-6
                spike_in = (spike_in - _mean) / _std
                spikes = self.bridge(spike_in[..., None], training=training)
                logits = self.snn(spikes, training=training)
                if return_stats:
                    spike_rate_mean = jnp.mean(spikes)
                    spike_rate_std = jnp.std(spikes)
                    return {
                        'logits': logits,
                        'cpc_features': cpc_features,
                        'spike_rate_mean': spike_rate_mean,
                        'spike_rate_std': spike_rate_std,
                    }
                return logits
        
        # Store references for individual component access
        cpc_encoder = self.cpc_encoder
        spike_bridge = self.spike_bridge
        snn_classifier = self.snn_classifier
        
        self.model = StandardCPCSNNModel()
        
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
                    # Contrastive loss (temporal InfoNCE) na cechach CPC
                    cpc_feats = out.get('cpc_features')
                    cpc_loss = temporal_info_nce_loss(
                        cpc_feats,
                        temperature=self.config.cpc_temperature,
                        max_prediction_steps=self.config.cpc_prediction_steps
                    ) if cpc_feats is not None else jnp.array(0.0)
                    if self.config.use_focal_loss:
                        probs = jax.nn.softmax(logits, axis=-1)
                        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
                        pt = probs[jnp.arange(len(y)), y]
                        focal_weight = (1 - pt) ** self.config.focal_gamma
                        loss = jnp.mean(focal_weight * ce_loss)
                    else:
                        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
                    # Joint objective: classification + alpha * cpc_loss (placeholder 0.0 if not computed)
                    total_loss = loss + getattr(self.config, 'cpc_joint_weight', 0.2) * cpc_loss
                    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
                    spike_mean = out['spike_rate_mean']
                    spike_std = out['spike_rate_std']
                    return total_loss, (acc, spike_mean, spike_std, cpc_loss)

                (loss, (acc, spike_mean, spike_std, cpc_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

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
                return new_state, loss, acc, spike_mean, spike_std, cpc_loss, total_norm, gn_cpc, gn_bridge, gn_snn

            # Donate state to reduce copies on GPU
            self._jit_update = jax.jit(update_fn, donate_argnums=(0,))

        new_state, loss, acc, spike_mean, spike_std, cpc_loss, total_norm, gn_cpc, gn_bridge, gn_snn = self._jit_update(train_state, signals, labels)

        metrics = {
            'step': int(new_state.step),
            'epoch': int(getattr(self, 'current_epoch', 0)),
            'total_loss': float(loss),
            'accuracy': float(acc),
            'cpc_loss': float(cpc_loss),
            'spike_rate_mean': float(spike_mean),
            'spike_rate_std': float(spike_std),
            'grad_norm_total': float(total_norm),
            'grad_norm_cpc': float(gn_cpc),
            'grad_norm_bridge': float(gn_bridge),
            'grad_norm_snn': float(gn_snn),
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
        tx = optax.chain(
            # ✅ Stabilizacja: adaptacyjne ograniczenie gradientu zapobiega gnorm=inf
            optax.adaptive_grad_clip(1.0),
            optax.clip_by_global_norm(5.0),
            optax.adamw(self.config.learning_rate, weight_decay=1e-4),
        )
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
                        "step=%s epoch=%s loss=%.4f acc=%.3f cpc=%.4f spike_mean=%.3f spike_std=%.3f gnorm=%.3f g_cpc=%.3f g_br=%.3f g_snn=%.3f",
                        metrics['step'], metrics['epoch'], metrics['total_loss'], metrics['accuracy'], metrics.get('cpc_loss', 0.0),
                        metrics['spike_rate_mean'], metrics['spike_rate_std'], metrics['grad_norm_total'],
                        metrics['grad_norm_cpc'], metrics['grad_norm_bridge'], metrics['grad_norm_snn']
                    )
                except Exception:
                    pass
            
            # Eval at end of epoch
            eval_batch = (test_signals[:batch_size], test_labels[:batch_size])
            eval_metrics = self.eval_step(state, eval_batch)
            # Epoch aggregation write
            if agg_count > 0:
                epoch_metrics = {
                    'epoch': int(self.current_epoch),
                    'mean_loss': agg['total_loss'] / agg_count,
                    'mean_accuracy': agg['accuracy'] / agg_count,
                    'mean_grad_norm_total': agg['grad_norm_total'] / agg_count,
                    'mean_spike_rate_mean': agg['spike_rate_mean'] / agg_count,
                    'mean_spike_rate_std': agg['spike_rate_std'] / agg_count,
                    'eval_loss': float(eval_metrics.get('total_loss', 0.0)),
                    'eval_accuracy': float(eval_metrics.get('accuracy', 0.0)),
                    'eval_spike_rate_mean': float(eval_metrics.get('spike_rate_mean', 0.0)),
                    'eval_spike_rate_std': float(eval_metrics.get('spike_rate_std', 0.0)),
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
                self.logger.info(
                    "eval epoch=%s loss=%.4f acc=%.3f spike_mean=%.3f spike_std=%.3f",
                    eval_metrics['epoch'], eval_metrics['total_loss'], eval_metrics['accuracy'],
                    eval_metrics['spike_rate_mean'], eval_metrics['spike_rate_std']
                )
            except Exception:
                pass
        
        # Final simple evaluation over entire test set in batches
        correct = 0
        total = 0
        for s in range(0, len(test_signals), batch_size):
            e = min(s + batch_size, len(test_signals))
            batch = (test_signals[s:e], test_labels[s:e])
            logits = state.apply_fn(state.params, batch[0], training=False)
            preds = jnp.argmax(logits, axis=-1)
            correct += int(jnp.sum(preds == batch[1]))
            total += (e - s)
        test_accuracy = correct / max(1, total)
        
        return {
            'success': True,
            'test_accuracy': float(test_accuracy),
            'epochs_completed': num_epochs
        }

# Export trainer classes
__all__ = [
    "TrainerBase",
    "CPCSNNTrainer"
]
