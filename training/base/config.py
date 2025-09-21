"""
Base training configuration classes.

This module contains configuration classes extracted from
base_trainer.py for better modularity.

Split from base_trainer.py for better maintainability.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Simplified training configuration - core parameters only."""
    
    # Model parameters - ✅ MEMORY OPTIMIZED
    model_name: str = "cpc_snn_gw"
    batch_size: int = 1  # ✅ MEMORY FIX: Ultra-small batch for GPU memory constraints
    learning_rate: float = 5e-5  # ✅ FIXED: Matching successful AResGW learning rate
    weight_decay: float = 1e-4
    num_epochs: int = 100
    num_classes: int = 2  # ✅ CONFIGURABLE: Binary classification by default
    label_smoothing: float = 0.1
    use_class_weighting: bool = True
    
    # Training optimization - MEMORY OPTIMIZED
    optimizer: str = "sgd"  # ✅ FIX: SGD uses 2x less GPU memory than Adam
    scheduler: str = "cosine"
    gradient_clipping: float = 1.0  # ✅ RE-ENABLED: Needed for CPC stability
    mixed_precision: bool = True
    grad_accum_steps: int = 1  # ✅ NEW: gradient accumulation
    
    # Monitoring
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 1000
    
    # Paths and experiment tracking
    output_dir: str = "outputs"
    use_wandb: bool = True
    use_tensorboard: bool = True
    project_name: str = "cpc-snn-gw"
    
    # Performance
    max_memory_gb: float = 8.0
    enable_profiling: bool = False
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "balanced_accuracy"  # loss | balanced_accuracy | f1
    early_stopping_mode: str = "max"  # min | max (for loss → min, for f1/balanced_accuracy → max)
    
    # ✅ New: checkpointing frequency
    checkpoint_every_epochs: int = 5
    
    # ✅ New: focal loss and class weighting controls
    use_focal_loss: bool = True
    focal_gamma: float = 1.8
    class1_weight: float = 1.1  # further reduce FP
    
    # ✅ New: Exponential Moving Average of parameters
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # ✅ SpikeBridge hyperparameters (exposed)
    spike_time_steps: int = 24
    spike_threshold: float = 0.55
    spike_learnable: bool = True
    spike_threshold_levels: int = 4
    spike_surrogate_type: str = "adaptive_multi_scale"
    spike_surrogate_beta: float = 4.0
    spike_pool_seq: bool = False
    spike_target_rate_low: float = 0.10
    spike_target_rate_high: float = 0.20
    
    # ✅ CPC pretraining / multitask parameters
    use_cpc_aux_loss: bool = True
    cpc_aux_weight: float = 0.2
    ce_loss_weight: float = 1.0
    cpc_freeze_first_n_convs: int = 0  # 0,1,2
    cpc_prediction_steps: int = 12
    cpc_num_negatives: int = 128
    cpc_use_hard_negatives: bool = True
    cpc_temperature: float = 0.07
    cpc_use_temporal_transformer: bool = True
    cpc_attention_heads: int = 8
    cpc_transformer_layers: int = 4
    cpc_dropout_rate: float = 0.1
    cpc_use_grad_checkpointing: bool = True
    cpc_use_mixed_precision: bool = True
    
    # ✅ SNN hyperparameters (exposed)
    snn_hidden_sizes: tuple = (256, 128, 64)  # ✅ 3 layers (256-128-64)
    snn_num_layers: int = 3  # ✅ Increased from 2 to 3
    snn_use_layer_norm: bool = True  # ✅ NEW: LayerNorm after each layer
    snn_dropout_rates: tuple = (0.2, 0.1, 0.0)  # ✅ Adaptive: decreases with depth
    snn_surrogate_betas: tuple = (10.0, 15.0, 20.0)  # ✅ Adaptive: increases with depth
    snn_use_input_projection: bool = True  # ✅ NEW: Project input to first layer
    
    # ✅ Data preprocessing parameters
    sequence_length: int = 2048  # 4096 samples (1 second at 4096 Hz) ÷ 2 for memory
    overlap: float = 0.5  # overlap between windows
    apply_whitening: bool = True
    bandpass_low: float = 20.0  # Hz
    bandpass_high: float = 1024.0  # Hz
    
    # ✅ Augmentation / injection / corrupted data support
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    noise_injection_std: float = 1e-24  # Very small compared to GW signal scale
    time_shift_samples: int = 50
    
    # ✅ Real LIGO data settings
    use_real_ligo_data: bool = True  # ✅ NEW: Use real GW150914 data
    synthetic_fallback: bool = True  # ✅ NEW: Fallback to synthetic if real fails
    real_data_cache_dir: str = "data_cache/real_ligo"
    
    def validate(self) -> bool:
        """Validate training configuration."""
        try:
            # Basic validation
            assert self.batch_size > 0, "batch_size must be positive"
            assert self.learning_rate > 0, "learning_rate must be positive"
            assert self.num_epochs > 0, "num_epochs must be positive"
            assert self.num_classes > 1, "num_classes must be > 1"
            assert 0 <= self.label_smoothing <= 1, "label_smoothing must be in [0,1]"
            
            # Optimizer validation
            valid_optimizers = ["sgd", "adam", "adamw"]
            assert self.optimizer in valid_optimizers, f"optimizer must be in {valid_optimizers}"
            
            valid_schedulers = ["constant", "cosine", "exponential", "linear"]
            assert self.scheduler in valid_schedulers, f"scheduler must be in {valid_schedulers}"
            
            # Memory validation
            assert self.max_memory_gb > 0, "max_memory_gb must be positive"
            assert self.grad_accum_steps > 0, "grad_accum_steps must be positive"
            
            # Early stopping validation
            valid_metrics = ["loss", "accuracy", "balanced_accuracy", "f1"]
            assert self.early_stopping_metric in valid_metrics, f"early_stopping_metric must be in {valid_metrics}"
            
            valid_modes = ["min", "max"]
            assert self.early_stopping_mode in valid_modes, f"early_stopping_mode must be in {valid_modes}"
            
            # SpikeBridge validation
            assert self.spike_time_steps > 0, "spike_time_steps must be positive"
            assert self.spike_threshold > 0, "spike_threshold must be positive"
            assert self.spike_threshold_levels > 0, "spike_threshold_levels must be positive"
            assert 0 < self.spike_target_rate_low < self.spike_target_rate_high < 1, "Invalid spike rate range"
            
            # CPC validation
            assert self.cpc_prediction_steps > 0, "cpc_prediction_steps must be positive"
            assert self.cpc_num_negatives > 0, "cpc_num_negatives must be positive"
            assert self.cpc_temperature > 0, "cpc_temperature must be positive"
            assert self.cpc_attention_heads > 0, "cpc_attention_heads must be positive"
            assert self.cpc_transformer_layers > 0, "cpc_transformer_layers must be positive"
            assert 0 <= self.cpc_dropout_rate <= 1, "cpc_dropout_rate must be in [0,1]"
            
            # SNN validation
            assert len(self.snn_hidden_sizes) > 0, "snn_hidden_sizes must not be empty"
            assert all(h > 0 for h in self.snn_hidden_sizes), "All SNN hidden sizes must be positive"
            assert self.snn_num_layers > 0, "snn_num_layers must be positive"
            assert len(self.snn_dropout_rates) == len(self.snn_hidden_sizes), "snn_dropout_rates length mismatch"
            assert len(self.snn_surrogate_betas) == len(self.snn_hidden_sizes), "snn_surrogate_betas length mismatch"
            
            # Data validation
            assert self.sequence_length > 0, "sequence_length must be positive"
            assert 0 <= self.overlap < 1, "overlap must be in [0,1)"
            assert 0 < self.bandpass_low < self.bandpass_high, "Invalid bandpass frequencies"
            assert 0 <= self.augmentation_prob <= 1, "augmentation_prob must be in [0,1]"
            
            return True
            
        except AssertionError as e:
            logger.error(f"Training configuration validation failed: {e}")
            return False
    
    def get_model_params(self) -> dict:
        """Get model-specific parameters."""
        return {
            'num_classes': self.num_classes,
            'spike_time_steps': self.spike_time_steps,
            'spike_threshold': self.spike_threshold,
            'spike_learnable': self.spike_learnable,
            'spike_threshold_levels': self.spike_threshold_levels,
            'spike_surrogate_type': self.spike_surrogate_type,
            'spike_surrogate_beta': self.spike_surrogate_beta,
            'snn_hidden_sizes': self.snn_hidden_sizes,
            'snn_num_layers': self.snn_num_layers,
            'snn_use_layer_norm': self.snn_use_layer_norm,
            'cpc_prediction_steps': self.cpc_prediction_steps,
            'cpc_temperature': self.cpc_temperature,
            'cpc_attention_heads': self.cpc_attention_heads,
            'cpc_transformer_layers': self.cpc_transformer_layers
        }
    
    def get_optimization_params(self) -> dict:
        """Get optimization-specific parameters."""
        return {
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'scheduler': self.scheduler,
            'gradient_clipping': self.gradient_clipping,
            'mixed_precision': self.mixed_precision,
            'grad_accum_steps': self.grad_accum_steps,
            'use_focal_loss': self.use_focal_loss,
            'focal_gamma': self.focal_gamma,
            'class1_weight': self.class1_weight,
            'use_ema': self.use_ema,
            'ema_decay': self.ema_decay
        }
    
    def get_data_params(self) -> dict:
        """Get data-specific parameters."""
        return {
            'sequence_length': self.sequence_length,
            'overlap': self.overlap,
            'apply_whitening': self.apply_whitening,
            'bandpass_low': self.bandpass_low,
            'bandpass_high': self.bandpass_high,
            'use_augmentation': self.use_augmentation,
            'augmentation_prob': self.augmentation_prob,
            'use_real_ligo_data': self.use_real_ligo_data,
            'synthetic_fallback': self.synthetic_fallback
        }


# Export configuration class
__all__ = [
    "TrainingConfig"
]
