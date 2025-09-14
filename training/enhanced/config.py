"""
Configuration classes for complete enhanced training system.

This module contains configuration classes extracted from
complete_enhanced_training.py for better modularity.

Split from complete_enhanced_training.py for better maintainability.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from flax import struct

from ..base.config import TrainingConfig

# Custom TrainState with batch_stats support
@struct.dataclass
class TrainStateWithBatchStats:
    """Custom TrainState that includes batch_stats for BatchNorm layers."""
    step: int
    apply_fn: Any = struct.field(pytree_node=False)
    params: Any
    tx: Any = struct.field(pytree_node=False)
    opt_state: Any
    batch_stats: Any

    def apply_gradients(self, *, grads, **kwargs):
        """Apply gradients to parameters."""
        import optax
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs
        )


@dataclass
class CompleteEnhancedConfig(TrainingConfig):
    """Configuration for complete enhanced training with all 5 improvements + MATHEMATICAL FRAMEWORK ENHANCEMENTS."""
    
    # 🧮 MATHEMATICAL FRAMEWORK ENHANCEMENTS
    # Based on "Neuromorphic Gravitational-Wave Detection: Complete Mathematical Framework"
    
    # Temporal InfoNCE (Equation 1) - mathematically proven for small batches
    use_temporal_infonce: bool = True
    temporal_infonce_weight: float = 0.3  # Framework: λ_temporal = 0.3
    max_prediction_steps: int = 12  # Framework: k_max = 12 steps
    
    # Phase-Preserving Encoding (Equation 2)
    use_phase_preserving_encoding: bool = True
    phase_preservation_weight: float = 0.25  # Framework: α_phase = 0.25
    edge_detection_thresholds: int = 4  # Framework: 4 edge detection levels
    
    # Enhanced Surrogate Gradients (Equation 3)
    use_adaptive_surrogate: bool = True
    surrogate_adaptation_rate: float = 0.02  # Framework: η_surrogate = 0.02
    multi_scale_surrogate: bool = True  # Framework: use multi-scale
    
    # Memory-Enhanced LIF (Equation 4)
    use_memory_enhanced_lif: bool = True
    memory_decay_constant: float = 0.95  # Framework: β_memory = 0.95
    refractory_enhancement: bool = True  # Framework: enhanced refractory
    
    # Momentum InfoNCE (Equation 5)
    use_momentum_infonce: bool = True
    momentum_coefficient: float = 0.999  # Framework: μ = 0.999
    hard_negative_ratio: float = 0.3  # Framework: ρ_hard = 0.3
    
    # ✅ 1. ADAPTIVE MULTI-SCALE SURROGATE GRADIENTS
    surrogate_gradient_type: str = "adaptive_multi_scale"  # 🚀 Use adaptive surrogate
    surrogate_beta_schedule: str = "curriculum"  # adaptive, curriculum, constant
    initial_surrogate_beta: float = 1.0
    final_surrogate_beta: float = 10.0
    surrogate_curriculum_epochs: int = 50
    
    # ✅ 2. TEMPORAL TRANSFORMER WITH MULTI-SCALE CONVOLUTION
    use_temporal_transformer: bool = True
    transformer_num_heads: int = 8
    transformer_num_layers: int = 4  
    transformer_dropout: float = 0.1
    multi_scale_conv_kernels: tuple = (3, 5, 7, 9)  # Multi-scale kernel sizes
    
    # ✅ 3. LEARNABLE MULTI-THRESHOLD SPIKE ENCODING
    spike_encoding_type: str = "learnable_multi_threshold"  # 🚀 Enhanced encoding
    num_threshold_levels: int = 4  # 🚀 4 threshold levels (vs 1 in basic)
    threshold_learning_rate: float = 1e-4  # Learning rate for thresholds
    
    # ✅ 4. ENHANCED LIF WITH MEMORY AND REFRACTORY PERIOD
    snn_type: str = "enhanced_lif_memory"  # enhanced_lif_memory, basic_lif, vectorized_lif
    use_long_term_memory: bool = True
    memory_decay: float = 0.95
    refractory_period: int = 3  # time steps
    use_adaptive_threshold: bool = True
    
    # ✅ 5. MOMENTUM-BASED INFONCE WITH HARD NEGATIVE MINING
    use_momentum_infonce: bool = True
    momentum_coefficient: float = 0.999
    memory_bank_size: int = 4096  # 🚀 Large memory bank
    hard_negative_ratio: float = 0.3  # 30% hard negatives
    curriculum_difficulty: str = "exponential"  # linear, exponential, cosine
    
    # 🔬 TRAINING ENHANCEMENTS
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    use_gradient_checkpointing: bool = True
    
    # 📊 EVALUATION ENHANCEMENTS
    compute_roc_auc: bool = True
    compute_pr_auc: bool = True
    compute_f1_macro: bool = True
    save_confusion_matrix: bool = True
    
    # 💾 CHECKPOINTING ENHANCEMENTS
    save_best_model: bool = True
    checkpoint_frequency: int = 10  # every N epochs
    keep_n_checkpoints: int = 5
    
    # 🎯 TARGET PERFORMANCE
    target_roc_auc: float = 0.95  # Target ROC-AUC for early stopping
    target_false_alarm_rate: float = 1e-5  # 1 per 100k samples
    target_inference_time_ms: float = 100.0  # <100ms per 4s segment

    def validate(self) -> bool:
        """Validate complete enhanced configuration."""
        try:
            # Validate base config first
            if not super().validate():
                return False
            
            # Validate enhanced parameters
            assert 0 <= self.temporal_infonce_weight <= 1, "temporal_infonce_weight must be in [0,1]"
            assert 1 <= self.max_prediction_steps <= 50, "max_prediction_steps must be in [1,50]"
            assert 0 <= self.phase_preservation_weight <= 1, "phase_preservation_weight must be in [0,1]"
            assert self.edge_detection_thresholds > 0, "edge_detection_thresholds must be positive"
            
            assert 0 < self.surrogate_adaptation_rate < 1, "surrogate_adaptation_rate must be in (0,1)"
            assert 0 < self.memory_decay_constant <= 1, "memory_decay_constant must be in (0,1]"
            assert 0 < self.momentum_coefficient <= 1, "momentum_coefficient must be in (0,1]"
            assert 0 <= self.hard_negative_ratio <= 1, "hard_negative_ratio must be in [0,1]"
            
            assert self.num_threshold_levels > 0, "num_threshold_levels must be positive"
            assert self.threshold_learning_rate > 0, "threshold_learning_rate must be positive"
            assert self.memory_bank_size > 0, "memory_bank_size must be positive"
            
            assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
            assert self.target_roc_auc > 0.5, "target_roc_auc must be > 0.5"
            assert self.target_false_alarm_rate > 0, "target_false_alarm_rate must be positive"
            assert self.target_inference_time_ms > 0, "target_inference_time_ms must be positive"
            
            # Validate string choices
            valid_surrogate_schedules = ["adaptive", "curriculum", "constant"]
            assert self.surrogate_beta_schedule in valid_surrogate_schedules, \
                f"surrogate_beta_schedule must be in {valid_surrogate_schedules}"
            
            valid_encoding_types = ["learnable_multi_threshold", "temporal_contrast", "phase_preserving"]
            assert self.spike_encoding_type in valid_encoding_types, \
                f"spike_encoding_type must be in {valid_encoding_types}"
            
            valid_snn_types = ["enhanced_lif_memory", "basic_lif", "vectorized_lif"]
            assert self.snn_type in valid_snn_types, \
                f"snn_type must be in {valid_snn_types}"
            
            valid_curriculum = ["linear", "exponential", "cosine"]
            assert self.curriculum_difficulty in valid_curriculum, \
                f"curriculum_difficulty must be in {valid_curriculum}"
            
            return True
            
        except AssertionError as e:
            import logging
            logging.getLogger(__name__).error(f"CompleteEnhancedConfig validation failed: {e}")
            return False


# Export configuration classes
__all__ = [
    "TrainStateWithBatchStats",
    "CompleteEnhancedConfig"
]

