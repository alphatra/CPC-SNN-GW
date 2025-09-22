"""
Configuration classes for SNN (Spiking Neural Network) components.

This module contains configuration dataclasses extracted from
snn_classifier.py for better modularity.

Split from snn_classifier.py for better organization.
"""

from dataclasses import dataclass
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass  
class SNNConfig:
    """Configuration for SNN classifiers and layers."""
    
    # Architecture parameters
    hidden_sizes: Tuple[int, ...] = (256, 128, 64)  # ✅ 3 layers (256-128-64)
    num_classes: int = 2  # ✅ FIXED: Binary classification (noise vs signal)
    num_layers: int = 3   # ✅ Increased from 2 to 3
    
    # ✅ OPTIMIZED LIF neuron parameters (based on PDF 2508.00063v1)
    tau_mem: float = 15e-3      # Optimized membrane time constant (15ms)
    tau_syn: float = 8e-3       # Optimized synaptic time constant (8ms)  
    threshold: float = 0.55     # Optimized spike threshold
    adaptive_threshold: bool = True  # Enable adaptive threshold
    reset_potential: float = 0.0   # Reset potential after spike
    membrane_reset: str = "subtract"  # Better than hard reset
    
    # ✅ OPTIMIZED Training parameters (based on PDF research)
    surrogate_gradient_type: str = "hard_sigmoid"  # More stable than fast_sigmoid
    surrogate_beta: float = 4.0    # Optimized for GW detection (was 10.0)
    
    # ✅ ADVANCED SNN FEATURES: From PDF 2508.00063v1
    spike_regularization_target: float = 0.15  # Target 15% spike rate
    use_recurrent_connections: bool = False  # Keep simple for GW detection
    use_lateral_inhibition: bool = False  # Not needed for binary classification
    
    # ✅ TEMPORAL PROCESSING: Optimized for GW signals
    temporal_integration_window: int = 16  # Window for temporal integration
    spike_history_length: int = 8  # Track recent spike history
    
    # ✅ PERFORMANCE OPTIMIZATIONS: For real-time processing
    enable_jit_compilation: bool = True
    use_sparse_operations: bool = False  # Dense is faster for small networks
    memory_efficient_mode: bool = False  # Prioritize speed over memory
    
    # Regularization
    dropout_rate: float = 0.0       # ✅ Adaptive: decreases with depth
    use_layer_norm: bool = True     # ✅ NEW: LayerNorm after each layer
    
    # Architecture features
    use_residual_connections: bool = False
    use_input_projection: bool = True   # ✅ NEW: Project input to first layer
    
    # Performance
    use_vectorized_layers: bool = True
    enable_memory_optimization: bool = True
    
    def validate(self) -> bool:
        """Validate SNN configuration parameters."""
        try:
            assert len(self.hidden_sizes) > 0, "hidden_sizes must not be empty"
            assert all(h > 0 for h in self.hidden_sizes), "All hidden sizes must be positive"
            assert self.num_classes > 1, "num_classes must be > 1"
            assert self.num_layers > 0, "num_layers must be positive"
            
            assert self.tau_mem > 0, "tau_mem must be positive"
            assert self.tau_syn > 0, "tau_syn must be positive"  
            assert self.threshold > 0, "threshold must be positive"
            
            assert self.surrogate_beta > 0, "surrogate_beta must be positive"
            assert 0 <= self.dropout_rate <= 1, "dropout_rate must be in [0, 1]"
            
            # Check consistency
            if len(self.hidden_sizes) != self.num_layers:
                logger.warning(f"hidden_sizes length ({len(self.hidden_sizes)}) != num_layers ({self.num_layers})")
            
            return True
            
        except AssertionError as e:
            logger.error(f"SNN configuration validation failed: {e}")
            return False


@dataclass
class LIFConfig:
    """Configuration for individual LIF (Leaky Integrate-and-Fire) layers."""
    
    # Membrane dynamics
    tau_mem: float = 20e-3      # Membrane time constant
    tau_syn: float = 5e-3       # Synaptic time constant
    threshold: float = 1.0      # Spike threshold
    reset_potential: float = 0.0   # Reset potential
    
    # Initialization
    v_init: float = 0.0         # Initial membrane potential
    
    # Gradient approximation
    surrogate_gradient_type: str = "fast_sigmoid"
    surrogate_beta: float = 10.0
    
    # Architecture
    features: int = 128         # Number of neurons in layer
    use_bias: bool = True
    use_layer_norm: bool = True
    
    # Regularization 
    dropout_rate: float = 0.0
    membrane_noise: float = 0.0  # Membrane noise level
    
    def validate(self) -> bool:
        """Validate LIF layer configuration."""
        try:
            assert self.tau_mem > 0, "tau_mem must be positive"
            assert self.tau_syn > 0, "tau_syn must be positive"
            assert self.threshold > 0, "threshold must be positive"
            assert self.features > 0, "features must be positive"
            assert self.surrogate_beta > 0, "surrogate_beta must be positive"
            assert 0 <= self.dropout_rate <= 1, "dropout_rate must be in [0, 1]"
            assert self.membrane_noise >= 0, "membrane_noise must be non-negative"
            
            return True
            
        except AssertionError as e:
            logger.error(f"LIF configuration validation failed: {e}")
            return False


@dataclass
class EnhancedSNNConfig(SNNConfig):
    """Enhanced configuration for advanced SNN classifiers."""
    
    # Enhanced architecture
    use_attention: bool = False
    attention_heads: int = 4
    
    # Advanced LIF features
    use_adaptive_threshold: bool = False
    threshold_adaptation_rate: float = 0.01
    
    # Memory features
    use_long_term_memory: bool = True
    memory_decay: float = 0.95
    
    # Spike statistics  
    target_spike_rate: float = 0.1  # Target average spike rate
    spike_rate_regularization: float = 0.01
    
    # Advanced training
    use_curriculum_learning: bool = False
    curriculum_schedule: str = "linear"  # "linear", "exponential", "cosine"
    
    def validate(self) -> bool:
        """Validate enhanced SNN configuration."""
        # First validate base config
        if not super().validate():
            return False
        
        try:
            assert self.attention_heads > 0, "attention_heads must be positive"
            assert self.threshold_adaptation_rate > 0, "threshold_adaptation_rate must be positive"
            assert 0 < self.memory_decay <= 1, "memory_decay must be in (0, 1]"
            assert 0 < self.target_spike_rate < 1, "target_spike_rate must be in (0, 1)"
            assert self.spike_rate_regularization >= 0, "spike_rate_regularization must be non-negative"
            assert self.curriculum_schedule in ["linear", "exponential", "cosine"], \
                f"Unknown curriculum_schedule: {self.curriculum_schedule}"
            
            return True
            
        except AssertionError as e:
            logger.error(f"Enhanced SNN configuration validation failed: {e}")
            return False


# Export configuration classes
__all__ = [
    "SNNConfig",
    "LIFConfig", 
    "EnhancedSNNConfig"
]

