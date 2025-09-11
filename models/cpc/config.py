"""
Configuration classes for CPC (Contrastive Predictive Coding) components.

This module contains all configuration dataclasses extracted from
cpc_encoder.py for better modularity and maintainability.

Split from cpc_encoder.py for better organization.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class TemporalTransformerConfig:
    """Configuration for Temporal Transformer in Enhanced CPC."""
    num_heads: int = 8
    num_layers: int = 4
    dropout_rate: float = 0.1
    multi_scale_kernels: Tuple[int, ...] = (3, 5, 7, 9)
    use_layer_norm: bool = True
    use_residual_connections: bool = True
    attention_dropout: float = 0.1
    feed_forward_dim: int = 512


@dataclass
class RealCPCConfig:
    """Configuration for Real CPC Encoder with production settings."""
    
    # Architecture parameters
    latent_dim: int = 256
    context_dim: int = 128
    num_layers: int = 4
    hidden_dim: int = 512
    
    # Sequence parameters
    sequence_length: int = 2048  # GW strain sequence length
    context_length: int = 64     # Context window for prediction
    prediction_steps: int = 12   # Future steps to predict
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout_rate: float = 0.1
    
    # InfoNCE parameters
    temperature: float = 0.1
    negative_sampling_ratio: int = 16
    
    # Data augmentation
    use_temporal_augmentation: bool = True
    augmentation_strength: float = 0.1
    
    # Advanced features
    use_attention: bool = True
    use_residual_connections: bool = True
    use_layer_normalization: bool = True
    
    # Optimization
    gradient_clipping: float = 1.0
    use_mixed_precision: bool = True
    
    # Validation
    validation_frequency: int = 10  # Validate every N epochs
    early_stopping_patience: int = 20
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        try:
            assert self.latent_dim > 0, "latent_dim must be positive"
            assert self.context_dim > 0, "context_dim must be positive"
            assert self.num_layers > 0, "num_layers must be positive"
            assert self.hidden_dim > 0, "hidden_dim must be positive"
            
            assert self.sequence_length > self.context_length, \
                "sequence_length must be greater than context_length"
            assert self.prediction_steps > 0, "prediction_steps must be positive"
            
            assert 0 < self.learning_rate < 1, "learning_rate must be in (0, 1)"
            assert self.weight_decay >= 0, "weight_decay must be non-negative"
            assert 0 <= self.dropout_rate <= 1, "dropout_rate must be in [0, 1]"
            
            assert self.temperature > 0, "temperature must be positive"
            assert self.negative_sampling_ratio > 0, "negative_sampling_ratio must be positive"
            
            assert 0 <= self.augmentation_strength <= 1, "augmentation_strength must be in [0, 1]"
            assert self.gradient_clipping > 0, "gradient_clipping must be positive"
            
            return True
            
        except AssertionError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


@dataclass  
class ExperimentConfig:
    """Experiment configuration for enhanced CPC training."""
    
    # Model parameters
    latent_dim: int = 128
    context_length: int = 64
    num_prediction_steps: int = 12
    
    # Training parameters  
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    warmup_epochs: int = 10
    
    # Data parameters
    sequence_length: int = 2048
    sampling_rate: float = 4096.0  # Hz
    
    # Loss parameters
    temperature: float = 0.1
    negative_sampling_factor: int = 16
    
    # Regularization
    dropout_rate: float = 0.1
    weight_decay: float = 1e-4
    gradient_clipping: float = 1.0
    
    # Advanced features
    use_transformer: bool = True
    transformer_config: Optional[TemporalTransformerConfig] = None
    
    # Augmentation
    use_augmentation: bool = True
    noise_level: float = 0.01
    time_shift_max: int = 10
    
    # Optimization
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    scheduler: str = "cosine"  # "constant", "linear", "cosine"
    
    # Monitoring
    log_frequency: int = 10
    validation_frequency: int = 5
    save_frequency: int = 20
    
    # Hardware
    use_mixed_precision: bool = True
    memory_efficient: bool = True
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.transformer_config is None and self.use_transformer:
            self.transformer_config = TemporalTransformerConfig()
        
        # Validation
        self.validate()
    
    def validate(self):
        """Validate experiment configuration."""
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.context_length > 0, "context_length must be positive" 
        assert self.num_prediction_steps > 0, "num_prediction_steps must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert 0 < self.learning_rate < 1, "learning_rate must be in (0, 1)"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.sequence_length > self.context_length, \
            "sequence_length must be greater than context_length"
        assert self.temperature > 0, "temperature must be positive"
        assert 0 <= self.dropout_rate <= 1, "dropout_rate must be in [0, 1]"
        assert self.optimizer in ["adam", "adamw", "sgd"], \
            f"Unknown optimizer: {self.optimizer}"
        assert self.scheduler in ["constant", "linear", "cosine"], \
            f"Unknown scheduler: {self.scheduler}"


@dataclass
class CPCConfig:
    """Standard CPC configuration for basic encoder."""
    
    # Basic architecture
    latent_dim: int = 256
    encoder_layers: int = 3
    context_layers: int = 2
    
    # Sequence processing
    context_length: int = 32
    prediction_steps: int = 8
    
    # Training
    temperature: float = 0.1
    learning_rate: float = 5e-4
    
    # Regularization
    dropout_rate: float = 0.0
    weight_decay: float = 0.0
    
    def validate(self) -> bool:
        """Validate basic CPC configuration."""
        try:
            assert self.latent_dim > 0, "latent_dim must be positive"
            assert self.encoder_layers > 0, "encoder_layers must be positive"
            assert self.context_layers > 0, "context_layers must be positive"
            assert self.context_length > 0, "context_length must be positive"
            assert self.prediction_steps > 0, "prediction_steps must be positive"
            assert self.temperature > 0, "temperature must be positive"
            assert 0 < self.learning_rate < 1, "learning_rate must be in (0, 1)"
            assert 0 <= self.dropout_rate <= 1, "dropout_rate must be in [0, 1]"
            assert self.weight_decay >= 0, "weight_decay must be non-negative"
            return True
        except AssertionError as e:
            logger.error(f"CPC configuration validation failed: {e}")
            return False


# Export configuration classes
__all__ = [
    "TemporalTransformerConfig",
    "RealCPCConfig",
    "ExperimentConfig", 
    "CPCConfig"
]

