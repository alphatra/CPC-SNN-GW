#!/usr/bin/env python3

"""
Unified Training Loop for CPC+SNN+Spike Gravitational Wave Detection

A comprehensive training system that combines:
- Contrastive Predictive Coding (CPC) for self-supervised learning
- Spiking Neural Networks (SNN) for neuromorphic processing
- Spike Bridge for optimal encoding/decoding
- Memory Bank integration for production workflows

Features:
- Multi-stage training (CPC pre-training → SNN fine-tuning → Joint optimization)
- Memory-efficient batch processing
- Automatic hyperparameter scheduling
- Comprehensive logging and monitoring
- Apple Silicon / JAX Metal optimization
- Production-ready checkpointing
"""

import os
import sys
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
import matplotlib.pyplot as plt

# Orbax for checkpointing
try:
    import orbax.checkpoint as ocp
    ORBAX_AVAILABLE = True
except ImportError:
    ORBAX_AVAILABLE = False
    print("Warning: Orbax not available. Checkpointing will be disabled.")

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Project imports
from ..models.cpc_encoder import EnhancedCPCEncoder
from ..models.snn_classifier import SNNClassifier
from ..models.spike_bridge import SpikeBridge
from ..data.gw_download import GWDataDownloader
from ..data.continuous_gw_generator import ContinuousGWGenerator
from ..data.label_utils import normalize_labels, validate_labels, get_class_weights
from ..data.cache_manager import ProfessionalCacheManager

logger = logging.getLogger(__name__)


@dataclass
class UnifiedTrainingConfig:
    """Configuration for unified CPC+SNN+Spike training."""
    
    # Training stages
    enable_cpc_pretraining: bool = True
    enable_snn_finetuning: bool = True
    enable_joint_optimization: bool = True
    
    # Model architecture
    cpc_hidden_dim: int = 256
    cpc_num_layers: int = 4
    snn_hidden_sizes: List[int] = field(default_factory=lambda: [128, 64, 32])
    snn_time_steps: int = 16
    spike_encoding_method: str = "poisson"  # "poisson", "temporal_contrast", "rate"
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    cpc_epochs: int = 50
    snn_epochs: int = 30
    joint_epochs: int = 20
    warmup_epochs: int = 5
    
    # Data parameters
    sequence_length: int = 1024
    num_classes: int = 3
    train_split: float = 0.7
    val_split: float = 0.2
    test_split: float = 0.1
    
    # Optimization
    optimizer: str = "adamw"  # "adamw", "sgd", "adam"
    weight_decay: float = 1e-4
    gradient_clipping: float = 1.0
    lr_schedule: str = "cosine"  # "cosine", "linear", "constant"
    
    # Loss weights
    cpc_loss_weight: float = 1.0
    snn_loss_weight: float = 1.0
    spike_regularization_weight: float = 0.1
    
    # Memory and performance
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    max_memory_gb: float = 8.0
    
    # Monitoring
    log_interval: int = 10
    val_interval: int = 100
    save_interval: int = 500
    
    # Paths
    output_dir: str = "outputs/unified_training"
    checkpoint_dir: str = "checkpoints/unified"
    data_dir: str = "data/gw_training"
    
    # Experiment tracking
    project_name: str = "cpc-snn-gw"
    experiment_name: str = "unified_training"
    tags: List[str] = field(default_factory=lambda: ["cpc", "snn", "spike", "gw"])
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.sequence_length > 0, "Sequence length must be positive"
        assert self.num_classes > 0, "Number of classes must be positive"
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, \
            "Train/val/test splits must sum to 1.0"


class JAXDataLoader:
    """
    Efficient JAX-compatible data loader with shuffling and batching.
    
    Features:
    - Memory-efficient batch iteration
    - Shuffling with JAX random keys
    - Proper batch size handling
    - Drop last option for consistent batch sizes
    """
    
    def __init__(self, 
                 data: jnp.ndarray,
                 labels: jnp.ndarray,
                 batch_size: int,
                 shuffle: bool = True,
                 drop_last: bool = True,
                 seed: int = 42):
        """
        Initialize JAX data loader.
        
        Args:
            data: Input data array
            labels: Label array
            batch_size: Batch size
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
            seed: Random seed for shuffling
        """
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        
        # Validate input
        assert len(data) == len(labels), "Data and labels must have same length"
        assert batch_size > 0, "Batch size must be positive"
        
        self.num_samples = len(data)
        
        # Calculate number of batches
        if drop_last:
            self.num_batches = self.num_samples // batch_size
        else:
            self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        
        # Initialize indices
        self.indices = jnp.arange(self.num_samples)
        
        logger.info(f"JAXDataLoader initialized: {self.num_samples} samples, {self.num_batches} batches, batch_size={batch_size}")
    
    def __len__(self):
        """Return number of batches."""
        return self.num_batches
    
    def __iter__(self):
        """Iterate over batches."""
        return self._create_batches()
    
    def _create_batches(self):
        """Create batch iterator."""
        # Shuffle indices if requested
        if self.shuffle:
            key = jax.random.PRNGKey(self.seed)
            shuffled_indices = jax.random.permutation(key, self.indices)
            # Update seed for next epoch
            self.seed = (self.seed + 1) % 2**32
        else:
            shuffled_indices = self.indices
        
        # Create batches
        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            
            # Get batch indices
            batch_indices = shuffled_indices[start_idx:end_idx]
            
            # Extract batch data
            batch_data = self.data[batch_indices]
            batch_labels = self.labels[batch_indices]
            
            # Skip incomplete batches if drop_last is True
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            
            yield batch_data, batch_labels
    
    def get_batch(self, batch_idx: int):
        """Get specific batch by index."""
        if batch_idx >= self.num_batches:
            raise IndexError(f"Batch index {batch_idx} out of range (0-{self.num_batches-1})")
        
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        
        # Get batch data
        batch_data = self.data[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]
        
        return batch_data, batch_labels
    
    def reset_seed(self, seed: int):
        """Reset random seed."""
        self.seed = seed


class UnifiedTrainer:
    """
    Unified trainer for CPC+SNN+Spike gravitational wave detection.
    
    Implements a multi-stage training approach:
    1. CPC pre-training for representation learning
    2. SNN fine-tuning for neuromorphic processing
    3. Joint optimization for end-to-end performance
    """
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate schedule with mixed precision support."""
        
        # Setup mixed precision if enabled
        if self.config.use_mixed_precision:
            try:
                # Try to use bfloat16 for mixed precision
                jax.config.update('jax_default_dtype_policy', 'bfloat16')
                logger.info("Mixed precision enabled: Using bfloat16")
            except Exception as e:
                logger.warning(f"Failed to enable bfloat16, falling back to float16: {e}")
                try:
                    jax.config.update('jax_default_dtype_policy', 'float16')
                    logger.info("Mixed precision enabled: Using float16")
                except Exception as e2:
                    logger.warning(f"Failed to enable mixed precision: {e2}")
        
        # Calculate steps per epoch for train loader
        steps_per_epoch = self.train_loader.num_batches if self.train_loader else 1000  # fallback
        
        # Calculate total steps correctly
        total_steps = (self.config.cpc_epochs + self.config.snn_epochs + 
                      self.config.joint_epochs) * steps_per_epoch
        
        # Learning rate schedule
        if self.config.lr_schedule == "cosine":
            schedule = optax.cosine_decay_schedule(
                init_value=self.config.learning_rate,
                decay_steps=total_steps,
                alpha=0.1
            )
        elif self.config.lr_schedule == "linear":
            schedule = optax.linear_schedule(
                init_value=self.config.learning_rate,
                end_value=self.config.learning_rate * 0.1,
                transition_steps=total_steps
            )
        else:
            schedule = self.config.learning_rate
        
        # Optimizer with mixed precision support
        if self.config.optimizer == "adamw":
            optimizer = optax.adamw(
                learning_rate=schedule,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            optimizer = optax.sgd(
                learning_rate=schedule,
                momentum=0.9
            )
        else:
            optimizer = optax.adam(learning_rate=schedule)
        
        # Add mixed precision wrapper if enabled
        if self.config.use_mixed_precision:
            # Use optax mixed precision wrapper
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.gradient_clipping) if self.config.gradient_clipping > 0 else optax.identity(),
                optax.mixed_precision(optimizer, threshold=32768.0)  # Automatic loss scaling
            )
            logger.info("Applied mixed precision optimizer wrapper")
        else:
            # Add gradient clipping for regular precision
            if self.config.gradient_clipping > 0:
                optimizer = optax.chain(
                    optax.clip_by_global_norm(self.config.gradient_clipping),
                    optimizer
                )
        
        self.optimizer = optimizer
        logger.info(f"Optimizer setup: {self.config.optimizer} with {self.config.lr_schedule} schedule")
        logger.info(f"Total training steps: {total_steps} ({steps_per_epoch} per epoch)")
        logger.info(f"Mixed precision: {self.config.use_mixed_precision}")
    
    def setup_device_optimization(self):
        """Setup device-specific optimizations (Apple Silicon, TPU, etc.)."""
        
        # Check available devices
        devices = jax.devices()
        logger.info(f"Available JAX devices: {devices}")
        
        # Apple Silicon optimizations
        if any('METAL' in str(device) for device in devices):
            logger.info("Detected Apple Silicon GPU, optimizing for Metal backend")
            try:
                # Metal-specific optimizations
                jax.config.update('jax_platform_name', 'metal')
                
                # Optimize memory usage for Apple Silicon
                if hasattr(jax.config, 'jax_gpu_memory_fraction'):
                    jax.config.update('jax_gpu_memory_fraction', 0.8)
                
                logger.info("Apple Silicon GPU optimizations applied")
            except Exception as e:
                logger.warning(f"Failed to apply Apple Silicon optimizations: {e}")
        
        # TPU optimizations
        elif any('TPU' in str(device) for device in devices):
            logger.info("Detected TPU, optimizing for TPU backend")
            try:
                # TPU-specific optimizations
                jax.config.update('jax_platform_name', 'tpu')
                
                # Enable XLA optimizations for TPU
                jax.config.update('jax_enable_x64', False)  # Use 32-bit for TPU
                
                logger.info("TPU optimizations applied")
            except Exception as e:
                logger.warning(f"Failed to apply TPU optimizations: {e}")
        
        # CPU fallback
        else:
            logger.info("Using CPU backend")
            try:
                jax.config.update('jax_platform_name', 'cpu')
                
                # CPU optimizations
                import os
                os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=true')
                
                logger.info("CPU optimizations applied")
            except Exception as e:
                logger.warning(f"Failed to apply CPU optimizations: {e}")
    
    def __init__(self, config: UnifiedTrainingConfig):
        """
        Initialize unified trainer with device optimizations.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.step = 0
        self.epoch = 0
        self.best_val_accuracy = 0.0
        
        # Setup device optimizations early
        self.setup_device_optimization()
        
        # Setup directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.cpc_encoder = None
        self.snn_classifier = None
        self.spike_bridge = None
        
        # Training state
        self.train_state = None
        self.optimizer = None
        
        # Data
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
        # Initialize experiment tracking
        self.setup_experiment_tracking()
        
        logger.info(f"Initialized UnifiedTrainer with config: {config}")
    
    def setup_experiment_tracking(self):
        """Setup experiment tracking with W&B."""
        if WANDB_AVAILABLE:
            wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                config=self.config.__dict__,
                tags=self.config.tags,
                dir=str(self.output_dir)
            )
            logger.info("Initialized Weights & Biases experiment tracking")
        else:
            logger.warning("W&B not available, skipping experiment tracking")
    
    def setup_models(self, input_shape: Tuple[int, ...]):
        """
        Setup all model components.
        
        Args:
            input_shape: Shape of input data (batch_size, sequence_length, features)
        """
        logger.info("Setting up models...")
        
        # Initialize CPC encoder
        self.cpc_encoder = EnhancedCPCEncoder(
            hidden_dim=self.config.cpc_hidden_dim,
            num_layers=self.config.cpc_num_layers,
            dropout_rate=0.1
        )
        
        # Initialize SNN classifier
        self.snn_classifier = SNNClassifier(
            hidden_sizes=self.config.snn_hidden_sizes,
            num_classes=self.config.num_classes,
            time_steps=self.config.snn_time_steps
        )
        
        # Initialize Spike Bridge
        self.spike_bridge = SpikeBridge(
            input_dim=self.config.cpc_hidden_dim,
            time_steps=self.config.snn_time_steps,
            encoding_method=self.config.spike_encoding_method
        )
        
        # Initialize model parameters
        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        
        # CPC encoder parameters
        dummy_input = jnp.ones((1, self.config.sequence_length, input_shape[-1]))
        self.cpc_params = self.cpc_encoder.init(key1, dummy_input)
        
        # SNN classifier parameters
        dummy_spikes = jnp.ones((1, self.config.snn_time_steps, self.config.cpc_hidden_dim))
        self.snn_params = self.snn_classifier.init(key2, dummy_spikes)
        
        # Spike bridge parameters
        dummy_cpc_output = jnp.ones((1, self.config.cpc_hidden_dim))
        self.spike_params = self.spike_bridge.init(key3, dummy_cpc_output)
        
        logger.info("Models initialized successfully")
    
    def setup_data_loaders(self, dataset: Dict[str, Any]):
        """
        Setup efficient data loaders for training, validation, and testing.
        
        Args:
            dataset: Dataset dictionary with 'data', 'labels', and 'metadata'
        """
        logger.info("Setting up efficient data loaders...")
        
        # Validate and normalize labels
        labels = normalize_labels(dataset['labels'])
        validate_labels(labels, expected_classes=self.config.num_classes)
        
        # Convert to JAX arrays for efficiency
        data_array = jnp.array(dataset['data'])
        labels_array = jnp.array(labels)
        
        # Split data
        total_samples = len(data_array)
        train_end = int(total_samples * self.config.train_split)
        val_end = train_end + int(total_samples * self.config.val_split)
        
        # Create splits
        train_data = data_array[:train_end]
        train_labels = labels_array[:train_end]
        
        val_data = data_array[train_end:val_end]
        val_labels = labels_array[train_end:val_end]
        
        test_data = data_array[val_end:]
        test_labels = labels_array[val_end:]
        
        # Create efficient data loaders
        self.train_loader = JAXDataLoader(
            train_data, 
            train_labels, 
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        self.val_loader = JAXDataLoader(
            val_data,
            val_labels,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False
        )
        
        self.test_loader = JAXDataLoader(
            test_data,
            test_labels,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False
        )
        
        logger.info(f"Data loaders created: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        logger.info(f"Batch counts: train={self.train_loader.num_batches}, val={self.val_loader.num_batches}, test={self.test_loader.num_batches}")
    
    def create_train_state(self, params: Dict[str, Any]) -> train_state.TrainState:
        """Create training state."""
        return train_state.TrainState.create(
            apply_fn=None,  # Will be set per stage
            params=params,
            tx=self.optimizer
        )
    
    @jax.jit
    def cpc_loss_fn(self, params, batch_data, key):
        """
        Corrected CPC loss function with proper InfoNCE implementation.
        
        Args:
            params: CPC encoder parameters
            batch_data: Input batch data
            key: Random key for dropout
            
        Returns:
            InfoNCE contrastive loss
        """
        # Forward pass through CPC encoder
        representations = self.cpc_encoder.apply(params, batch_data, training=True, rngs={'dropout': key})
        
        # Use enhanced InfoNCE loss
        return self.enhanced_info_nce_loss(representations, temperature=0.1)
    
    @jax.jit
    def enhanced_info_nce_loss(self, representations, temperature=0.1, use_explicit_pairing=False):
        """
        Enhanced InfoNCE loss with proper positive/negative sampling.
        
        Args:
            representations: Batch of representations [batch_size, hidden_dim]
            temperature: Temperature parameter for scaling
            use_explicit_pairing: If True, use even/odd pairing. If False, use self-supervised mode
            
        Returns:
            InfoNCE loss
        """
        batch_size = representations.shape[0]
        
        # L2 normalize
        representations = representations / (jnp.linalg.norm(representations, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity matrix
        similarity_matrix = jnp.matmul(representations, representations.T) / temperature
        
        if use_explicit_pairing:
            # Explicit pairing mode - requires even batch size
            if batch_size % 2 != 0:
                # Fallback to self-supervised mode for odd batch sizes
                return self._self_supervised_info_nce_loss(representations, temperature)
            
            # Create positive pair mask for even/odd pairing
            positive_mask = jnp.zeros((batch_size, batch_size))
            for i in range(0, batch_size, 2):
                positive_mask = positive_mask.at[i, i+1].set(1)
                positive_mask = positive_mask.at[i+1, i].set(1)
            
            # Remove diagonal (self-similarity)
            similarity_matrix = similarity_matrix - jnp.eye(batch_size) * 1e9
            
            # Compute InfoNCE loss
            losses = []
            for i in range(batch_size):
                # Find positive sample
                positive_idx = jnp.where(positive_mask[i] == 1)[0]
                if len(positive_idx) > 0:
                    pos_sim = similarity_matrix[i, positive_idx[0]]
                    
                    # All similarities (positive + negatives)
                    all_sims = similarity_matrix[i]
                    
                    # InfoNCE loss: -log(exp(pos_sim) / sum(exp(all_sims)))
                    loss = -pos_sim + jnp.logsumexp(all_sims)
                    losses.append(loss)
            
            return jnp.mean(jnp.array(losses))
        else:
            # Self-supervised mode - use augmented views or temporal shifts
            return self._self_supervised_info_nce_loss(representations, temperature)
    
    @jax.jit
    def _self_supervised_info_nce_loss(self, representations, temperature=0.1):
        """
        Self-supervised InfoNCE loss using augmented views.
        
        Args:
            representations: Batch of representations [batch_size, hidden_dim]
            temperature: Temperature parameter for scaling
            
        Returns:
            InfoNCE loss
        """
        batch_size = representations.shape[0]
        
        # Create augmented views by adding small noise
        key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
        noise_scale = 0.1
        noise = jax.random.normal(key, representations.shape) * noise_scale
        augmented_representations = representations + noise
        
        # L2 normalize both original and augmented
        representations = representations / (jnp.linalg.norm(representations, axis=1, keepdims=True) + 1e-8)
        augmented_representations = augmented_representations / (jnp.linalg.norm(augmented_representations, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity matrices
        # Positive pairs: original vs augmented (same index)
        positive_sims = jnp.sum(representations * augmented_representations, axis=1) / temperature
        
        # Negative pairs: original vs all augmented (different indices)
        negative_sims = jnp.matmul(representations, augmented_representations.T) / temperature
        
        # Remove diagonal (positive pairs) from negatives
        negative_sims = negative_sims - jnp.eye(batch_size) * 1e9
        
        # InfoNCE loss for each sample
        losses = []
        for i in range(batch_size):
            pos_sim = positive_sims[i]
            neg_sims = negative_sims[i]
            
            # Combine positive and negative similarities
            all_sims = jnp.concatenate([pos_sim.reshape(1), neg_sims])
            
            # InfoNCE loss: -log(exp(pos_sim) / sum(exp(all_sims)))
            loss = -pos_sim + jnp.logsumexp(all_sims)
            losses.append(loss)
        
        return jnp.mean(jnp.array(losses))
    
    @jax.jit
    def _contrastive_augmentation(self, representations, key, augmentation_type="noise"):
        """
        Apply contrastive augmentation to representations.
        
        Args:
            representations: Input representations [batch_size, hidden_dim]
            key: JAX random key
            augmentation_type: Type of augmentation ("noise", "dropout", "mixup")
            
        Returns:
            Augmented representations
        """
        if augmentation_type == "noise":
            # Add Gaussian noise
            noise_scale = 0.1
            noise = jax.random.normal(key, representations.shape) * noise_scale
            return representations + noise
        
        elif augmentation_type == "dropout":
            # Apply dropout
            dropout_rate = 0.2
            mask = jax.random.bernoulli(key, 1 - dropout_rate, representations.shape)
            return representations * mask / (1 - dropout_rate)
        
        elif augmentation_type == "mixup":
            # Apply mixup between samples
            alpha = 0.2
            lam = jax.random.beta(key, alpha, alpha)
            
            # Shuffle indices for mixup
            indices = jax.random.permutation(key, jnp.arange(representations.shape[0]))
            shuffled_representations = representations[indices]
            
            return lam * representations + (1 - lam) * shuffled_representations
        
        else:
            return representations
    
    @jax.jit
    def snn_loss_fn(self, params, spike_data, labels):
        """SNN loss function."""
        # Forward pass through SNN classifier
        logits = self.snn_classifier.apply(params, spike_data, training=True)
        
        # Classification loss
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        
        return jnp.mean(loss)
    
    @jax.jit
    def joint_loss_fn(self, cpc_params, snn_params, spike_params, batch_data, labels, key):
        """Joint loss function for end-to-end training with improved spike regularization."""
        # Use gradient checkpointing for memory efficiency with long sequences
        if self.config.sequence_length >= 512:  # Enable for long sequences
            return self._joint_loss_fn_with_checkpointing(cpc_params, snn_params, spike_params, batch_data, labels, key)
        else:
            return self._joint_loss_fn_standard(cpc_params, snn_params, spike_params, batch_data, labels, key)
    
    @jax.jit
    def _joint_loss_fn_standard(self, cpc_params, snn_params, spike_params, batch_data, labels, key):
        """Standard joint loss function without checkpointing."""
        # Forward pass through entire pipeline
        # CPC encoder
        representations = self.cpc_encoder.apply(cpc_params, batch_data, training=True, rngs={'dropout': key})
        
        # Spike bridge
        spike_data = self.spike_bridge.apply(spike_params, representations)
        
        # SNN classifier
        logits = self.snn_classifier.apply(snn_params, spike_data, training=True)
        
        # Classification loss
        classification_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        
        # Improved spike regularization
        spike_reg = self._compute_spike_regularization(spike_data)
        
        total_loss = jnp.mean(classification_loss) + self.config.spike_regularization_weight * spike_reg
        
        return total_loss, {
            'classification_loss': jnp.mean(classification_loss), 
            'spike_reg': spike_reg,
            'spike_rate': jnp.mean(spike_data),
            'spike_sparsity': jnp.mean(spike_data == 0)
        }
    
    @jax.jit
    def _joint_loss_fn_with_checkpointing(self, cpc_params, snn_params, spike_params, batch_data, labels, key):
        """Joint loss function with gradient checkpointing for memory efficiency."""
        
        # Define checkpointed forward passes
        @jax.remat
        def cpc_forward(params, data, rng_key):
            """Checkpointed CPC forward pass."""
            return self.cpc_encoder.apply(params, data, training=True, rngs={'dropout': rng_key})
        
        @jax.remat
        def spike_forward(params, representations):
            """Checkpointed spike bridge forward pass."""
            return self.spike_bridge.apply(params, representations)
        
        @jax.remat
        def snn_forward(params, spike_data):
            """Checkpointed SNN forward pass."""
            return self.snn_classifier.apply(params, spike_data, training=True)
        
        # Forward pass with checkpointing
        representations = cpc_forward(cpc_params, batch_data, key)
        spike_data = spike_forward(spike_params, representations)
        logits = snn_forward(snn_params, spike_data)
        
        # Classification loss
        classification_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        
        # Improved spike regularization
        spike_reg = self._compute_spike_regularization(spike_data)
        
        total_loss = jnp.mean(classification_loss) + self.config.spike_regularization_weight * spike_reg
        
        return total_loss, {
            'classification_loss': jnp.mean(classification_loss), 
            'spike_reg': spike_reg,
            'spike_rate': jnp.mean(spike_data),
            'spike_sparsity': jnp.mean(spike_data == 0)
        }
    
    @jax.jit
    def _memory_efficient_forward(self, cpc_params, snn_params, spike_params, batch_data, key):
        """
        Memory-efficient forward pass with selective checkpointing.
        
        Args:
            cpc_params: CPC encoder parameters
            snn_params: SNN classifier parameters
            spike_params: Spike bridge parameters
            batch_data: Input batch data
            key: Random key
            
        Returns:
            (representations, spike_data, logits)
        """
        # Use different strategies based on sequence length
        if self.config.sequence_length >= 1024:
            # Very long sequences - checkpoint everything
            @jax.remat
            def full_forward(params_tuple, data, rng_key):
                cpc_p, snn_p, spike_p = params_tuple
                repr_data = self.cpc_encoder.apply(cpc_p, data, training=True, rngs={'dropout': rng_key})
                spike_data = self.spike_bridge.apply(spike_p, repr_data)
                logits = self.snn_classifier.apply(snn_p, spike_data, training=True)
                return repr_data, spike_data, logits
            
            return full_forward((cpc_params, snn_params, spike_params), batch_data, key)
        
        elif self.config.sequence_length >= 512:
            # Medium sequences - checkpoint CPC and SNN
            @jax.remat
            def cpc_forward(params, data, rng_key):
                return self.cpc_encoder.apply(params, data, training=True, rngs={'dropout': rng_key})
            
            @jax.remat
            def snn_forward(params, spike_data):
                return self.snn_classifier.apply(params, spike_data, training=True)
            
            representations = cpc_forward(cpc_params, batch_data, key)
            spike_data = self.spike_bridge.apply(spike_params, representations)
            logits = snn_forward(snn_params, spike_data)
            
            return representations, spike_data, logits
        
        else:
            # Short sequences - no checkpointing
            representations = self.cpc_encoder.apply(cpc_params, batch_data, training=True, rngs={'dropout': key})
            spike_data = self.spike_bridge.apply(spike_params, representations)
            logits = self.snn_classifier.apply(snn_params, spike_data, training=True)
            
            return representations, spike_data, logits
    
    @jax.jit
    def _compute_spike_regularization(self, spike_data):
        """
        Compute spike regularization with multiple strategies.
        
        Args:
            spike_data: Spike data [batch_size, time_steps, features]
            
        Returns:
            Spike regularization loss
        """
        # Strategy 1: L1 regularization (encourage sparsity)
        l1_reg = jnp.mean(jnp.abs(spike_data))
        
        # Strategy 2: Firing rate regularization (target specific rate)
        target_spike_rate = 0.1  # Target 10% firing rate
        spike_rate = jnp.mean(spike_data)
        rate_reg = jnp.abs(spike_rate - target_spike_rate)
        
        # Strategy 3: Temporal smoothness (discourage rapid changes)
        if spike_data.ndim >= 3:  # [batch, time, features]
            spike_diff = jnp.diff(spike_data, axis=1)  # Difference along time axis
            temporal_reg = jnp.mean(jnp.abs(spike_diff))
        else:
            temporal_reg = 0.0
        
        # Strategy 4: Population diversity (encourage different neurons to fire)
        # Variance across neurons at each time step
        if spike_data.ndim >= 3:
            neuron_variance = jnp.var(spike_data, axis=2)  # Variance across features
            diversity_reg = -jnp.mean(neuron_variance)  # Negative because we want to maximize variance
        else:
            diversity_reg = 0.0
        
        # Combine regularization terms
        total_reg = (
            0.5 * l1_reg +           # L1 sparsity
            0.3 * rate_reg +         # Rate targeting
            0.1 * temporal_reg +     # Temporal smoothness
            0.1 * diversity_reg      # Population diversity
        )
        
        return total_reg
    
    @jax.jit
    def _spike_statistics(self, spike_data):
        """
        Compute detailed spike statistics for monitoring.
        
        Args:
            spike_data: Spike data [batch_size, time_steps, features]
            
        Returns:
            Dictionary of spike statistics
        """
        # Basic statistics
        mean_rate = jnp.mean(spike_data)
        std_rate = jnp.std(spike_data)
        sparsity = jnp.mean(spike_data == 0)
        
        # Temporal statistics
        if spike_data.ndim >= 3:
            # Mean firing rate per time step
            temporal_rates = jnp.mean(spike_data, axis=(0, 2))  # Average over batch and features
            temporal_std = jnp.std(temporal_rates)
            
            # Mean firing rate per neuron
            neuron_rates = jnp.mean(spike_data, axis=(0, 1))  # Average over batch and time
            neuron_std = jnp.std(neuron_rates)
            
            # Burst detection (consecutive spikes)
            spike_binary = spike_data > 0
            bursts = jnp.sum(spike_binary[:, :-1] * spike_binary[:, 1:])  # Consecutive spikes
            burst_ratio = bursts / jnp.sum(spike_binary)
        else:
            temporal_std = 0.0
            neuron_std = 0.0
            burst_ratio = 0.0
        
        return {
            'mean_rate': mean_rate,
            'std_rate': std_rate,
            'sparsity': sparsity,
            'temporal_std': temporal_std,
            'neuron_std': neuron_std,
            'burst_ratio': burst_ratio
        }
    
    # Training methods with improved data loaders and error handling
    def train_cpc_stage(self):
        """Stage 1: CPC pre-training for representation learning."""
        if not self.config.enable_cpc_pretraining:
            logger.info("CPC pre-training disabled, skipping...")
            return
        
        logger.info("Starting CPC pre-training stage...")
        
        # Setup CPC training state
        cpc_state = self.create_train_state(self.cpc_params)
        
        # Training loop with error handling
        for epoch in range(self.config.cpc_epochs):
            epoch_loss = 0.0
            num_batches = 0
            failed_batches = 0
            
            # Reset data loader for new epoch
            self.train_loader.reset_seed(self.step + epoch)
            
            # Training batches using efficient data loader
            for batch_data, batch_labels in self.train_loader:
                try:
                    # Check for NaN/Inf in input data
                    if jnp.any(jnp.isnan(batch_data)) or jnp.any(jnp.isinf(batch_data)):
                        logger.warning(f"NaN/Inf detected in batch, skipping...")
                        failed_batches += 1
                        continue
                    
                    # Training step
                    key = jax.random.PRNGKey(self.step)
                    loss, grads = jax.value_and_grad(self.cpc_loss_fn)(cpc_state.params, batch_data, key)
                    
                    # Check for NaN/Inf in loss and gradients
                    if jnp.isnan(loss) or jnp.isinf(loss):
                        logger.warning(f"NaN/Inf loss detected at step {self.step}, skipping batch...")
                        failed_batches += 1
                        continue
                    
                    # Check gradients for NaN/Inf
                    grad_finite = jax.tree_map(lambda x: jnp.all(jnp.isfinite(x)), grads)
                    if not jax.tree_util.tree_all(grad_finite):
                        logger.warning(f"NaN/Inf gradients detected at step {self.step}, skipping batch...")
                        failed_batches += 1
                        continue
                    
                    # Update parameters
                    cpc_state = cpc_state.apply_gradients(grads=grads)
                    
                    epoch_loss += loss
                    num_batches += 1
                    self.step += 1
                    
                    # Log progress
                    if self.step % self.config.log_interval == 0:
                        logger.info(f"CPC Step {self.step}: Loss = {loss:.4f}")
                        
                        if WANDB_AVAILABLE:
                            wandb.log({
                                'cpc_loss': loss,
                                'cpc_epoch': epoch,
                                'step': self.step,
                                'failed_batches': failed_batches
                            })
                
                except Exception as e:
                    logger.error(f"Error in CPC training step {self.step}: {str(e)}")
                    failed_batches += 1
                    continue
            
            # Epoch summary
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                logger.info(f"CPC Epoch {epoch}: Average Loss = {avg_loss:.4f}, Failed batches = {failed_batches}")
            else:
                logger.error(f"CPC Epoch {epoch}: No successful batches processed!")
                break
            
            # Save checkpoint
            if epoch % 10 == 0:
                try:
                    self.save_checkpoint(cpc_state.params, f"cpc_epoch_{epoch}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint for epoch {epoch}: {str(e)}")
        
        # Update CPC parameters
        self.cpc_params = cpc_state.params
        logger.info("CPC pre-training completed")
    
    def train_snn_stage(self):
        """Stage 2: SNN fine-tuning on encoded representations."""
        if not self.config.enable_snn_finetuning:
            logger.info("SNN fine-tuning disabled, skipping...")
            return
        
        logger.info("Starting SNN fine-tuning stage...")
        
        # Setup SNN training state
        snn_state = self.create_train_state(self.snn_params)
        
        # Training loop with error handling
        for epoch in range(self.config.snn_epochs):
            epoch_loss = 0.0
            num_batches = 0
            failed_batches = 0
            
            # Reset data loader for new epoch
            self.train_loader.reset_seed(self.step + epoch)
            
            # Training batches using efficient data loader
            for batch_data, batch_labels in self.train_loader:
                try:
                    # Check for NaN/Inf in input data
                    if jnp.any(jnp.isnan(batch_data)) or jnp.any(jnp.isinf(batch_data)):
                        logger.warning(f"NaN/Inf detected in batch, skipping...")
                        failed_batches += 1
                        continue
                    
                    # Encode to spikes using frozen CPC encoder
                    representations = self.cpc_encoder.apply(self.cpc_params, batch_data, training=False)
                    
                    # Check representations for NaN/Inf
                    if jnp.any(jnp.isnan(representations)) or jnp.any(jnp.isinf(representations)):
                        logger.warning(f"NaN/Inf in CPC representations, skipping...")
                        failed_batches += 1
                        continue
                    
                    spike_data = self.spike_bridge.apply(self.spike_params, representations)
                    
                    # Check spike data for NaN/Inf
                    if jnp.any(jnp.isnan(spike_data)) or jnp.any(jnp.isinf(spike_data)):
                        logger.warning(f"NaN/Inf in spike data, skipping...")
                        failed_batches += 1
                        continue
                    
                    # Training step
                    loss, grads = jax.value_and_grad(self.snn_loss_fn)(snn_state.params, spike_data, batch_labels)
                    
                    # Check for NaN/Inf in loss and gradients
                    if jnp.isnan(loss) or jnp.isinf(loss):
                        logger.warning(f"NaN/Inf loss detected at step {self.step}, skipping batch...")
                        failed_batches += 1
                        continue
                    
                    # Check gradients for NaN/Inf
                    grad_finite = jax.tree_map(lambda x: jnp.all(jnp.isfinite(x)), grads)
                    if not jax.tree_util.tree_all(grad_finite):
                        logger.warning(f"NaN/Inf gradients detected at step {self.step}, skipping batch...")
                        failed_batches += 1
                        continue
                    
                    # Update parameters
                    snn_state = snn_state.apply_gradients(grads=grads)
                    
                    epoch_loss += loss
                    num_batches += 1
                    self.step += 1
                    
                    # Log progress
                    if self.step % self.config.log_interval == 0:
                        logger.info(f"SNN Step {self.step}: Loss = {loss:.4f}")
                        
                        if WANDB_AVAILABLE:
                            wandb.log({
                                'snn_loss': loss,
                                'snn_epoch': epoch,
                                'step': self.step,
                                'failed_batches': failed_batches
                            })
                
                except Exception as e:
                    logger.error(f"Error in SNN training step {self.step}: {str(e)}")
                    failed_batches += 1
                    continue
            
            # Epoch summary
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                logger.info(f"SNN Epoch {epoch}: Average Loss = {avg_loss:.4f}, Failed batches = {failed_batches}")
            else:
                logger.error(f"SNN Epoch {epoch}: No successful batches processed!")
                break
            
            # Save checkpoint
            if epoch % 10 == 0:
                try:
                    self.save_checkpoint(snn_state.params, f"snn_epoch_{epoch}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint for epoch {epoch}: {str(e)}")
        
        # Update SNN parameters
        self.snn_params = snn_state.params
        logger.info("SNN fine-tuning completed")
    
    def train_joint_stage(self):
        """Stage 3: Joint optimization of entire pipeline."""
        if not self.config.enable_joint_optimization:
            logger.info("Joint optimization disabled, skipping...")
            return
        
        logger.info("Starting joint optimization stage...")
        
        # Create combined parameters
        joint_params = {
            'cpc': self.cpc_params,
            'snn': self.snn_params,
            'spike': self.spike_params
        }
        
        # Setup joint training state
        joint_state = self.create_train_state(joint_params)
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10  # Early stopping patience
        
        # Training loop with error handling
        for epoch in range(self.config.joint_epochs):
            epoch_loss = 0.0
            num_batches = 0
            failed_batches = 0
            
            # Reset data loader for new epoch
            self.train_loader.reset_seed(self.step + epoch)
            
            # Training batches using efficient data loader
            for batch_data, batch_labels in self.train_loader:
                try:
                    # Check for NaN/Inf in input data
                    if jnp.any(jnp.isnan(batch_data)) or jnp.any(jnp.isinf(batch_data)):
                        logger.warning(f"NaN/Inf detected in batch, skipping...")
                        failed_batches += 1
                        continue
                    
                    # Training step
                    key = jax.random.PRNGKey(self.step)
                    (loss, metrics), grads = jax.value_and_grad(self.joint_loss_fn, has_aux=True)(
                        joint_state.params['cpc'],
                        joint_state.params['snn'],
                        joint_state.params['spike'],
                        batch_data,
                        batch_labels,
                        key
                    )
                    
                    # Check for NaN/Inf in loss and gradients
                    if jnp.isnan(loss) or jnp.isinf(loss):
                        logger.warning(f"NaN/Inf loss detected at step {self.step}, skipping batch...")
                        failed_batches += 1
                        continue
                    
                    # Check gradients for NaN/Inf
                    grad_finite = jax.tree_map(lambda x: jnp.all(jnp.isfinite(x)), grads)
                    if not jax.tree_util.tree_all(grad_finite):
                        logger.warning(f"NaN/Inf gradients detected at step {self.step}, skipping batch...")
                        failed_batches += 1
                        continue
                    
                    # Update parameters
                    joint_state = joint_state.apply_gradients(grads=grads)
                    
                    epoch_loss += loss
                    num_batches += 1
                    self.step += 1
                    
                    # Log progress
                    if self.step % self.config.log_interval == 0:
                        logger.info(f"Joint Step {self.step}: Loss = {loss:.4f}, Class Loss = {metrics['classification_loss']:.4f}")
                        
                        if WANDB_AVAILABLE:
                            wandb.log({
                                'joint_loss': loss,
                                'classification_loss': metrics['classification_loss'],
                                'spike_regularization': metrics['spike_reg'],
                                'joint_epoch': epoch,
                                'step': self.step,
                                'failed_batches': failed_batches
                            })
                
                except Exception as e:
                    logger.error(f"Error in joint training step {self.step}: {str(e)}")
                    failed_batches += 1
                    continue
            
            # Epoch summary
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                logger.info(f"Joint Epoch {epoch}: Average Loss = {avg_loss:.4f}, Failed batches = {failed_batches}")
            else:
                logger.error(f"Joint Epoch {epoch}: No successful batches processed!")
                break
            
            # Validation and enhanced early stopping
            if epoch % 5 == 0:
                try:
                    val_metrics = self.validate_model(joint_state.params, compute_loss=True)
                    
                    logger.info(f"Validation - Epoch {epoch}: Accuracy = {val_metrics['accuracy']:.4f}, Loss = {val_metrics.get('loss', 'N/A')}")
                    
                    # Enhanced W&B logging
                    self.enhanced_wandb_logging(val_metrics, epoch, stage="validation")
                    
                    # Enhanced early stopping
                    should_stop, stop_reason = self.enhanced_early_stopping(val_metrics, epoch, patience=patience)
                    
                    # Check if we have a new best model
                    if hasattr(self, 'early_stopping_state'):
                        if val_metrics['accuracy'] > self.early_stopping_state['best_accuracy']:
                            self.best_val_accuracy = val_metrics['accuracy']
                            self.save_checkpoint(joint_state.params, "best_model")
                            logger.info(f"New best model saved with accuracy: {val_metrics['accuracy']:.4f}")
                    
                    # Check early stopping
                    if should_stop:
                        logger.info(f"Early stopping triggered: {stop_reason}")
                        break
                
                except Exception as e:
                    logger.error(f"Error during validation at epoch {epoch}: {str(e)}")
                    
            # Enhanced training metrics logging
            if epoch % 2 == 0:  # Log training metrics every 2 epochs
                try:
                    train_metrics = {
                        'loss': avg_loss,
                        'failed_batches': failed_batches,
                        'successful_batches': num_batches,
                        'epoch': epoch
                    }
                    self.enhanced_wandb_logging(train_metrics, epoch, stage="training")
                
                except Exception as e:
                    logger.error(f"Error logging training metrics for epoch {epoch}: {str(e)}")
            
            # Save checkpoint
            if epoch % 10 == 0:
                try:
                    self.save_checkpoint(joint_state.params, f"joint_epoch_{epoch}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint for epoch {epoch}: {str(e)}")
        
        # Update final parameters
        self.cpc_params = joint_state.params['cpc']
        self.snn_params = joint_state.params['snn']
        self.spike_params = joint_state.params['spike']
        
        logger.info("Joint optimization completed")
    
    @jax.jit
    def validate_model(self, params, compute_loss=True):
        """
        Validate model performance with optional loss computation.
        
        Args:
            params: Model parameters
            compute_loss: Whether to compute validation loss
            
        Returns:
            Dictionary with validation metrics
        """
        correct = 0
        total = 0
        total_loss = 0.0
        num_batches = 0
        
        # Detailed metrics
        all_predictions = []
        all_labels = []
        all_spike_stats = []
        
        for batch_data, batch_labels in self.val_loader:
            try:
                # Forward pass
                representations = self.cpc_encoder.apply(params['cpc'], batch_data, training=False)
                spike_data = self.spike_bridge.apply(params['spike'], representations)
                logits = self.snn_classifier.apply(params['snn'], spike_data, training=False)
                
                # Predictions
                predictions = jnp.argmax(logits, axis=1)
                
                # Accuracy
                correct += jnp.sum(predictions == batch_labels)
                total += len(batch_labels)
                
                # Store for detailed analysis
                all_predictions.extend(predictions.tolist())
                all_labels.extend(batch_labels.tolist())
                
                # Compute loss if requested
                if compute_loss:
                    # Classification loss
                    classification_loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_labels)
                    
                    # Spike regularization
                    spike_reg = self._compute_spike_regularization(spike_data)
                    
                    # Total loss
                    batch_loss = jnp.mean(classification_loss) + self.config.spike_regularization_weight * spike_reg
                    total_loss += batch_loss
                    
                    # Spike statistics
                    spike_stats = self._spike_statistics(spike_data)
                    all_spike_stats.append(spike_stats)
                    
                num_batches += 1
                
            except Exception as e:
                logger.warning(f"Error in validation batch: {str(e)}")
                continue
        
        # Compute metrics
        accuracy = float(correct / total) if total > 0 else 0.0
        avg_loss = float(total_loss / num_batches) if num_batches > 0 and compute_loss else None
        
        # Detailed metrics
        validation_metrics = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'num_samples': total,
            'num_batches': num_batches
        }
        
        # Add spike statistics if available
        if all_spike_stats:
            # Average spike statistics across batches
            avg_spike_stats = {}
            for key in all_spike_stats[0].keys():
                avg_spike_stats[f'spike_{key}'] = float(jnp.mean(jnp.array([stats[key] for stats in all_spike_stats])))
            validation_metrics.update(avg_spike_stats)
        
        # Per-class metrics
        if len(all_predictions) > 0 and len(all_labels) > 0:
            # Convert to numpy for easier computation
            pred_array = jnp.array(all_predictions)
            label_array = jnp.array(all_labels)
            
            # Per-class accuracy
            for class_idx in range(self.config.num_classes):
                class_mask = label_array == class_idx
                if jnp.sum(class_mask) > 0:
                    class_acc = jnp.sum(pred_array[class_mask] == label_array[class_mask]) / jnp.sum(class_mask)
                    validation_metrics[f'class_{class_idx}_accuracy'] = float(class_acc)
                    validation_metrics[f'class_{class_idx}_samples'] = int(jnp.sum(class_mask))
        
        return validation_metrics
    
    def enhanced_early_stopping(self, val_metrics, epoch, patience=10):
        """
        Enhanced early stopping with multiple criteria.
        
        Args:
            val_metrics: Validation metrics dictionary
            epoch: Current epoch
            patience: Patience for early stopping
            
        Returns:
            (should_stop, reason)
        """
        if not hasattr(self, 'early_stopping_state'):
            self.early_stopping_state = {
                'best_loss': float('inf'),
                'best_accuracy': 0.0,
                'patience_counter': 0,
                'best_epoch': 0,
                'no_improvement_epochs': 0,
                'loss_history': [],
                'accuracy_history': []
            }
        
        state = self.early_stopping_state
        
        # Update history
        if val_metrics.get('loss') is not None:
            state['loss_history'].append(val_metrics['loss'])
        state['accuracy_history'].append(val_metrics['accuracy'])
        
        # Check improvement
        improved = False
        improvement_reason = ""
        
        # Primary criterion: validation loss (if available)
        if val_metrics.get('loss') is not None:
            if val_metrics['loss'] < state['best_loss']:
                state['best_loss'] = val_metrics['loss']
                state['best_epoch'] = epoch
                improved = True
                improvement_reason = f"loss improved to {val_metrics['loss']:.4f}"
        
        # Secondary criterion: validation accuracy
        if val_metrics['accuracy'] > state['best_accuracy']:
            state['best_accuracy'] = val_metrics['accuracy']
            if not improved:  # Only set if loss didn't improve
                state['best_epoch'] = epoch
                improved = True
                improvement_reason = f"accuracy improved to {val_metrics['accuracy']:.4f}"
        
        # Update counters
        if improved:
            state['patience_counter'] = 0
            state['no_improvement_epochs'] = 0
        else:
            state['patience_counter'] += 1
            state['no_improvement_epochs'] += 1
        
        # Check stopping criteria
        should_stop = False
        stop_reason = ""
        
        if state['patience_counter'] >= patience:
            should_stop = True
            stop_reason = f"No improvement for {patience} epochs"
        
        # Additional stopping criteria
        elif len(state['loss_history']) >= 5:
            # Check if loss is increasing consistently
            recent_losses = state['loss_history'][-5:]
            if len(recent_losses) == 5 and all(recent_losses[i] <= recent_losses[i+1] for i in range(4)):
                should_stop = True
                stop_reason = "Loss increasing consistently"
        
        elif len(state['accuracy_history']) >= 10:
            # Check if accuracy has plateaued
            recent_accs = state['accuracy_history'][-10:]
            if len(recent_accs) == 10 and max(recent_accs) - min(recent_accs) < 0.001:
                should_stop = True
                stop_reason = "Accuracy plateaued"
        
        # Log early stopping state
        logger.info(f"Early stopping - Epoch {epoch}: {improvement_reason if improved else 'No improvement'}")
        logger.info(f"  Best loss: {state['best_loss']:.4f} (epoch {state['best_epoch']})")
        logger.info(f"  Best accuracy: {state['best_accuracy']:.4f}")
        logger.info(f"  Patience: {state['patience_counter']}/{patience}")
        
        return should_stop, stop_reason
    
    def enhanced_wandb_logging(self, metrics, epoch, stage="training"):
        """
        Enhanced W&B logging with histograms and detailed metrics.
        
        Args:
            metrics: Metrics dictionary
            epoch: Current epoch
            stage: Training stage ("training", "validation", "test")
        """
        if not WANDB_AVAILABLE:
            return
        
        # Basic metrics
        log_dict = {
            f'{stage}_epoch': epoch,
            'step': self.step
        }
        
        # Add all metrics with stage prefix
        for key, value in metrics.items():
            if isinstance(value, (int, float, jnp.ndarray)):
                if isinstance(value, jnp.ndarray):
                    if value.ndim == 0:  # Scalar
                        log_dict[f'{stage}_{key}'] = float(value)
                    else:  # Array - log as histogram
                        log_dict[f'{stage}_{key}_hist'] = wandb.Histogram(value)
                        log_dict[f'{stage}_{key}_mean'] = float(jnp.mean(value))
                        log_dict[f'{stage}_{key}_std'] = float(jnp.std(value))
                else:
                    log_dict[f'{stage}_{key}'] = value
        
        # Add early stopping metrics if available
        if hasattr(self, 'early_stopping_state') and stage == "validation":
            state = self.early_stopping_state
            log_dict.update({
                'early_stopping_patience': state['patience_counter'],
                'early_stopping_best_loss': state['best_loss'],
                'early_stopping_best_accuracy': state['best_accuracy'],
                'early_stopping_best_epoch': state['best_epoch']
            })
        
        # Log to W&B
        wandb.log(log_dict)
        
        # Log learning rate if available
        if hasattr(self, 'train_state') and hasattr(self.train_state, 'opt_state'):
            try:
                # Get current learning rate from train state - different optimizers store it differently
                if hasattr(self.train_state.opt_state, 'hyperparams') and 'learning_rate' in self.train_state.opt_state.hyperparams:
                    lr = self.train_state.opt_state.hyperparams['learning_rate']
                    # If it's a schedule, evaluate it
                    if callable(lr):
                        lr = lr(self.train_state.step)
                    wandb.log({f'{stage}_learning_rate': float(lr)})
                else:
                    # Fallback to config learning rate
                    wandb.log({f'{stage}_learning_rate': self.config.learning_rate})
            except Exception as e:
                logger.warning(f"Could not log learning rate: {e}")
                pass  # Ignore if can't get learning rate
    
    def save_checkpoint(self, params, name: str):
        """Save model checkpoint using Orbax."""
        if not ORBAX_AVAILABLE:
            logger.warning("Orbax not available. Skipping checkpoint save.")
            return
            
        checkpoint_path = self.checkpoint_dir / name
        
        # Create Orbax checkpointer
        checkpointer = ocp.StandardCheckpointer()
        
        # Save using Orbax
        try:
            checkpointer.save(checkpoint_path, params)
            logger.info(f"Saved Orbax checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Fallback to numpy if Orbax fails
            checkpoint_path_npz = self.checkpoint_dir / f"{name}.npz"
            params_np = jax.tree_map(np.array, params)
            np.savez(checkpoint_path_npz, **params_np)
            logger.info(f"Saved fallback numpy checkpoint: {checkpoint_path_npz}")
    
    def load_checkpoint(self, name: str):
        """Load model checkpoint using Orbax."""
        if not ORBAX_AVAILABLE:
            logger.warning("Orbax not available. Attempting numpy fallback.")
            return self._load_checkpoint_numpy(name)
            
        checkpoint_path = self.checkpoint_dir / name
        
        # Try Orbax first
        if checkpoint_path.exists():
            try:
                checkpointer = ocp.StandardCheckpointer()
                params = checkpointer.restore(checkpoint_path)
                logger.info(f"Loaded Orbax checkpoint: {checkpoint_path}")
                return params
            except Exception as e:
                logger.warning(f"Failed to load Orbax checkpoint: {e}. Trying numpy fallback.")
                return self._load_checkpoint_numpy(name)
        else:
            # Try numpy fallback
            return self._load_checkpoint_numpy(name)
    
    def _load_checkpoint_numpy(self, name: str):
        """Fallback method to load numpy checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{name}.npz"
        
        if checkpoint_path.exists():
            # Load
            params_np = np.load(checkpoint_path)
            
            # Convert back to JAX
            params = jax.tree_map(jnp.array, dict(params_np))
            
            logger.info(f"Loaded numpy checkpoint: {checkpoint_path}")
            return params
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None
    
    def train(self, dataset: Dict[str, Any]):
        """
        Main training function.
        
        Args:
            dataset: Training dataset
        """
        logger.info("Starting unified training...")
        
        # Setup
        self.setup_data_loaders(dataset)
        input_shape = dataset['data'].shape
        self.setup_models(input_shape)
        self.setup_optimizer()
        
        # Training stages
        stage_start = time.time()
        
        # Stage 1: CPC pre-training
        self.train_cpc_stage()
        
        # Stage 2: SNN fine-tuning
        self.train_snn_stage()
        
        # Stage 3: Joint optimization
        self.train_joint_stage()
        
        # Final evaluation
        final_accuracy = self.evaluate_test_set()
        
        total_time = time.time() - stage_start
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Final test accuracy: {final_accuracy:.4f}")
        
        if WANDB_AVAILABLE:
            wandb.log({
                'final_test_accuracy': final_accuracy,
                'total_training_time': total_time
            })
            wandb.finish()
        
        return {
            'final_accuracy': final_accuracy,
            'training_time': total_time,
            'best_val_accuracy': self.best_val_accuracy
        }
    
    @jax.jit
    def evaluate_test_set(self):
        """Evaluate on test set."""
        logger.info("Evaluating on test set...")
        
        # Load best model
        best_params = self.load_checkpoint("best_model")
        if best_params is None:
            logger.warning("No best model found, using current parameters")
            best_params = {
                'cpc': self.cpc_params,
                'snn': self.snn_params,
                'spike': self.spike_params
            }
        
        correct = 0
        total = 0
        
        for batch_data, batch_labels in self.test_loader:
            try:
                # Forward pass
                representations = self.cpc_encoder.apply(best_params['cpc'], batch_data, training=False)
                spike_data = self.spike_bridge.apply(best_params['spike'], representations)
                logits = self.snn_classifier.apply(best_params['snn'], spike_data, training=False)
                
                # Predictions
                predictions = jnp.argmax(logits, axis=1)
                
                # Accuracy
                correct += jnp.sum(predictions == batch_labels)
                total += len(batch_labels)
                
            except Exception as e:
                logger.warning(f"Error in test batch: {str(e)}")
                continue
        
        accuracy = float(correct / total) if total > 0 else 0.0
        logger.info(f"Test set accuracy: {accuracy:.4f}")
        
        return accuracy


def create_unified_trainer(config_path: Optional[str] = None) -> UnifiedTrainer:
    """
    Create unified trainer from configuration.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Configured UnifiedTrainer instance
    """
    if config_path and Path(config_path).exists():
        # Load configuration from file
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = UnifiedTrainingConfig(**config_dict)
    else:
        # Use default configuration
        config = UnifiedTrainingConfig()
    
    return UnifiedTrainer(config)


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified CPC+SNN+Spike Training")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--data", type=str, help="Path to training data")
    parser.add_argument("--output", type=str, default="outputs/unified_training", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create trainer
    trainer = create_unified_trainer(args.config)
    
    # Load or generate data
    if args.data:
        # Load existing dataset
        dataset = np.load(args.data, allow_pickle=True).item()
    else:
        # Generate synthetic dataset
        logger.info("Generating synthetic dataset...")
        generator = ContinuousGWGenerator()
        dataset = generator.generate_training_dataset(
            num_signals=1000,
            signal_duration=4.0
        )
    
    # Train model
    results = trainer.train(dataset)
    
    # Save results
    results_path = Path(args.output) / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Training completed. Results saved to {results_path}")


if __name__ == "__main__":
    main()