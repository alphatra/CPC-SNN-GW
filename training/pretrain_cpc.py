"""
CPC Pretraining: Self-Supervised Representation Learning

Clean implementation of Contrastive Predictive Coding pretraining:
- Self-supervised learning on unlabeled gravitational wave data
- InfoNCE contrastive loss for temporal prediction
- Optimized for Apple Silicon with JAX/Metal backend
- Professional logging and monitoring
- Production-ready checkpointing
"""

import logging
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

# Import base trainer and utilities
from .base_trainer import TrainerBase, TrainingConfig
from .training_metrics import create_training_metrics

# Import models and data
from models.cpc.core import CPCEncoder
from models.cpc.losses import enhanced_info_nce_loss
from data.gw_synthetic_generator import ContinuousGWGenerator

logger = logging.getLogger(__name__)


@dataclass
class CPCPretrainConfig(TrainingConfig):
    """Configuration for CPC pretraining."""
    
    # CPC-specific parameters
    latent_dim: int = 256
    context_length: int = 64
    prediction_steps: int = 12
    temperature: float = 0.1
    
    # Data parameters
    signal_duration: float = 4.0
    num_pretraining_signals: int = 1000
    include_noise_ratio: float = 0.3
    
    # Training optimization
    warmup_steps: int = 1000
    use_cosine_schedule: bool = True


class CPCPretrainer(TrainerBase):
    """
    CPC Pretrainer for self-supervised representation learning.
    
    Features:
    - Self-supervised contrastive learning
    - InfoNCE loss with temperature scaling
    - Flexible context/prediction setup
    - Professional training pipeline
    """
    
    def __init__(self, config: CPCPretrainConfig):
        super().__init__(config)
        self.config: CPCPretrainConfig = config
        
        # Initialize data generator
        from data.gw_signal_params import SignalConfiguration
        
        signal_config = SignalConfiguration(
            base_frequency=50.0,
            freq_range=(20.0, 500.0),
            duration=config.signal_duration
        )
        
        self.continuous_generator = ContinuousGWGenerator(
            config=signal_config,
            output_dir=str(self.directories['output'] / 'continuous_gw_cache')
        )
        
        logger.info("Initialized CPCPretrainer for self-supervised learning")
    
    def create_model(self):
        """Create CPC encoder for pretraining."""
        return CPCEncoder(latent_dim=self.config.latent_dim)
    
    def create_train_state(self, model, sample_input):
        """Create training state with CPC-optimized scheduler."""
        key = jax.random.PRNGKey(42)
        params = model.init(key, sample_input)
        
        # Learning rate schedule for CPC pretraining
        if self.config.use_cosine_schedule:
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                decay_steps=self.config.num_epochs * 100,  # Estimate
                end_value=0.0
            )
        else:
            schedule = self.config.learning_rate
        
        optimizer = optax.adamw(
            learning_rate=schedule,
            weight_decay=self.config.weight_decay
        )
        
        # Add gradient clipping
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.gradient_clipping),
            optimizer
        )
        
        return train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer
        )
    
    def generate_pretraining_data(self, key: jnp.ndarray) -> Dict:
        """Generate mixed data for CPC pretraining."""
        logger.info("Generating CPC pretraining dataset...")
        
        # Generate signals with noise for robust representations
        dataset = self.continuous_generator.generate_training_dataset(
            num_signals=self.config.num_pretraining_signals,
            signal_duration=self.config.signal_duration,
            include_noise_only=True  # Include pure noise for robustness
        )
        
        # For CPC pretraining, we use all data (signals + noise)
        # Labels not used in self-supervised learning
        pretraining_data = {
            'data': dataset['data'],
            'metadata': dataset.get('metadata', [])
        }
        
        logger.info(f"Pretraining dataset: {pretraining_data['data'].shape}")
        return pretraining_data
    
    def train_step(self, train_state, batch):
        """CPC training step with InfoNCE loss."""
        x = batch  # No labels needed for self-supervised learning
        
        def loss_fn(params):
            # Encode sequences
            latents = train_state.apply_fn(params, x)
            
            # Create context and target sequences for contrastive learning
            context_len = self.config.context_length
            if latents.shape[1] <= context_len:
                # If sequence too short, use first half as context
                context_len = latents.shape[1] // 2
            
            context = latents[:, :context_len]  # First part
            targets = latents[:, context_len:context_len+self.config.prediction_steps]  # Next steps
            
            # Ensure we have targets
            if targets.shape[1] == 0:
                targets = latents[:, -1:]  # Use last step as target
            
            # InfoNCE contrastive loss
            loss = enhanced_info_nce_loss(
                context, targets, 
                temperature=self.config.temperature
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
    
    def eval_step(self, train_state, batch):
        """CPC evaluation step."""
        x = batch
        
        # Forward pass - same as training but no gradients
        latents = train_state.apply_fn(train_state.params, x)
        
        # Compute contrastive loss
        context_len = self.config.context_length
        if latents.shape[1] <= context_len:
            context_len = latents.shape[1] // 2
        
        context = latents[:, :context_len]
        targets = latents[:, context_len:context_len+self.config.prediction_steps]
        
        if targets.shape[1] == 0:
            targets = latents[:, -1:]
        
        loss = enhanced_info_nce_loss(
            context, targets,
            temperature=self.config.temperature
        )
        
        metrics = create_training_metrics(
            step=train_state.step,
            epoch=0,
            loss=float(loss)
        )
        
        return metrics
    
    def run_pretraining(self, key: jnp.ndarray = None) -> Dict:
        """Run complete CPC pretraining pipeline."""
        if key is None:
            key = jax.random.PRNGKey(42)
        
        logger.info("Starting CPC pretraining pipeline...")
        
        # Generate pretraining data
        dataset = self.generate_pretraining_data(key)
        
        # Split data for validation
        split_idx = int(len(dataset['data']) * 0.9)  # 90% train, 10% val
        train_data = dataset['data'][:split_idx]
        val_data = dataset['data'][split_idx:]
        
        logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
        
        # Create model and training state
        model = self.create_model()
        sample_input = train_data[:1]
        self.train_state = self.create_train_state(model, sample_input)
        
        # Training loop
        best_val_loss = float('inf')
        training_history = []
        
        for epoch in range(self.config.num_epochs):
            # Training
            epoch_train_metrics = []
            num_batches = len(train_data) // self.config.batch_size
            
            # Shuffle training data
            key, subkey = jax.random.split(key)
            indices = jax.random.permutation(subkey, len(train_data))
            shuffled_data = train_data[indices]
            
            for i in range(num_batches):
                start_idx = i * self.config.batch_size
                end_idx = start_idx + self.config.batch_size
                
                batch = shuffled_data[start_idx:end_idx]
                self.train_state, metrics = self.train_step(self.train_state, batch)
                epoch_train_metrics.append(metrics)
            
            # Validation
            epoch_val_metrics = []
            val_batches = len(val_data) // self.config.batch_size
            
            for i in range(val_batches):
                start_idx = i * self.config.batch_size
                end_idx = start_idx + self.config.batch_size
                
                batch = val_data[start_idx:end_idx]
                val_metrics = self.eval_step(self.train_state, batch)
                epoch_val_metrics.append(val_metrics)
            
            # Compute epoch averages
            avg_train_loss = float(jnp.mean(jnp.array([m.loss for m in epoch_train_metrics])))
            avg_val_loss = float(jnp.mean(jnp.array([m.loss for m in epoch_val_metrics])))
            
            # Update best validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            
            # Log and save history
            training_history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        
        logger.info("CPC pretraining completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        return {
            'model_state': self.train_state,
            'training_history': training_history,
            'best_val_loss': best_val_loss,
            'config': self.config
        }


def create_cpc_pretrainer(config: Optional[CPCPretrainConfig] = None) -> CPCPretrainer:
    """Factory function to create CPC pretrainer."""
    if config is None:
        config = CPCPretrainConfig()
    
    return CPCPretrainer(config)


def run_cpc_pretraining_experiment():
    """Run CPC pretraining experiment."""
    logger.info("🚀 Starting CPC Pretraining Experiment")
    
    config = CPCPretrainConfig(
        num_epochs=50,
        batch_size=1,  # ✅ MEMORY FIX: Ultra-small batch for GPU memory constraints
        learning_rate=1e-3,
        latent_dim=256,
        num_pretraining_signals=500,
        context_length=32,
        prediction_steps=8
    )
    
    pretrainer = create_cpc_pretrainer(config)
    results = pretrainer.run_pretraining()
    
    logger.info("✅ CPC pretraining experiment completed")
    logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
    
    return results 