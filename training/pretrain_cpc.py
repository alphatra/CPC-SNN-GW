#!/usr/bin/env python3
"""
CPC Encoder Pretraining Script

Self-supervised contrastive pretraining dla gravitational wave detection.
Implements InfoNCE loss with gradient accumulation dla Apple Silicon optimization.

Usage:
    python pretrain_cpc.py --config config.yaml --num_steps 100000
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import orbax.checkpoint as ocp
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass, asdict
import yaml
import wandb

from ..models.cpc_encoder import CPCEncoder, info_nce_loss
from ..data.gw_download import ProductionGWOSCDownloader, AdvancedDataPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class CPCTrainingConfig:
    """Configuration dla CPC pretraining."""
    # Architecture
    latent_dim: int = 256
    downsample_factor: int = 16
    conv_channels: Tuple[int, ...] = (32, 64, 128)
    
    # Training  
    num_steps: int = 100_000
    batch_size: int = 16
    accumulation_steps: int = 4  # Gradient accumulation dla Metal limitations
    learning_rate: float = 1e-3
    warmup_steps: int = 5_000
    use_float16_matmul: bool = True  # Can disable on Apple Silicon if problematic
    
    # InfoNCE Loss
    context_length: int = 12  # Context windows dla prediction
    prediction_length: int = 4  # Future windows to predict
    num_negatives: int = 128
    temperature: float = 0.1
    
    # Data
    segment_duration: float = 4.0  # seconds
    sample_rate: int = 4096
    detectors: List[str] = None  # Will default to ['H1', 'L1']
    
    # Monitoring
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000
    
    # Paths
    output_dir: str = "experiments/cpc_pretraining"
    wandb_project: str = "ligo-cpc-snn"
    
    def __post_init__(self):
        if self.detectors is None:
            self.detectors = ['H1', 'L1']


class CPCTrainingState:
    """Training state dla CPC pretraining."""
    
    def __init__(self, config: CPCTrainingConfig):
        self.config = config
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Enable mixed precision for better performance
        jax.config.update("jax_enable_x64", False)
        
        # Configure matmul precision based on config and platform
        if config.use_float16_matmul:
            platform = jax.default_backend()
            if platform == "cpu" and "arm64" in str(jax.devices()[0]):
                logger.warning(
                    "Detected Apple Silicon (ARM64). float16 matmul may cause gradient clamping. "
                    "Consider setting use_float16_matmul=False if training is unstable."
                )
            jax.config.update("jax_default_matmul_precision", "float16")
        else:
            jax.config.update("jax_default_matmul_precision", "float32")
        
        # Initialize model
        self.model = CPCEncoder(
            latent_dim=config.latent_dim,
            conv_channels=config.conv_channels
        )
        
        # Initialize optimizer with warmup schedule
        self.scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=config.warmup_steps,
            decay_steps=max(config.num_steps - config.warmup_steps, 1)
        )
        
        # Use MultiSteps for gradient accumulation
        inner_optimizer = optax.adam(learning_rate=self.scheduler)
        self.optimizer = optax.MultiSteps(
            inner_optimizer, 
            every_k_schedule=config.accumulation_steps
        )
        
        # Initialize data pipeline
        self.downloader = ProductionGWOSCDownloader()
        self.preprocessor = AdvancedDataPreprocessor(
            sample_rate=config.sample_rate,
            apply_whitening=True
        )
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'learning_rate': [],
            'gradient_norm': [],
            'processing_time': [],
            'examples_per_sec': []
        }
        
    def initialize_params(self, key: jnp.ndarray, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Initialize model parameters."""
        dummy_input = jnp.zeros((1,) + input_shape)
        return self.model.init(key, dummy_input)
        
    def create_training_segments(self, strain_data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Create context-target pairs dla contrastive learning.
        
        Args:
            strain_data: Input strain timeseries [batch, time]
            
        Returns:
            context_segments: Context windows [batch, context_length, features]
            target_segments: Target windows dla prediction [batch, prediction_length, features]
        """
        batch_size, seq_len = strain_data.shape
        
        # Encode full sequence
        encoded = self.model.apply(self.params, strain_data)  # [batch, downsampled_time, latent_dim]
        _, encoded_len, latent_dim = encoded.shape
        
        # Create sliding windows
        total_length = self.config.context_length + self.config.prediction_length
        
        if encoded_len < total_length:
            # Pad if sequence too short
            pad_length = total_length - encoded_len
            encoded = jnp.pad(encoded, ((0, 0), (0, pad_length), (0, 0)))
            encoded_len = total_length
            
        # ✅ Generuj losowy indeks startowy dla każdej próbki w batchu
        max_start = encoded_len - total_length
        if max_start <= 0:
            start_indices = jnp.zeros((batch_size,), dtype=jnp.int32)
        else:
            start_indices = jax.random.randint(
                jax.random.PRNGKey(self.step), 
                (batch_size,), # Kształt (batch_size,)
                0, 
                max_start + 1
            )
        
        # ✅ Użyj jax.vmap do wydajnego wycinania okien dla każdej próbki
        def slice_windows(sequence, start):
            context = jax.lax.dynamic_slice_in_dim(sequence, start, self.config.context_length, axis=0)
            target = jax.lax.dynamic_slice_in_dim(sequence, start + self.config.context_length, self.config.prediction_length, axis=0)
            return context, target
        
        # vmap po wymiarze batcha (oś 0)
        context_segments, target_segments = jax.vmap(slice_windows)(encoded, start_indices)
        
        return context_segments, target_segments


def create_train_step(model, optimizer):
    """Create train step function with model and optimizer closure."""
    
    @jax.jit
    def train_step(state: Dict[str, Any], 
                  strain_batch: jnp.ndarray,
                  key: jnp.ndarray) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Single training step z gradient accumulation.
        
        Args:
            state: Training state dict {params, opt_state}
            strain_batch: Batch of strain data [batch, time]
            key: Random key dla training
            
        Returns:
            Updated state and metrics
        """
        params, opt_state = state['params'], state['opt_state']
        
        def loss_fn(params, strain_data, key):
            # Forward pass przez CPC encoder
            encoded = model.apply(params, strain_data)
            
            # Create context-target pairs with random windows
            batch_size, seq_len, latent_dim = encoded.shape
            context_length = 12
            prediction_length = 4
            total_length = context_length + prediction_length
            
            # ✅ Use random starting indices for each sample
            if seq_len < total_length:
                # Pad if too short
                pad_length = total_length - seq_len
                encoded = jnp.pad(encoded, ((0, 0), (0, pad_length), (0, 0)))
                seq_len = total_length
            
            max_start = seq_len - total_length
            if max_start <= 0:
                start_indices = jnp.zeros((batch_size,), dtype=jnp.int32)
            else:
                start_indices = jax.random.randint(key, (batch_size,), 0, max_start + 1)
            
            # ✅ Use vmap for efficient window slicing
            def slice_windows(sequence, start):
                context = jax.lax.dynamic_slice_in_dim(sequence, start, context_length, axis=0)
                target = jax.lax.dynamic_slice_in_dim(sequence, start + context_length, prediction_length, axis=0)
                return context, target
            
            context, target = jax.vmap(slice_windows)(encoded, start_indices)
            
            # InfoNCE loss
            return info_nce_loss(context, target, temperature=0.1)
        
        # Compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(params, strain_batch, key)
        
        # Gradient norm dla monitoring
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        updated_state = {'params': params, 'opt_state': opt_state}
        metrics = {'loss': loss, 'grad_norm': grad_norm}
        
        return updated_state, metrics
    
    return train_step


class CPCPretrainer:
    """Main pretraining orchestrator dla CPC encoder."""
    
    def __init__(self, config: CPCTrainingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize training state
        self.training_state = CPCTrainingState(config)
        
        # Setup checkpointing
        self.checkpointer = ocp.StandardCheckpointer()
        
        # Setup W&B logging
        if config.wandb_project:
            wandb.init(
                project=config.wandb_project,
                config=asdict(config),
                name=f"cpc_pretraining_{int(time.time())}"
            )
            
    def _setup_logging(self):
        """Setup file and console logging."""
        log_file = self.output_dir / "training.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def generate_training_data(self) -> List[jnp.ndarray]:
        """
        Generate training dataset from historical GW events.
        
        Returns:
            List of preprocessed strain segments
        """
        logger.info("Generating training dataset from GWOSC...")
        
        # Historical GW events dla training
        training_events = [
            # Major detections with clear signals
            ('H1', 1126259446, 4.0),  # GW150914
            ('L1', 1126259446, 4.0),  # GW150914
            ('H1', 1128678900, 4.0),  # GW151012
            ('L1', 1128678900, 4.0),  # GW151012  
            ('H1', 1135136350, 4.0),  # GW151226
            ('L1', 1135136350, 4.0),  # GW151226
            ('H1', 1167559936, 4.0),  # GW170104
            ('L1', 1167559936, 4.0),  # GW170104
            ('H1', 1180922494, 4.0),  # GW170608
            ('L1', 1180922494, 4.0),  # GW170608
        ]
        
        # Download data in batches
        logger.info(f"Downloading {len(training_events)} training segments...")
        
        strain_segments = self.training_state.downloader.fetch_batch(
            training_events, max_workers=4
        )
        
        # Preprocess all segments
        processed_segments = []
        successful = 0
        
        for i, strain_data in enumerate(strain_segments):
            if strain_data is not None:
                try:
                    result = self.training_state.preprocessor.process(strain_data)
                    
                    if result.quality.is_valid and result.quality.snr_estimate > 5.0:
                        processed_segments.append(result.strain_data)
                        successful += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to process segment {i}: {e}")
                    
        logger.info(f"Successfully processed {successful}/{len(training_events)} segments")
        
        return processed_segments
        
    def train(self):
        """Main training loop dla CPC pretraining."""
        logger.info("🚀 Starting CPC Pretraining")
        logger.info("=" * 60)
        logger.info(f"Config: {asdict(self.config)}")
        
        # Generate training data
        training_data = self.generate_training_data()
        
        if len(training_data) < 4:
            raise ValueError(f"Insufficient training data: {len(training_data)} segments")
            
        # Initialize model parameters
        key = jax.random.PRNGKey(42)
        input_shape = (int(self.config.segment_duration * self.config.sample_rate),)
        
        params = self.training_state.initialize_params(key, input_shape)
        opt_state = self.training_state.optimizer.init(params)
        
        state = {'params': params, 'opt_state': opt_state}
        
        # Create train step function
        train_step = create_train_step(
            self.training_state.model, 
            self.training_state.optimizer
        )
        
        # Training loop
        logger.info(f"Training dla {self.config.num_steps} steps...")
        
        start_time = time.perf_counter()
        
        for step in range(self.config.num_steps):
            step_start = time.perf_counter()
            
            # Sample random batch from training data
            batch_indices = jax.random.choice(
                jax.random.PRNGKey(step), 
                len(training_data), 
                (self.config.batch_size,),
                replace=True
            )
            
            strain_batch = jnp.stack([training_data[i] for i in batch_indices])
            
            # Training step
            key = jax.random.PRNGKey(step)
            state, metrics = train_step(state, strain_batch, key)
            
            step_time = time.perf_counter() - step_start
            
            # Logging
            if step % self.config.log_every == 0:
                lr = self.training_state.scheduler(step)
                examples_per_sec = self.config.batch_size / step_time
                
                logger.info(
                    f"Step {step:6d}: loss={metrics['loss']:.4f}, "
                    f"grad_norm={metrics['grad_norm']:.4f}, "
                    f"lr={lr:.2e}, "
                    f"examples/sec={examples_per_sec:.1f}"
                )
                
                # W&B logging
                if wandb.run:
                    wandb.log({
                        'train/loss': metrics['loss'],
                        'train/grad_norm': metrics['grad_norm'],
                        'train/learning_rate': lr,
                        'train/examples_per_sec': examples_per_sec,
                        'step': step
                    })
                    
            # Checkpointing
            if step % self.config.save_every == 0 and step > 0:
                checkpoint_path = self.output_dir / f"checkpoint_{step}"
                
                self.checkpointer.save(
                    checkpoint_path,
                    {'params': state['params'], 'step': step}
                )
                
                logger.info(f"Saved checkpoint at step {step}")
                
        total_time = time.perf_counter() - start_time
        
        # Final save
        final_path = self.output_dir / "final_checkpoint"
        self.checkpointer.save(
            final_path,
            {'params': state['params'], 'step': self.config.num_steps}
        )
        
        logger.info("🎉 CPC Pretraining completed!")
        logger.info(f"Total training time: {total_time/3600:.2f} hours")
        logger.info(f"Final checkpoint saved to: {final_path}")
        
        if wandb.run:
            wandb.finish()


def main():
    """Command-line interface dla CPC pretraining."""
    parser = argparse.ArgumentParser(description="CPC Encoder Pretraining")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--num_steps", type=int, default=100_000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="experiments/cpc_pretraining", help="Output directory")
    
    args = parser.parse_args()
    
    # Load config
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = CPCTrainingConfig(**config_dict)
    else:
        config = CPCTrainingConfig()
        
    # Override z command line args
    if args.num_steps:
        config.num_steps = args.num_steps
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.output_dir:
        config.output_dir = args.output_dir
        
    # Initialize and run training
    pretrainer = CPCPretrainer(config)
    pretrainer.train()


if __name__ == "__main__":
    main() 