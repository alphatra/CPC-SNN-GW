"""
LHC-Optimized Training Command

This module implements training with LHC-optimized SNN parameters based on
Dillon et al. "Anomaly detection with spiking neural networks for LHC physics"
arXiv:2508.00063v1 [hep-ph] 31 Jul 2025

Key optimizations:
- Time steps: 5-10 (vs current 32)
- Threshold: 1.2 (vs current 0.55) 
- Beta decay: 0.9
- Multi-step processing methodology
- Binary latent space optimization

Usage:
    python cli.py train-lhc --config configs/lhc_optimized.yaml
"""

import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from training.base.config import TrainingConfig
from training.base.trainer import CPCSNNTrainer
from models.snn.lhc_optimized import create_lhc_optimized_snn
from utils.config_loader import load_config

logger = logging.getLogger(__name__)


class LHCOptimizedTrainingConfig(TrainingConfig):
    """Extended training config for LHC-optimized parameters."""
    
    # ✅ LHC PARAMETERS: Override defaults with LHC optimal values
    spike_time_steps: int = 5  # LHC optimal (vs 32)
    spike_threshold: float = 1.2  # LHC optimal (vs 0.55)
    spike_beta: float = 0.9  # LHC membrane decay
    
    # ✅ LHC ARCHITECTURE: Small efficient architecture
    snn_hidden_sizes: tuple = (64, 32)  # Smaller than current (128, 128, 64)
    snn_latent_dim: int = 4  # Binary latent space
    
    # ✅ LHC TRAINING: Based on paper methodology
    learning_rate: float = 0.001  # LHC optimal (Adam default)
    batch_size: int = 8  # Small batches for stability
    
    # ✅ LHC OPTIMIZATION: Conservative settings
    adaptive_grad_clip_threshold: float = 1.0  # Higher for stability
    global_grad_clip_norm: float = 1.0
    per_module_grad_clip: bool = False  # Simplified
    
    # ✅ LHC CPC: Simplified parameters
    cpc_temperature: float = 0.20  # Higher for stability
    cpc_aux_weight: float = 0.10  # Moderate influence
    cpc_prediction_steps: int = 4  # Reduced for efficiency


class LHCOptimizedTrainer(CPCSNNTrainer):
    """
    Trainer optimized for LHC parameters and methodology.
    
    Key differences from base trainer:
    - Uses LHC-optimized SNN architecture
    - Multi-step processing with same input
    - Binary latent space optimization
    - Conservative gradient clipping
    """
    
    def __init__(self, config: LHCOptimizedTrainingConfig):
        """Initialize LHC-optimized trainer."""
        super().__init__(config)
        self.lhc_config = config
        
        logger.info("🚀 Initializing LHC-Optimized Trainer")
        logger.info(f"   ⚡ Time steps: {config.spike_time_steps} (LHC optimal)")
        logger.info(f"   🎯 Threshold: {config.spike_threshold} (LHC optimal)")
        logger.info(f"   🔥 Beta decay: {config.spike_beta} (LHC optimal)")
        logger.info(f"   🏗️ Architecture: {config.snn_hidden_sizes}")
        logger.info(f"   📊 Latent dim: {config.snn_latent_dim} (binary space)")
    
    def create_model(self):
        """Create LHC-optimized model architecture."""
        logger.info("🏗️ Creating LHC-optimized model...")
        
        # ✅ LHC SNN: Use optimized architecture
        snn_model = create_lhc_optimized_snn(
            num_classes=self.config.num_classes,
            input_features=1024,  # Assuming downsampled GW data
            time_steps=self.lhc_config.spike_time_steps,
            threshold=self.lhc_config.spike_threshold,
            latent_dim=self.lhc_config.snn_latent_dim
        )
        
        # ✅ LHC CPC: Simplified CPC encoder (if needed)
        # For now, use SNN directly for classification
        # TODO: Integrate with CPC encoder when ready
        
        return snn_model
    
    def create_optimizer(self) -> optax.GradientTransformation:
        """Create LHC-optimized optimizer."""
        logger.info("⚙️ Creating LHC-optimized optimizer...")
        
        # ✅ LHC OPTIMIZER: Adam with paper parameters
        optimizer_chain = []
        
        # Adam optimizer with LHC parameters
        optimizer_chain.append(
            optax.adam(
                learning_rate=self.lhc_config.learning_rate,
                b1=0.9,   # LHC paper default
                b2=0.999, # LHC paper default
                eps=1e-8
            )
        )
        
        # ✅ LHC CLIPPING: Conservative gradient clipping
        if self.lhc_config.adaptive_grad_clip_threshold > 0:
            optimizer_chain.append(
                optax.clip_by_global_norm(self.lhc_config.global_grad_clip_norm)
            )
        
        return optax.chain(*optimizer_chain)
    
    def loss_fn(self, params, batch, rng_key, training: bool = True) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        LHC-optimized loss function.
        
        Based on LHC paper methodology:
        - MSE reconstruction loss (if using autoencoder)
        - Binary cross-entropy for classification
        - Minimal regularization
        """
        signals, labels = batch
        
        # Forward pass through LHC-optimized model
        logits, snn_metrics = self.model.apply(
            params, signals, training=training, rngs={'dropout': rng_key}
        )
        
        # ✅ LHC LOSS: Binary cross-entropy (like LHC paper)
        labels_onehot = jax.nn.one_hot(labels, self.config.num_classes)
        classification_loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
        
        # ✅ LHC REGULARIZATION: Minimal spike regularization
        spike_rate = snn_metrics.get('avg_spike_rate', 0.0)
        target_spike_rate = 0.15  # Target 15% spike rate (LHC-style)
        spike_reg_loss = 0.001 * jnp.abs(spike_rate - target_spike_rate)
        
        # Total loss
        total_loss = classification_loss + spike_reg_loss
        
        # Metrics for monitoring
        metrics = {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'spike_reg_loss': spike_reg_loss,
            'spike_rate_mean': spike_rate,
            'active_neuron_ratio': snn_metrics.get('avg_active_ratio', 0.0),
            'time_steps_used': snn_metrics.get('time_steps', self.lhc_config.spike_time_steps),
            'threshold_used': snn_metrics.get('threshold', self.lhc_config.spike_threshold),
            'binary_latent_configs': snn_metrics.get('latent_configurations', 0)
        }
        
        return total_loss, metrics
    
    def train_step(self, state: train_state.TrainState, batch, rng_key) -> Tuple[train_state.TrainState, Dict[str, Any]]:
        """LHC-optimized training step."""
        
        def loss_and_grad_fn(params):
            loss, metrics = self.loss_fn(params, batch, rng_key, training=True)
            return loss, metrics
        
        # Compute gradients
        (loss, metrics), grads = jax.value_and_grad(loss_and_grad_fn, has_aux=True)(state.params)
        
        # ✅ LHC GRADIENT MONITORING: Track gradient norms
        grad_norm = optax.global_norm(grads)
        metrics['grad_norm_total'] = grad_norm
        
        # Update parameters
        state = state.apply_gradients(grads=grads)
        
        return state, metrics
    
    def eval_step(self, state: train_state.TrainState, batch, rng_key) -> Dict[str, Any]:
        """LHC-optimized evaluation step."""
        loss, metrics = self.loss_fn(state.params, batch, rng_key, training=False)
        
        # Calculate accuracy
        signals, labels = batch
        logits, _ = self.model.apply(state.params, signals, training=False)
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == labels)
        
        metrics['accuracy'] = accuracy
        metrics['eval_loss'] = loss
        
        return metrics


def create_lhc_optimized_trainer(config_path: Optional[str] = None) -> LHCOptimizedTrainer:
    """
    Factory function to create LHC-optimized trainer.
    
    Args:
        config_path: Path to LHC-optimized config file
        
    Returns:
        Configured LHC-optimized trainer
    """
    if config_path is None:
        config_path = str(project_root / "configs" / "lhc_optimized.yaml")
    
    # Load configuration
    config_dict = load_config(config_path)
    
    # Create LHC-optimized training config
    training_config = LHCOptimizedTrainingConfig(
        # System settings
        num_classes=config_dict['model']['snn']['num_classes'],
        batch_size=config_dict['training']['batch_size'],
        learning_rate=config_dict['training']['learning_rate'],
        num_epochs=config_dict['training']['num_epochs'],
        
        # LHC-optimized SNN settings
        spike_time_steps=config_dict['model']['snn']['time_steps'],
        spike_threshold=config_dict['model']['snn']['threshold'],
        spike_beta=config_dict['model']['snn']['beta'],
        snn_hidden_sizes=tuple(config_dict['model']['snn']['hidden_sizes']),
        snn_latent_dim=config_dict['model']['snn']['latent_dim'],
        
        # LHC-optimized training settings
        adaptive_grad_clip_threshold=config_dict['training']['adaptive_grad_clip_threshold'],
        global_grad_clip_norm=config_dict['training']['global_grad_clip_norm'],
        per_module_grad_clip=config_dict['training']['per_module_grad_clip'],
        
        # LHC-optimized CPC settings
        cpc_temperature=config_dict['training']['cpc_temperature'],
        cpc_aux_weight=config_dict['training']['cpc_aux_weight'],
        cpc_prediction_steps=config_dict['training']['cpc_prediction_steps'],
    )
    
    logger.info("✅ LHC-Optimized Training Config created:")
    logger.info(f"   Time steps: {training_config.spike_time_steps}")
    logger.info(f"   Threshold: {training_config.spike_threshold}")
    logger.info(f"   Architecture: {training_config.snn_hidden_sizes}")
    logger.info(f"   Learning rate: {training_config.learning_rate}")
    logger.info(f"   Batch size: {training_config.batch_size}")
    
    return LHCOptimizedTrainer(training_config)


def run_lhc_optimized_training(
    config_path: str = None,
    output_dir: str = "outputs/lhc_optimized_training",
    max_epochs: int = 50,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete LHC-optimized training session.
    
    Args:
        config_path: Path to LHC configuration
        output_dir: Output directory for results
        max_epochs: Maximum training epochs
        verbose: Enable verbose logging
        
    Returns:
        Training results and metrics
    """
    logger.info("🚀 Starting LHC-Optimized Training Session")
    logger.info("=" * 80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = create_lhc_optimized_trainer(config_path)
    
    # TODO: Load data (integrate with existing data loaders)
    # For now, create synthetic data for testing
    logger.info("📊 Creating synthetic test data...")
    key = jax.random.PRNGKey(42)
    
    # Synthetic GW-like data
    batch_size = trainer.lhc_config.batch_size
    input_features = 1024
    
    train_signals = jax.random.normal(key, (batch_size * 10, input_features))
    train_labels = jax.random.randint(jax.random.split(key)[0], (batch_size * 10,), 0, 2)
    
    test_signals = jax.random.normal(jax.random.split(key)[1], (batch_size * 2, input_features))
    test_labels = jax.random.randint(jax.random.split(key)[2], (batch_size * 2,), 0, 2)
    
    # Initialize model and optimizer
    logger.info("🏗️ Initializing model and optimizer...")
    model = trainer.create_model()
    optimizer = trainer.create_optimizer()
    
    # Initialize training state
    params = model.init(key, train_signals[:batch_size])
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    # Training loop
    logger.info("🎯 Starting training loop...")
    training_metrics = []
    
    num_train_batches = len(train_signals) // batch_size
    num_test_batches = len(test_signals) // batch_size
    
    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        epoch_metrics = []
        
        # Training batches
        for batch_idx in range(num_train_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch = (
                train_signals[start_idx:end_idx],
                train_labels[start_idx:end_idx]
            )
            
            rng_key = jax.random.fold_in(key, epoch * num_train_batches + batch_idx)
            state, step_metrics = trainer.train_step(state, batch, rng_key)
            epoch_metrics.append(step_metrics)
        
        # Average epoch metrics
        avg_metrics = {}
        for metric_key in epoch_metrics[0].keys():
            avg_metrics[metric_key] = jnp.mean(jnp.array([m[metric_key] for m in epoch_metrics]))
        
        # Evaluation
        eval_metrics = []
        for batch_idx in range(num_test_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch = (
                test_signals[start_idx:end_idx],
                test_labels[start_idx:end_idx]
            )
            
            rng_key = jax.random.fold_in(key, epoch + 1000)
            eval_step_metrics = trainer.eval_step(state, batch, rng_key)
            eval_metrics.append(eval_step_metrics)
        
        # Average evaluation metrics
        avg_eval_metrics = {}
        for metric_key in eval_metrics[0].keys():
            avg_eval_metrics[f"eval_{metric_key}"] = jnp.mean(jnp.array([m[metric_key] for m in eval_metrics]))
        
        # Combine metrics
        combined_metrics = {**avg_metrics, **avg_eval_metrics}
        combined_metrics['epoch'] = epoch
        combined_metrics['epoch_time'] = time.time() - epoch_start_time
        
        training_metrics.append(combined_metrics)
        
        if verbose and epoch % 5 == 0:
            logger.info(f"Epoch {epoch:3d}: "
                       f"loss={combined_metrics['total_loss']:.4f}, "
                       f"acc={combined_metrics.get('eval_accuracy', 0.0):.4f}, "
                       f"spike_rate={combined_metrics['spike_rate_mean']:.4f}, "
                       f"time={combined_metrics['epoch_time']:.2f}s")
    
    # Final results
    final_metrics = training_metrics[-1]
    
    logger.info("🎊 LHC-Optimized Training Complete!")
    logger.info(f"   Final loss: {final_metrics['total_loss']:.4f}")
    logger.info(f"   Final accuracy: {final_metrics.get('eval_accuracy', 0.0):.4f}")
    logger.info(f"   Final spike rate: {final_metrics['spike_rate_mean']:.4f}")
    logger.info(f"   Time steps used: {final_metrics['time_steps_used']}")
    logger.info(f"   Threshold used: {final_metrics['threshold_used']}")
    logger.info(f"   Binary latent configs: {final_metrics.get('binary_latent_configs', 0):,}")
    
    # Save results
    results = {
        'training_metrics': training_metrics,
        'final_metrics': final_metrics,
        'config': {
            'time_steps': trainer.lhc_config.spike_time_steps,
            'threshold': trainer.lhc_config.spike_threshold,
            'beta': trainer.lhc_config.spike_beta,
            'architecture': trainer.lhc_config.snn_hidden_sizes,
            'latent_dim': trainer.lhc_config.snn_latent_dim,
        },
        'lhc_optimizations': {
            'time_step_reduction': f"32 → {trainer.lhc_config.spike_time_steps} ({32/trainer.lhc_config.spike_time_steps:.1f}x efficiency)",
            'threshold_increase': f"0.55 → {trainer.lhc_config.spike_threshold} ({trainer.lhc_config.spike_threshold/0.55:.1f}x selectivity)",
            'binary_latent_space': f"2^({trainer.lhc_config.spike_time_steps}*{trainer.lhc_config.snn_latent_dim}) = {2**(trainer.lhc_config.spike_time_steps * trainer.lhc_config.snn_latent_dim):,} configurations"
        }
    }
    
    # Save to file
    import json
    results_file = output_path / "lhc_training_results.json"
    with open(results_file, 'w') as f:
        # Convert jax arrays to regular python types for JSON serialization
        json_safe_results = {}
        for key, value in results.items():
            if key == 'training_metrics':
                json_safe_results[key] = [
                    {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
                    for metrics in value
                ]
            else:
                json_safe_results[key] = value
        
        json.dump(json_safe_results, f, indent=2)
    
    logger.info(f"📁 Results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    # Test LHC-optimized training
    results = run_lhc_optimized_training(
        config_path=None,  # Use default path
        output_dir="outputs/lhc_optimized_training",
        max_epochs=20,
        verbose=True
    )
    
    logger.info("✅ LHC-Optimized Training Test Complete!")
