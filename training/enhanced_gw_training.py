#!/usr/bin/env python3

"""
Enhanced Gravitational Wave Training Pipeline

Combines continuous GW signals (PyFstat) + binary GW signals (GWOSC)
for comprehensive neuromorphic CPC+SNN detector training.

Represents next-generation multi-signal type GW detection.
"""

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import logging
import time
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from dataclasses import dataclass

from ..models.cpc_encoder import create_enhanced_cpc_encoder, enhanced_info_nce_loss
from ..models.snn_classifier import create_snn_classifier, SNNTrainer
from ..models.spike_bridge import SpikeBridge, SpikeEncodingStrategy
from ..data.continuous_gw_generator import ContinuousGWGenerator, create_mixed_gw_dataset
from ..data.gw_download import ProductionGWOSCDownloader

logger = logging.getLogger(__name__)


@dataclass
class EnhancedTrainingConfig:
    """Configuration for enhanced GW training."""
    # Data configuration
    num_continuous_signals: int = 200
    num_binary_signals: int = 200
    signal_duration: float = 4.0  # seconds
    mix_ratio: float = 0.5  # Ratio continuous:binary
    
    # Training configuration
    batch_size: int = 16
    learning_rate: float = 1e-3
    num_epochs: int = 50
    
    # Model configuration
    cpc_latent_dim: int = 128
    snn_hidden_size: int = 64
    spike_time_steps: int = 50
    
    # Spike encoding
    spike_encoding: SpikeEncodingStrategy = SpikeEncodingStrategy.POISSON_RATE
    spike_rate: float = 100.0  # Hz
    
    # Output
    output_dir: str = "enhanced_gw_training_outputs"
    save_models: bool = True
    

class EnhancedGWTrainer:
    """
    Enhanced Gravitational Wave Trainer.
    
    Trains CPC+SNN neuromorphic detector on combined
    continuous + binary gravitational wave signals.
    """
    
    def __init__(self, config: EnhancedTrainingConfig):
        """
        Initialize enhanced GW trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data generators
        self.continuous_generator = ContinuousGWGenerator(
            base_frequency=50.0,
            freq_range=(20.0, 200.0),
            duration=config.signal_duration
        )
        
        self.binary_downloader = ProductionGWOSCDownloader()
        
        # Initialize models
        self.cpc_model = create_enhanced_cpc_encoder(
            latent_dim=config.cpc_latent_dim,
            use_batch_norm=False,  # Simplified for now
            dropout_rate=0.0
        )
        
        self.snn_model = create_snn_classifier(
            hidden_size=config.snn_hidden_size,
            num_classes=3  # 3 classes: noise, continuous, binary
        )
        
        self.spike_bridge = SpikeBridge(
            encoding_strategy=config.spike_encoding,
            spike_time_steps=config.spike_time_steps,
            max_spike_rate=config.spike_rate
        )
        
        # Optimizers
        self.cpc_optimizer = optax.adam(config.learning_rate)
        self.snn_trainer = SNNTrainer(
            snn_model=self.snn_model,
            learning_rate=config.learning_rate
        )
        
        logger.info("Initialized Enhanced GW Trainer")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Signal types: continuous + binary")
        logger.info(f"  Mix ratio: {config.mix_ratio}")
        
    def generate_enhanced_dataset(self) -> Dict:
        """
        Generate enhanced dataset with continuous + binary + noise signals.
        
        Returns:
            Enhanced dataset with multiple signal types
        """
        logger.info("Generating enhanced multi-signal dataset...")
        
        # Generate continuous signals
        continuous_data = self.continuous_generator.generate_training_dataset(
            num_signals=self.config.num_continuous_signals,
            signal_duration=self.config.signal_duration,
            include_noise_only=False  # Handle noise separately
        )
        
        # Generate synthetic binary signals (simplified)
        binary_data = self._generate_synthetic_binary_signals()
        
        # Generate pure noise samples
        noise_data = self._generate_noise_samples()
        
        # Combine all data
        enhanced_dataset = self._combine_datasets(continuous_data, binary_data, noise_data)
        
        logger.info(f"Enhanced dataset generated:")
        logger.info(f"  Total samples: {enhanced_dataset['data'].shape[0]}")
        logger.info(f"  Signal types: {jnp.unique(enhanced_dataset['labels'])}")
        logger.info(f"  Data shape: {enhanced_dataset['data'].shape}")
        
        return enhanced_dataset
    
    def _generate_synthetic_binary_signals(self) -> Dict:
        """Generate synthetic binary GW signals (simplified chirp model)."""
        logger.info(f"Generating {self.config.num_binary_signals} synthetic binary signals...")
        
        all_data = []
        all_metadata = []
        
        # ✅ Stwórz jeden główny klucz
        key = jax.random.PRNGKey(42)
        
        for i in range(self.config.num_binary_signals):
            # ✅ Dziel klucz w każdej iteracji
            key, f_key, noise_key = jax.random.split(key, 3)
            
            # Simple chirp parameters
            f_start = jnp.array(35.0 + 10 * jax.random.uniform(f_key))
            f_end = jnp.array(250.0)
            duration = self.config.signal_duration
            
            # Generate chirp signal
            t = jnp.linspace(0, duration, int(duration * 4096))
            
            # Frequency evolution (simplified)
            alpha = (f_end - f_start) / duration
            freq_t = f_start + alpha * t
            
            # Chirp signal
            signal = 1e-21 * jnp.sin(2 * jnp.pi * jnp.cumsum(freq_t) / 4096)
            
            # Add noise
            noise = jax.random.normal(noise_key, signal.shape) * 1e-23
            binary_signal = signal + noise
            
            all_data.append(binary_signal)
            all_metadata.append({
                'signal_type': 'binary_merger',
                'f_start': float(f_start),
                'f_end': float(f_end),
                'detector': 'H1'
            })
        
        return {
            'data': jnp.stack(all_data),
            'metadata': all_metadata
        }
    
    def _generate_noise_samples(self) -> Dict:
        """Generate pure noise samples."""
        num_noise = self.config.num_continuous_signals + self.config.num_binary_signals
        logger.info(f"Generating {num_noise} pure noise samples...")
        
        noise_length = int(self.config.signal_duration * 4096)
        
        all_noise = []
        all_metadata = []
        
        # ✅ Stwórz jeden główny klucz
        key = jax.random.PRNGKey(42)
        
        for i in range(num_noise):
            # ✅ Dziel klucz w każdej iteracji
            key, noise_key = jax.random.split(key, 2)
            
            noise = jax.random.normal(noise_key, (noise_length,)) * 1e-23
            
            all_noise.append(noise)
            all_metadata.append({
                'signal_type': 'noise_only',
                'detector': 'H1'
            })
        
        return {
            'data': jnp.stack(all_noise),
            'metadata': all_metadata
        }
    
    def _combine_datasets(self, continuous_data: Dict, binary_data: Dict, noise_data: Dict) -> Dict:
        """Combine continuous, binary, and noise datasets."""
        # Extract data arrays
        cont_data = continuous_data['data'][continuous_data['labels'] == 1]  # Only signals
        bin_data = binary_data['data']
        noise_data_array = noise_data['data']
        
        # Combine data
        all_data = jnp.concatenate([cont_data, bin_data, noise_data_array], axis=0)
        
        # Create labels: 0=noise, 1=continuous, 2=binary
        cont_labels = jnp.ones(cont_data.shape[0], dtype=jnp.int32)  # Continuous
        bin_labels = jnp.ones(bin_data.shape[0], dtype=jnp.int32) * 2  # Binary
        noise_labels = jnp.zeros(noise_data_array.shape[0], dtype=jnp.int32)  # Noise
        
        all_labels = jnp.concatenate([cont_labels, bin_labels, noise_labels])
        
        # Combine metadata
        all_metadata = (
            [m for m, l in zip(continuous_data['metadata'], continuous_data['labels']) if l == 1] +
            binary_data['metadata'] + 
            noise_data['metadata']
        )
        
        # Shuffle data
        key = jax.random.PRNGKey(42)
        indices = jax.random.permutation(key, len(all_data))
        
        return {
            'data': all_data[indices],
            'labels': all_labels[indices],
            'metadata': [all_metadata[i] for i in indices],
            'signal_types': ['noise', 'continuous_gw', 'binary_merger'],
            'num_classes': 3
        }
    
    def train_cpc_encoder(self, dataset: Dict, num_epochs: int = 20) -> Dict:
        """
        Train CPC encoder on enhanced dataset.
        
        Args:
            dataset: Enhanced multi-signal dataset
            num_epochs: Number of training epochs
            
        Returns:
            Training results and final parameters
        """
        logger.info("Training CPC encoder on enhanced dataset...")
        
        # Initialize parameters
        key = jax.random.PRNGKey(42)
        dummy_input = jnp.ones((1, int(self.config.signal_duration * 4096)))
        cpc_params = self.cpc_model.init(key, dummy_input)
        opt_state = self.cpc_optimizer.init(cpc_params)
        
        # Training loop
        losses = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Shuffle data for each epoch
            key, subkey = jax.random.split(key)
            indices = jax.random.permutation(subkey, len(dataset['data']))
            shuffled_data = dataset['data'][indices]
            
            # Batch training
            num_batches = len(shuffled_data) // self.config.batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.config.batch_size
                end_idx = start_idx + self.config.batch_size
                batch_data = shuffled_data[start_idx:end_idx]
                
                # ✅ Proper key splitting for each batch
                key, batch_key = jax.random.split(key)
                
                # Training step
                cpc_params, opt_state, loss = self._cpc_training_step(
                    cpc_params, opt_state, batch_data, batch_key
                )
                
                epoch_losses.append(loss)
            
            avg_loss = jnp.mean(jnp.array(epoch_losses))
            losses.append(avg_loss)
            
            if epoch % 5 == 0:
                logger.info(f"CPC Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        logger.info(f"CPC training completed. Final loss: {losses[-1]:.4f}")
        
        return {
            'cpc_params': cpc_params,
            'losses': losses,
            'epochs': num_epochs
        }
    
    @jax.jit
    def _cpc_training_step(self, params, opt_state, batch, key):
        """Single CPC training step."""
        def loss_fn(params):
            # Encode batch
            encoded = self.cpc_model.apply(params, batch)
            
            # Create context-target pairs for contrastive learning
            context_len = encoded.shape[1] // 2
            context = encoded[:, :context_len]
            targets = encoded[:, context_len:]
            
            # Enhanced InfoNCE loss with proper temperature parameter
            loss = enhanced_info_nce_loss(context, targets, temperature=0.1)
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = self.cpc_optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss
    
    def train_snn_classifier(self, dataset: Dict, cpc_params: Dict, num_epochs: int = 30) -> Dict:
        """
        Train SNN classifier on CPC-encoded spike trains.
        
        Args:
            dataset: Enhanced dataset
            cpc_params: Trained CPC parameters
            num_epochs: Number of training epochs
            
        Returns:
            Training results and SNN parameters
        """
        logger.info("Training SNN classifier on spike-encoded features...")
        
        # Initialize SNN parameters
        key = jax.random.PRNGKey(123)
        dummy_spikes = jnp.ones((1, self.config.spike_time_steps, self.config.cpc_latent_dim))
        snn_params = self.snn_model.init(key, dummy_spikes)
        opt_state = self.snn_trainer.optimizer.init(snn_params)
        
        # Encode all data through CPC
        logger.info("Encoding data through trained CPC...")
        cpc_features = self.cpc_model.apply(cpc_params, dataset['data'])
        
        # Convert to spike trains using vmap
        logger.info("Converting to spike trains...")
        
        # Create keys for each sample
        keys = jax.random.split(jax.random.PRNGKey(42), len(cpc_features))
        
        # Use vmap for efficient spike encoding
        spike_data = jax.vmap(
            lambda features, key: self.spike_bridge.encode(features[None], key)[0]
        )(cpc_features, keys)
        labels = dataset['labels']
        
        logger.info(f"Spike data shape: {spike_data.shape}")
        
        # Training loop
        accuracies = []
        losses = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_accs = []
            
            # Batch training
            num_batches = len(spike_data) // self.config.batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.config.batch_size
                end_idx = start_idx + self.config.batch_size
                
                batch_spikes = spike_data[start_idx:end_idx]
                batch_labels = labels[start_idx:end_idx]
                
                # Training step - train_step returns accuracy as well
                snn_params, opt_state, loss, acc = self.snn_trainer.train_step(
                    snn_params, opt_state, batch_spikes, batch_labels
                )
                
                # No need for separate accuracy call as it's returned from train_step
                
                epoch_losses.append(loss)
                epoch_accs.append(acc)
            
            avg_loss = jnp.mean(jnp.array(epoch_losses))
            avg_acc = jnp.mean(jnp.array(epoch_accs))
            
            losses.append(avg_loss)
            accuracies.append(avg_acc)
            
            if epoch % 5 == 0:
                logger.info(f"SNN Epoch {epoch}: Loss = {avg_loss:.4f}, Acc = {avg_acc:.3f}")
        
        logger.info(f"SNN training completed. Final accuracy: {accuracies[-1]:.3f}")
        
        return {
            'snn_params': snn_params,
            'losses': losses,
            'accuracies': accuracies,
            'epochs': num_epochs,
            'final_accuracy': float(accuracies[-1])
        }
    
    def evaluate_multi_signal_performance(self, dataset: Dict, cpc_params: Dict, snn_params: Dict) -> Dict:
        """
        Evaluate performance on different signal types.
        
        Args:
            dataset: Test dataset
            cpc_params: Trained CPC parameters  
            snn_params: Trained SNN parameters
            
        Returns:
            Performance metrics by signal type
        """
        logger.info("Evaluating multi-signal performance...")
        
        # Encode through full pipeline
        cpc_features = self.cpc_model.apply(cpc_params, dataset['data'])
        
        # Convert to spike trains using vmap
        keys = jax.random.split(jax.random.PRNGKey(5000), len(cpc_features))
        spike_data = jax.vmap(
            lambda features, key: self.spike_bridge.encode(features[None], key)[0]
        )(cpc_features, keys)
        
        # Get predictions
        logits = self.snn_model.apply(snn_params, spike_data)
        predictions = jnp.argmax(logits, axis=1)
        true_labels = dataset['labels']
        
        # Compute metrics by signal type
        results = {}
        for signal_type_idx, signal_type in enumerate(dataset['signal_types']):
            mask = true_labels == signal_type_idx
            if jnp.sum(mask) > 0:
                type_predictions = predictions[mask]
                type_labels = true_labels[mask]
                accuracy = jnp.mean(type_predictions == type_labels)
                
                results[signal_type] = {
                    'accuracy': float(accuracy),
                    'num_samples': int(jnp.sum(mask)),
                    'true_positives': int(jnp.sum((type_predictions == signal_type_idx) & (type_labels == signal_type_idx)))
                }
        
        # Overall accuracy
        overall_accuracy = jnp.mean(predictions == true_labels)
        results['overall'] = {
            'accuracy': float(overall_accuracy),
            'num_samples': len(true_labels)
        }
        
        logger.info("Multi-signal performance evaluation:")
        for signal_type, metrics in results.items():
            logger.info(f"  {signal_type}: {metrics['accuracy']:.3f} ({metrics['num_samples']} samples)")
        
        return results
    
    def run_enhanced_training(self) -> Dict:
        """
        Run complete enhanced training pipeline.
        
        Returns:
            Complete training results
        """
        logger.info("🚀 Starting Enhanced GW Training Pipeline...")
        start_time = time.time()
        
        # Generate enhanced dataset
        dataset = self.generate_enhanced_dataset()
        
        # Split into train/test (80/20)
        split_idx = int(0.8 * len(dataset['data']))
        train_data = {
            'data': dataset['data'][:split_idx],
            'labels': dataset['labels'][:split_idx],
            'metadata': dataset['metadata'][:split_idx],
            'signal_types': dataset['signal_types'],
            'num_classes': dataset['num_classes']
        }
        
        test_data = {
            'data': dataset['data'][split_idx:],
            'labels': dataset['labels'][split_idx:],
            'metadata': dataset['metadata'][split_idx:],
            'signal_types': dataset['signal_types'],
            'num_classes': dataset['num_classes']
        }
        
        # Train CPC encoder
        cpc_results = self.train_cpc_encoder(train_data, num_epochs=20)
        
        # Train SNN classifier
        snn_results = self.train_snn_classifier(train_data, cpc_results['cpc_params'], num_epochs=30)
        
        # Evaluate performance
        performance = self.evaluate_multi_signal_performance(
            test_data, cpc_results['cpc_params'], snn_results['snn_params']
        )
        
        # Save models if requested
        if self.config.save_models:
            self._save_models(cpc_results['cpc_params'], snn_results['snn_params'])
        
        total_time = time.time() - start_time
        
        final_results = {
            'cpc_training': cpc_results,
            'snn_training': snn_results,
            'performance': performance,
            'training_time': total_time,
            'config': self.config
        }
        
        logger.info(f"🎉 Enhanced Training Completed!")
        logger.info(f"  Total time: {total_time:.1f} seconds")
        logger.info(f"  Overall accuracy: {performance['overall']['accuracy']:.3f}")
        logger.info(f"  CPC final loss: {cpc_results['losses'][-1]:.4f}")
        logger.info(f"  SNN final accuracy: {snn_results['final_accuracy']:.3f}")
        
        return final_results
    
    def _save_checkpoint(self, cpc_params: Dict, snn_params: Dict, step: int):
        """Save checkpoint using Orbax."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpointer = ocp.StandardCheckpointer()
        checkpoint_path = checkpoint_dir / f"step_{step}"
        
        checkpointer.save(
            checkpoint_path,
            {
                'cpc_params': cpc_params,
                'snn_params': snn_params,
                'step': step,
                'config': self.config
            }
        )
        
        logger.info(f"Saved checkpoint at step {step}")
    
    def _save_models(self, cpc_params: Dict, snn_params: Dict):
        """Save trained model parameters."""
        import pickle
        
        models_dir = self.output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Save CPC parameters
        with open(models_dir / "cpc_params.pkl", "wb") as f:
            pickle.dump(cpc_params, f)
        
        # Save SNN parameters  
        with open(models_dir / "snn_params.pkl", "wb") as f:
            pickle.dump(snn_params, f)
        
        logger.info(f"Models saved to {models_dir}")


def run_enhanced_gw_training_experiment():
    """Quick test of enhanced training."""
    print("🌟 Enhanced GW Training Test")
    
    config = EnhancedTrainingConfig(
        num_continuous_signals=5,  # Small test
        num_binary_signals=5,
        signal_duration=2.0,  # Shorter for testing
        batch_size=4,
        num_epochs=2
    )
    
    trainer = EnhancedGWTrainer(config)
    dataset = trainer.generate_enhanced_dataset()
    
    print(f"✅ Dataset generated: {dataset['data'].shape}")
    print(f"✅ Signal types: {dataset['signal_types']}")
    return True


if __name__ == "__main__":
    import os
    os.environ['JAX_PLATFORMS'] = 'cpu'
    success = run_enhanced_gw_training_experiment()
    print("✅ Test completed!" if success else "❌ Test failed!") 