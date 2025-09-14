"""
Advanced GW trainer implementation.

This module contains the RealAdvancedGWTrainer class extracted from
advanced_training.py for better modularity.

Split from advanced_training.py for better maintainability.
"""

import logging
import time
from typing import Dict, Any, Optional
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from .attention import AttentionCPCEncoder
from .snn_deep import DeepSNN

logger = logging.getLogger(__name__)


class RealAdvancedGWTrainer:
    """
    Real advanced trainer with actual gradient updates.
    Replaces mock training from Executive Summary analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize real advanced trainer."""
        self.config = config
        self.setup_models()
        self.setup_optimizers()
        self.setup_loss_functions()
        
        # Training state
        self.train_state = None
        self.current_epoch = 0
        self.best_accuracy = 0.0
        
        # Metrics tracking
        self.training_metrics = {
            'cpc_losses': [],
            'classification_losses': [],
            'total_losses': [],
            'accuracies': [],
            'spike_rates': []
        }
        
        logger.info("Real Advanced GW Trainer initialized")
        
    def setup_models(self):
        """Setup model architectures."""
        # CPC encoder with attention
        self.cpc_encoder = AttentionCPCEncoder(
            latent_dim=self.config.get('latent_dim', 256),
            num_heads=self.config.get('num_attention_heads', 8),
            dropout_rate=self.config.get('dropout_rate', 0.1)
        )
        
        # Spike bridge
        from models.bridge.core import ValidatedSpikeBridge
        self.spike_bridge = ValidatedSpikeBridge(
            spike_encoding="temporal_contrast",  # Fixed from Executive Summary
            time_steps=self.config.get('snn_time_steps', 16),
            threshold=self.config.get('spike_threshold', 0.1)
        )
        
        # Deep SNN classifier
        self.snn_classifier = DeepSNN(
            hidden_dims=self.config.get('snn_hidden_dims', (256, 128, 64)),
            num_classes=self.config.get('num_classes', 2),
            time_steps=self.config.get('snn_time_steps', 16),
            tau_mem=self.config.get('tau_mem', 20.0),
            tau_syn=self.config.get('tau_syn', 5.0)
        )
        
    def setup_optimizers(self):
        """Setup optimizers with proper scheduling."""
        # Cosine annealing with warmup (Executive Summary fix)
        base_lr = self.config.get('learning_rate', 3e-4)
        warmup_epochs = self.config.get('warmup_epochs', 10)
        max_epochs = self.config.get('max_epochs', 100)
        
        # Warmup + cosine annealing schedule
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=base_lr,
            warmup_steps=warmup_epochs,
            decay_steps=max_epochs,
            end_value=base_lr * 0.01
        )
        
        # Optimizer with L2 regularization (Executive Summary fix)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Gradient clipping
            optax.adamw(
                learning_rate=schedule,
                weight_decay=self.config.get('weight_decay', 1e-5)  # L2 regularization
            )
        )
        
    def setup_loss_functions(self):
        """Setup loss functions."""
        # Focal loss for class imbalance (Executive Summary fix)
        self.focal_loss_alpha = self.config.get('focal_loss_alpha', 0.25)
        self.focal_loss_gamma = self.config.get('focal_loss_gamma', 2.0)
        
        # CPC loss temperature
        self.cpc_temperature = self.config.get('cpc_temperature', 0.1)
    
    def focal_loss(self, logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        """Compute focal loss for class imbalance."""
        # Convert to probabilities
        probs = jax.nn.softmax(logits, axis=-1)
        
        # Get probability of true class
        true_class_probs = probs[jnp.arange(len(labels)), labels]
        
        # Focal loss formula: -α(1-p)^γ log(p)
        focal_weight = self.focal_loss_alpha * jnp.power(1 - true_class_probs, self.focal_loss_gamma)
        cross_entropy = -jnp.log(true_class_probs + 1e-8)
        
        return jnp.mean(focal_weight * cross_entropy)
    
    def enhanced_loss_fn(self, params: Dict, batch: Dict, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Dict]:
        """Advanced loss function with CPC + classification + regularization."""
        strain_data = batch['strain']
        labels = batch['labels']
        
        # ✅ FORWARD PASS: Complete pipeline
        # CPC encoding with attention
        cpc_features = self.cpc_encoder.apply(params['cpc'], strain_data, training=True)
        
        # Spike encoding
        spike_trains = self.spike_bridge.apply(
            params['spike_bridge'], 
            cpc_features, 
            training=True
        )
        
        # SNN classification
        snn_output = self.snn_classifier.apply(
            params['snn'], 
            spike_trains, 
            training=True
        )
        
        logits = snn_output['logits']
        
        # ✅ LOSS COMPONENTS
        
        # 1. Classification loss with focal loss for imbalance
        classification_loss = self.focal_loss(logits, labels)
        
        # 2. CPC contrastive loss (InfoNCE)
        if cpc_features.shape[1] > 1:  # Need temporal dimension
            # Simple InfoNCE loss
            context = cpc_features[:, :-1, :]  # All but last
            targets = cpc_features[:, 1:, :]   # All but first
            
            # Flatten temporal dimension
            context_flat = context.reshape(-1, context.shape[-1])
            targets_flat = targets.reshape(-1, targets.shape[-1])
            
            if context_flat.shape[0] > 1:  # Need multiple samples
                # L2 normalize
                context_norm = context_flat / (jnp.linalg.norm(context_flat, axis=-1, keepdims=True) + 1e-8)
                targets_norm = targets_flat / (jnp.linalg.norm(targets_flat, axis=-1, keepdims=True) + 1e-8)
                
                # Similarity matrix
                sim_matrix = jnp.dot(context_norm, targets_norm.T) / self.cpc_temperature
                
                # InfoNCE loss
                num_samples = sim_matrix.shape[0]
                labels_cpc = jnp.arange(num_samples)
                log_softmax = jax.nn.log_softmax(sim_matrix, axis=1)
                cpc_loss = -jnp.mean(log_softmax[labels_cpc, labels_cpc])
            else:
                cpc_loss = jnp.array(0.0)
        else:
            cpc_loss = jnp.array(0.0)
        
        # 3. Spike rate regularization
        spike_rate = snn_output['final_spike_rate']
        target_spike_rate = self.config.get('target_spike_rate', 0.1)
        spike_reg_loss = jnp.square(spike_rate - target_spike_rate)
        
        # ✅ TOTAL LOSS: Weighted combination
        cpc_weight = self.config.get('cpc_loss_weight', 0.3)
        spike_reg_weight = self.config.get('spike_reg_weight', 0.1)
        
        total_loss = (classification_loss + 
                     cpc_weight * cpc_loss + 
                     spike_reg_weight * spike_reg_loss)
        
        # ✅ METRICS: Compute accuracy
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
        
        aux_data = {
            'classification_loss': classification_loss,
            'cpc_loss': cpc_loss,
            'spike_reg_loss': spike_reg_loss,
            'total_loss': total_loss,
            'accuracy': accuracy,
            'spike_rate': spike_rate
        }
        
        return total_loss, aux_data
    
    def train_step(self, train_state: train_state.TrainState, batch: Dict) -> Tuple[train_state.TrainState, Dict]:
        """Execute single training step with real gradients."""
        
        def loss_fn(params):
            return self.enhanced_loss_fn(params, batch, jax.random.PRNGKey(train_state.step))
        
        # Compute loss and gradients
        (total_loss, aux_data), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        
        # Apply gradients
        new_train_state = train_state.apply_gradients(grads=grads)
        
        # Update metrics
        metrics = {
            'step': int(new_train_state.step),
            'epoch': self.current_epoch,
            'total_loss': float(total_loss),
            'classification_loss': float(aux_data['classification_loss']),
            'cpc_loss': float(aux_data['cpc_loss']),
            'accuracy': float(aux_data['accuracy']),
            'spike_rate': float(aux_data['spike_rate'])
        }
        
        # Store metrics
        self.training_metrics['total_losses'].append(metrics['total_loss'])
        self.training_metrics['classification_losses'].append(metrics['classification_loss'])
        self.training_metrics['cpc_losses'].append(metrics['cpc_loss'])
        self.training_metrics['accuracies'].append(metrics['accuracy'])
        self.training_metrics['spike_rates'].append(metrics['spike_rate'])
        
        return new_train_state, metrics
    
    def eval_step(self, train_state: train_state.TrainState, batch: Dict) -> Dict:
        """Execute evaluation step."""
        strain_data = batch['strain']
        labels = batch['labels']
        
        # Forward pass without gradients
        cpc_features = self.cpc_encoder.apply(train_state.params['cpc'], strain_data, training=False)
        spike_trains = self.spike_bridge.apply(train_state.params['spike_bridge'], cpc_features, training=False)
        snn_output = self.snn_classifier.apply(train_state.params['snn'], spike_trains, training=False)
        
        logits = snn_output['logits']
        
        # Compute metrics
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        
        return {
            'eval_loss': float(loss),
            'eval_accuracy': float(accuracy),
            'eval_spike_rate': float(snn_output['final_spike_rate'])
        }
    
    def initialize_training_state(self, 
                                sample_batch: Dict[str, jnp.ndarray],
                                key: jax.random.PRNGKey) -> train_state.TrainState:
        """Initialize training state with real parameter initialization."""
        # Split keys for different components
        key_cpc, key_bridge, key_snn = jax.random.split(key, 3)
        
        # Sample input for initialization
        sample_strain = sample_batch['strain']  # [batch_size, seq_len]
        
        # Initialize CPC encoder
        cpc_params = self.cpc_encoder.init(key_cpc, sample_strain, training=True)
        
        # Initialize spike bridge (need CPC output shape)
        sample_cpc_features = self.cpc_encoder.apply(cpc_params, sample_strain, training=False)
        spike_bridge_params = self.spike_bridge.init(key_bridge, sample_cpc_features, training=True)
        
        # Initialize SNN classifier (need spike output shape)
        sample_spikes = self.spike_bridge.apply(spike_bridge_params, sample_cpc_features, training=False)
        snn_params = self.snn_classifier.init(key_snn, sample_spikes, training=True)
        
        # Combine all parameters
        all_params = {
            'cpc': cpc_params,
            'spike_bridge': spike_bridge_params,
            'snn': snn_params
        }
        
        # Create training state
        self.train_state = train_state.TrainState.create(
            apply_fn=None,  # Will be set per component
            params=all_params,
            tx=self.optimizer
        )
        
        logger.info("✅ Training state initialized with real parameters")
        return self.train_state
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.training_metrics['total_losses']:
            return {'no_training_data': True}
        
        # Calculate statistics
        metrics = self.training_metrics
        
        summary = {
            'total_steps': len(metrics['total_losses']),
            'current_epoch': self.current_epoch,
            'best_accuracy': self.best_accuracy,
            
            # Loss statistics
            'final_total_loss': metrics['total_losses'][-1] if metrics['total_losses'] else 0.0,
            'final_classification_loss': metrics['classification_losses'][-1] if metrics['classification_losses'] else 0.0,
            'final_cpc_loss': metrics['cpc_losses'][-1] if metrics['cpc_losses'] else 0.0,
            
            # Accuracy statistics
            'final_accuracy': metrics['accuracies'][-1] if metrics['accuracies'] else 0.0,
            'mean_accuracy': float(jnp.mean(jnp.array(metrics['accuracies']))) if metrics['accuracies'] else 0.0,
            'max_accuracy': float(jnp.max(jnp.array(metrics['accuracies']))) if metrics['accuracies'] else 0.0,
            
            # Spike rate statistics
            'final_spike_rate': metrics['spike_rates'][-1] if metrics['spike_rates'] else 0.0,
            'mean_spike_rate': float(jnp.mean(jnp.array(metrics['spike_rates']))) if metrics['spike_rates'] else 0.0,
            
            # Training quality indicators
            'loss_decreased': (metrics['total_losses'][-1] < metrics['total_losses'][0]) if len(metrics['total_losses']) > 1 else False,
            'accuracy_improved': (metrics['accuracies'][-1] > metrics['accuracies'][0]) if len(metrics['accuracies']) > 1 else False,
            
            # Configuration
            'config': self.config
        }
        
        return summary


def create_real_advanced_trainer(config: Dict[str, Any]) -> RealAdvancedGWTrainer:
    """
    Create real advanced trainer with configuration.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Configured RealAdvancedGWTrainer
    """
    # Default configuration
    default_config = {
        'latent_dim': 256,
        'num_attention_heads': 8,
        'dropout_rate': 0.1,
        'snn_hidden_dims': (256, 128, 64),
        'num_classes': 2,
        'snn_time_steps': 16,
        'spike_threshold': 0.1,
        'tau_mem': 20.0,
        'tau_syn': 5.0,
        'learning_rate': 3e-4,
        'warmup_epochs': 10,
        'max_epochs': 100,
        'weight_decay': 1e-5,
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
        'cpc_temperature': 0.1,
        'target_spike_rate': 0.1,
        'cpc_loss_weight': 0.3,
        'spike_reg_weight': 0.1
    }
    
    # Merge with provided config
    merged_config = {**default_config, **config}
    
    logger.info(f"Creating RealAdvancedGWTrainer with config: {len(merged_config)} parameters")
    
    return RealAdvancedGWTrainer(merged_config)


def create_advanced_training_config(**kwargs) -> Dict[str, Any]:
    """
    Create advanced training configuration with overrides.
    
    Args:
        **kwargs: Configuration parameter overrides
        
    Returns:
        Advanced training configuration dictionary
    """
    # Base advanced configuration
    config = {
        # Model architecture
        'latent_dim': 256,
        'num_attention_heads': 8,
        'dropout_rate': 0.1,
        
        # SNN configuration
        'snn_hidden_dims': (256, 128, 64),
        'num_classes': 2,
        'snn_time_steps': 16,
        'tau_mem': 20.0,
        'tau_syn': 5.0,
        'spike_threshold': 0.1,
        'target_spike_rate': 0.1,
        
        # Training parameters
        'learning_rate': 3e-4,
        'warmup_epochs': 10,
        'max_epochs': 100,
        'weight_decay': 1e-5,
        
        # Loss parameters
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
        'cpc_temperature': 0.1,
        'cpc_loss_weight': 0.3,
        'spike_reg_weight': 0.1
    }
    
    # Apply overrides
    config.update(kwargs)
    
    return config


# Export trainer and factory
__all__ = [
    "RealAdvancedGWTrainer",
    "create_real_advanced_trainer",
    "create_advanced_training_config"
]
