#!/usr/bin/env python3

"""
Advanced Training with Real Gradient Updates
Addresses Executive Summary Priority 5: Replace Mock Training
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from typing import Dict, Any, Optional, Tuple, List
import logging
import time
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class AttentionCPCEncoder(nn.Module):
    """
    CPC encoder with multi-head self-attention for enhanced representation learning.
    Executive Summary implementation: attention-enhanced CPC.
    """
    
    latent_dim: int = 256
    num_heads: int = 8
    dropout_rate: float = 0.1
    
    def setup(self):
        """Initialize attention CPC encoder components."""
        # Convolutional feature extraction
        self.conv_stack = [
            nn.Conv(64, kernel_size=(7,), strides=(2,), padding='SAME'),
            nn.BatchNorm(),
            nn.Conv(128, kernel_size=(5,), strides=(2,), padding='SAME'), 
            nn.BatchNorm(),
            nn.Conv(self.latent_dim, kernel_size=(3,), strides=(1,), padding='SAME'),
            nn.BatchNorm()
        ]
        
        # Multi-head self-attention for temporal modeling
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=jnp.float32,
            dropout_rate=self.dropout_rate
        )
        
        # Position encoding
        self.pos_embedding = nn.Embed(2048, self.latent_dim)  # Max sequence length 2048
        
        # Layer normalization and feedforward
        self.layer_norm1 = nn.LayerNorm()
        self.layer_norm2 = nn.LayerNorm()
        
        self.feedforward = [
            nn.Dense(self.latent_dim * 4),
            nn.gelu,
            nn.Dropout(self.dropout_rate),
            nn.Dense(self.latent_dim),
            nn.Dropout(self.dropout_rate)
        ]
        
        # Context prediction head
        self.context_predictor = nn.Dense(self.latent_dim)
        
    def __call__(self, x: jnp.ndarray, training: bool = True) -> Dict[str, jnp.ndarray]:
        """
        Forward pass with attention-enhanced feature extraction.
        
        Args:
            x: Input strain data [batch_size, seq_len]
            training: Training mode flag
            
        Returns:
            Dictionary with encoded features and context predictions
        """
        batch_size, seq_len = x.shape
        
        # Add feature dimension for convolution
        x = x[:, :, None]  # [batch_size, seq_len, 1]
        
        # Convolutional feature extraction
        for i, layer in enumerate(self.conv_stack):
            if isinstance(layer, nn.BatchNorm):
                x = layer(x, use_running_average=not training)
            elif isinstance(layer, (nn.Conv, nn.Dense)):
                x = layer(x)
            else:
                x = layer(x)  # Activation functions
        
        # x shape: [batch_size, reduced_seq_len, latent_dim]
        reduced_seq_len = x.shape[1]
        
        # Add positional encoding
        positions = jnp.arange(reduced_seq_len)[None, :]  # [1, reduced_seq_len]
        pos_embeddings = self.pos_embedding(positions)  # [1, reduced_seq_len, latent_dim]
        x = x + pos_embeddings
        
        # Self-attention layer
        x_norm1 = self.layer_norm1(x)
        attention_output = self.attention(
            x_norm1, x_norm1, x_norm1,
            deterministic=not training
        )
        x = x + attention_output  # Residual connection
        
        # Feedforward layer
        x_norm2 = self.layer_norm2(x)
        ff_output = x_norm2
        for layer in self.feedforward:
            if isinstance(layer, nn.Dropout):
                ff_output = layer(ff_output, deterministic=not training)
            else:
                ff_output = layer(ff_output)
        
        x = x + ff_output  # Residual connection
        
        # Context prediction for CPC loss
        context_predictions = self.context_predictor(x)
        
        return {
            'encoded_features': x,  # [batch_size, reduced_seq_len, latent_dim]
            'context_predictions': context_predictions,
            'sequence_length': reduced_seq_len
        }

class DeepSNN(nn.Module):
    """
    Deep 3-layer Spiking Neural Network for classification.
    Executive Summary implementation: deep SNN (256→128→64→classes).
    """
    
    hidden_dims: List[int] = (256, 128, 64)
    num_classes: int = 2
    time_steps: int = 16
    tau_mem: float = 20.0
    tau_syn: float = 5.0
    threshold: float = 1.0
    surrogate_beta: float = 4.0
    
    def setup(self):
        """Initialize deep SNN layers."""
        # LIF neuron layers
        self.lif_layers = []
        input_dim = None  # Will be inferred
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            self.lif_layers.append(
                LIFLayer(
                    hidden_dim=hidden_dim,
                    tau_mem=self.tau_mem,
                    tau_syn=self.tau_syn,
                    threshold=self.threshold,
                    surrogate_beta=self.surrogate_beta,
                    name=f'lif_layer_{i}'
                )
            )
        
        # Final classification layer
        self.classifier = nn.Dense(self.num_classes)
        
        # Layer normalization for stability
        self.layer_norms = [nn.LayerNorm() for _ in self.hidden_dims]
        
    def __call__(self, 
                 spike_trains: jnp.ndarray,
                 training: bool = True) -> Dict[str, jnp.ndarray]:
        """
        Process spike trains through deep SNN.
        
        Args:
            spike_trains: Input spikes [batch_size, time_steps, seq_len, feature_dim]
            training: Training mode flag
            
        Returns:
            Classification outputs and intermediate activations
        """
        batch_size, time_steps, seq_len, feature_dim = spike_trains.shape
        
        # Flatten spatial dimensions for processing
        # [batch_size, time_steps, seq_len * feature_dim]
        x = spike_trains.reshape(batch_size, time_steps, -1)
        
        # Process through LIF layers
        layer_outputs = []
        layer_states = []
        
        for i, (lif_layer, layer_norm) in enumerate(zip(self.lif_layers, self.layer_norms)):
            # Apply layer normalization to input
            if i == 0:
                x_norm = x  # Don't normalize first layer input (spikes)
            else:
                x_norm = layer_norm(x, use_running_average=not training)
            
            # Process through LIF layer
            x, states = lif_layer(x_norm, training=training)
            
            layer_outputs.append(x)
            layer_states.append(states)
        
        # Global average pooling over time and space
        # [batch_size, time_steps, final_hidden_dim] → [batch_size, final_hidden_dim]
        pooled_output = jnp.mean(x, axis=(1, 2))  # Average over time and spatial dims
        
        # Final classification
        logits = self.classifier(pooled_output)
        
        return {
            'logits': logits,  # [batch_size, num_classes]
            'pooled_features': pooled_output,
            'layer_outputs': layer_outputs,
            'layer_states': layer_states,
            'final_spike_rate': jnp.mean(x)
        }

class LIFLayer(nn.Module):
    """
    Leaky Integrate-and-Fire neuron layer with proper dynamics.
    """
    
    hidden_dim: int
    tau_mem: float = 20.0
    tau_syn: float = 5.0  
    threshold: float = 1.0
    surrogate_beta: float = 4.0
    
    def setup(self):
        """Initialize LIF layer parameters."""
        self.dense = nn.Dense(self.hidden_dim)
        
        # LIF parameters
        self.alpha = jnp.exp(-1.0 / self.tau_mem)  # Membrane decay
        self.beta = jnp.exp(-1.0 / self.tau_syn)   # Synaptic decay
        
    def __call__(self, 
                 x: jnp.ndarray,
                 training: bool = True) -> Tuple[jnp.ndarray, Dict]:
        """
        Process input through LIF dynamics.
        
        Args:
            x: Input [batch_size, time_steps, input_dim]
            training: Training mode
            
        Returns:
            Output spikes and internal states
        """
        batch_size, time_steps, input_dim = x.shape
        
        # Apply linear transformation
        x_transformed = self.dense(x)  # [batch_size, time_steps, hidden_dim]
        
        # Initialize states
        v_mem = jnp.zeros((batch_size, self.hidden_dim))  # Membrane potential
        i_syn = jnp.zeros((batch_size, self.hidden_dim))  # Synaptic current
        
        # Collect outputs
        spike_outputs = []
        membrane_history = []
        
        # Simulate LIF dynamics
        for t in range(time_steps):
            # Update synaptic current
            i_syn = self.beta * i_syn + x_transformed[:, t, :]
            
            # Update membrane potential
            v_mem = self.alpha * v_mem + i_syn
            
            # Generate spikes with surrogate gradient
            spikes = self._spike_function(v_mem - self.threshold)
            
            # Reset membrane potential where spikes occurred
            v_mem = v_mem * (1 - spikes)
            
            spike_outputs.append(spikes)
            membrane_history.append(v_mem)
        
        # Stack outputs: [batch_size, time_steps, hidden_dim]
        output_spikes = jnp.stack(spike_outputs, axis=1)
        
        states = {
            'final_v_mem': v_mem,
            'final_i_syn': i_syn,
            'membrane_history': jnp.stack(membrane_history, axis=1),
            'spike_rate': jnp.mean(output_spikes)
        }
        
        return output_spikes, states
        
    def _spike_function(self, x: jnp.ndarray) -> jnp.ndarray:
        """Spike function with surrogate gradient."""
        # Forward: Heaviside step
        spikes = (x >= 0).astype(jnp.float32)
        
        # Custom gradient using fast sigmoid surrogate
        @jax.custom_vjp
        def spike_with_grad(x):
            return spikes
        
        def spike_fwd(x):
            return spike_with_grad(x), x
            
        def spike_bwd(res, g):
            x = res
            # Fast sigmoid surrogate gradient
            surrogate_grad = self.surrogate_beta * jnp.exp(-self.surrogate_beta * jnp.abs(x)) / \
                           (2 * (1 + jnp.exp(-self.surrogate_beta * jnp.abs(x)))**2)
            return g * surrogate_grad,
        
        spike_with_grad.defvjp(spike_fwd, spike_bwd)
        return spike_with_grad(x)

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
        from models.spike_bridge import ValidatedSpikeBridge
        self.spike_bridge = ValidatedSpikeBridge(
            spike_encoding="temporal_contrast",  # Fixed from Executive Summary
            time_steps=self.config.get('snn_time_steps', 16),
            threshold=self.config.get('spike_threshold', 0.1)
        )
        
        # Deep SNN classifier
        self.snn_classifier = DeepSNN(
            hidden_dims=self.config.get('snn_hidden_dims', [256, 128, 64]),
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
        
    def initialize_training_state(self, 
                                sample_batch: Dict[str, jnp.ndarray],
                                key: jax.random.PRNGKey) -> train_state.TrainState:
        """Initialize training state with real parameter initialization."""
        # Split keys for different components
        key_cpc, key_bridge, key_snn = jax.random.split(key, 3)
        
        # Sample input for initialization
        sample_strain = sample_batch['strain']  # [batch_size, seq_len]
        batch_size = sample_strain.shape[0]
        
        # Initialize CPC encoder
        cpc_variables = self.cpc_encoder.init(key_cpc, sample_strain, training=True)
        
        # Get CPC output for bridge initialization
        cpc_output = self.cpc_encoder.apply(cpc_variables, sample_strain, training=True)
        
        # Initialize spike bridge
        bridge_variables = self.spike_bridge.init(
            key_bridge, cpc_output['encoded_features'], training=True
        )
        
        # Get spike output for SNN initialization
        spike_output = self.spike_bridge.apply(
            bridge_variables, cpc_output['encoded_features'], training=True
        )
        
        # Initialize SNN classifier
        snn_variables = self.snn_classifier.init(key_snn, spike_output, training=True)
        
        # Combine all parameters
        params = {
            'cpc_encoder': cpc_variables,
            'spike_bridge': bridge_variables,
            'snn_classifier': snn_variables
        }
        
        # Create training state
        self.train_state = train_state.TrainState.create(
            apply_fn=None,  # We'll use individual apply functions
            params=params,
            tx=self.optimizer
        )
        
        logger.info(f"Training state initialized with {self._count_parameters(params)} parameters")
        return self.train_state
        
    def _count_parameters(self, params: Dict) -> int:
        """Count total number of parameters."""
        def count_tree(tree):
            if isinstance(tree, dict):
                return sum(count_tree(v) for v in tree.values())
            elif hasattr(tree, 'size'):
                return tree.size
            else:
                return 0
        
        return count_tree(params)
    
    def focal_loss(self, logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        """
        Focal loss for addressing class imbalance.
        Executive Summary fix: focal loss implementation.
        """
        # Convert labels to one-hot if needed
        if len(labels.shape) == 1:
            labels_onehot = jax.nn.one_hot(labels, logits.shape[-1])
        else:
            labels_onehot = labels
            
        # Compute probabilities
        probs = jax.nn.softmax(logits, axis=-1)
        
        # Focal loss computation
        ce_loss = -jnp.sum(labels_onehot * jax.nn.log_softmax(logits, axis=-1), axis=-1)
        p_t = jnp.sum(labels_onehot * probs, axis=-1)
        
        focal_weight = self.focal_loss_alpha * jnp.power(1 - p_t, self.focal_loss_gamma)
        focal_loss = focal_weight * ce_loss
        
        return jnp.mean(focal_loss)
    
    def cpc_loss(self, 
                 encoded_features: jnp.ndarray,
                 context_predictions: jnp.ndarray) -> jnp.ndarray:
        """
        Contrastive Predictive Coding loss.
        """
        batch_size, seq_len, feature_dim = encoded_features.shape
        
        # Simple InfoNCE loss implementation
        # Predict next time step from current context
        if seq_len <= 1:
            return jnp.array(0.0)  # Skip if sequence too short
            
        context = context_predictions[:, :-1, :]  # [batch_size, seq_len-1, feature_dim]
        targets = encoded_features[:, 1:, :]     # [batch_size, seq_len-1, feature_dim]
        
        # Compute similarities
        similarities = jnp.sum(context * targets, axis=-1)  # [batch_size, seq_len-1]
        
        # Normalize by temperature
        similarities = similarities / self.cpc_temperature
        
        # InfoNCE loss (simplified)
        loss = -jnp.mean(similarities)
        
        return loss
    
    def mixup_augmentation(self, 
                          batch: Dict[str, jnp.ndarray],
                          alpha: float = 0.2,
                          key: jax.random.PRNGKey = None) -> Dict[str, jnp.ndarray]:
        """
        Mixup data augmentation.
        Executive Summary fix: mixup implementation.
        """
        if key is None:
            key = jax.random.PRNGKey(42)
            
        batch_size = batch['strain'].shape[0]
        
        # Sample mixing coefficient
        lam = jax.random.beta(key, alpha, alpha, shape=())
        
        # Generate random permutation
        indices = jax.random.permutation(jax.random.split(key)[1], batch_size)
        
        # Mix inputs
        mixed_strain = lam * batch['strain'] + (1 - lam) * batch['strain'][indices]
        
        # Mix labels (soft targets)
        labels_onehot = jax.nn.one_hot(batch['labels'], 2)  # Assuming 2 classes
        mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[indices]
        
        return {
            'strain': mixed_strain,
            'labels': mixed_labels,
            'mixup_lambda': lam
        }
    
    def train_step(self, 
                  state: train_state.TrainState,
                  batch: Dict[str, jnp.ndarray],
                  key: jax.random.PRNGKey,
                  use_mixup: bool = True) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """
        Single training step with real gradient updates.
        Executive Summary fix: real training instead of mock.
        """
        
        def loss_fn(params, batch_data, rng_key):
            """Compute total loss for the batch."""
            # Apply mixup if enabled
            if use_mixup:
                batch_data = self.mixup_augmentation(batch_data, alpha=0.2, key=rng_key)
            
            strain = batch_data['strain']
            labels = batch_data['labels']
            
            # Forward pass through CPC encoder
            cpc_output = self.cpc_encoder.apply(
                params['cpc_encoder'], strain, training=True, rngs={'dropout': rng_key}
            )
            
            # Convert to spikes
            spike_trains = self.spike_bridge.apply(
                params['spike_bridge'], cpc_output['encoded_features'], training=True
            )
            
            # SNN classification
            snn_output = self.snn_classifier.apply(
                params['snn_classifier'], spike_trains, training=True
            )
            
            # Compute losses
            # 1. CPC loss for representation learning
            cpc_loss_value = self.cpc_loss(
                cpc_output['encoded_features'],
                cpc_output['context_predictions']
            )
            
            # 2. Classification loss (focal loss for imbalance)
            if len(labels.shape) > 1:  # Already one-hot from mixup
                classification_loss_value = -jnp.mean(
                    jnp.sum(labels * jax.nn.log_softmax(snn_output['logits'], axis=-1), axis=-1)
                )
            else:
                classification_loss_value = self.focal_loss(snn_output['logits'], labels)
            
            # 3. Combined loss with adaptive weighting (Executive Summary fix)
            if self.current_epoch < self.config.get('cpc_pretrain_epochs', 20):
                # Early training: focus on CPC
                total_loss = 0.8 * cpc_loss_value + 0.2 * classification_loss_value
            else:
                # Later training: focus on classification
                total_loss = 0.3 * cpc_loss_value + 0.7 * classification_loss_value
            
            # Return losses and auxiliary information
            aux = {
                'cpc_loss': cpc_loss_value,
                'classification_loss': classification_loss_value,
                'total_loss': total_loss,
                'spike_rate': snn_output['final_spike_rate'],
                'logits': snn_output['logits']
            }
            
            return total_loss, aux
        
        # Compute gradients
        (loss_value, aux), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, batch, key
        )
        
        # Update parameters
        new_state = state.apply_gradients(grads=gradients)
        
        # Compute accuracy
        if len(batch['labels'].shape) == 1:  # Hard labels
            predictions = jnp.argmax(aux['logits'], axis=-1)
            accuracy = jnp.mean(predictions == batch['labels'])
        else:  # Soft labels from mixup
            hard_labels = jnp.argmax(batch['labels'], axis=-1)
            predictions = jnp.argmax(aux['logits'], axis=-1)
            accuracy = jnp.mean(predictions == hard_labels)
        
        # Training metrics
        metrics = {
            'loss': float(loss_value),
            'cpc_loss': float(aux['cpc_loss']),
            'classification_loss': float(aux['classification_loss']),
            'accuracy': float(accuracy),
            'spike_rate': float(aux['spike_rate']),
            'learning_rate': float(new_state.opt_state[1].hyperparams['learning_rate'])
        }
        
        return new_state, metrics
    
    def train_epoch(self, 
                   dataloader: Any,
                   key: jax.random.PRNGKey) -> Dict[str, float]:
        """
        Train for one epoch with real data processing.
        Executive Summary fix: real epoch training.
        """
        if self.train_state is None:
            raise ValueError("Training state not initialized")
            
        epoch_metrics = {
            'loss': [],
            'cpc_loss': [],
            'classification_loss': [],
            'accuracy': [],
            'spike_rate': []
        }
        
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Generate batch-specific random key
            batch_key = jax.random.fold_in(key, batch_idx)
            
            # Training step
            self.train_state, batch_metrics = self.train_step(
                self.train_state, batch, batch_key, use_mixup=True
            )
            
            # Accumulate metrics
            for metric_name in epoch_metrics:
                if metric_name in batch_metrics:
                    epoch_metrics[metric_name].append(batch_metrics[metric_name])
            
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}: loss={batch_metrics['loss']:.4f}, "
                           f"acc={batch_metrics['accuracy']:.3f}")
        
        # Average metrics over epoch
        averaged_metrics = {}
        for metric_name, values in epoch_metrics.items():
            if values:
                averaged_metrics[metric_name] = float(np.mean(values))
        
        # Update training history
        for metric_name in ['cpc_loss', 'classification_loss', 'total_loss', 'accuracy', 'spike_rate']:
            if metric_name in averaged_metrics:
                if metric_name == 'total_loss':
                    self.training_metrics['total_losses'].append(averaged_metrics['loss'])
                else:
                    self.training_metrics[f"{metric_name}s"].append(averaged_metrics[metric_name])
        
        logger.info(f"Epoch {self.current_epoch} completed: "
                   f"loss={averaged_metrics.get('loss', 0):.4f}, "
                   f"accuracy={averaged_metrics.get('accuracy', 0):.3f}")
        
        return averaged_metrics
        
    def save_checkpoint(self, filepath: Path, metadata: Dict[str, Any] = None):
        """Save training checkpoint."""
        if self.train_state is None:
            logger.warning("No training state to save")
            return
            
        checkpoint_data = {
            'params': self.train_state.params,
            'opt_state': self.train_state.opt_state,
            'step': self.train_state.step,
            'epoch': self.current_epoch,
            'training_metrics': self.training_metrics,
            'config': self.config,
            'metadata': metadata or {}
        }
        
        # Save using JAX serialization
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
        logger.info(f"Checkpoint saved: {filepath}")

# Factory function
def create_real_advanced_trainer(config: Dict[str, Any]) -> RealAdvancedGWTrainer:
    """Create real advanced trainer with validated configuration."""
    return RealAdvancedGWTrainer(config) 