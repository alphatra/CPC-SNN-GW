"""
Enhanced Spiking Neural Network (SNN) Classifier

Neuromorphic binary classifier using optimized JAX/Flax LIF neurons
for energy-efficient gravitational wave detection.

Key improvements:
- Vectorized LIF update for single fused kernel performance
- Surrogate gradient methods (fast-sigmoid, atan, piecewise)
- Batched validation with F1, AUROC, confusion matrix on GPU/TPU
- Memory-efficient implementations for long sequences
- Comprehensive metrics without host synchronization
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SurrogateGradientType(Enum):
    """Available surrogate gradient methods."""
    FAST_SIGMOID = "fast_sigmoid"
    ATAN = "atan"
    PIECEWISE = "piecewise"
    TRIANGULAR = "triangular"
    EXPONENTIAL = "exponential"


@dataclass
class SNNConfig:
    """Configuration for SNN classifier."""
    # Architecture
    hidden_size: int = 128
    num_classes: int = 2
    num_layers: int = 2
    
    # LIF parameters
    tau_mem: float = 20e-3  # Membrane time constant
    tau_syn: float = 5e-3   # Synaptic time constant
    threshold: float = 1.0  # Spike threshold
    dt: float = 1e-3        # Time step
    
    # Surrogate gradient
    surrogate_type: SurrogateGradientType = SurrogateGradientType.FAST_SIGMOID
    surrogate_beta: float = 10.0  # Surrogate gradient steepness
    
    # Training
    dropout_rate: float = 0.1
    use_batch_norm: bool = False
    
    # Optimization
    use_fused_kernel: bool = True
    memory_efficient: bool = True


def create_surrogate_gradient_fn(gradient_type: SurrogateGradientType, 
                                beta: float = 10.0) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Create surrogate gradient function.
    
    Args:
        gradient_type: Type of surrogate gradient
        beta: Steepness parameter
        
    Returns:
        Surrogate gradient function
    """
    if gradient_type == SurrogateGradientType.FAST_SIGMOID:
        def fast_sigmoid(x):
            return 1.0 / (1.0 + jnp.abs(beta * x))
        return fast_sigmoid
    
    elif gradient_type == SurrogateGradientType.ATAN:
        def atan_surrogate(x):
            return beta / (1.0 + (beta * x)**2)
        return atan_surrogate
    
    elif gradient_type == SurrogateGradientType.PIECEWISE:
        def piecewise_surrogate(x):
            return jnp.where(
                jnp.abs(x) < 1.0 / beta,
                beta * (1.0 - jnp.abs(beta * x)),
                0.0
            )
        return piecewise_surrogate
    
    elif gradient_type == SurrogateGradientType.TRIANGULAR:
        def triangular_surrogate(x):
            return jnp.maximum(0.0, 1.0 - jnp.abs(beta * x))
        return triangular_surrogate
    
    elif gradient_type == SurrogateGradientType.EXPONENTIAL:
        def exponential_surrogate(x):
            return beta * jnp.exp(-beta * jnp.abs(x))
        return exponential_surrogate
    
    else:
        raise ValueError(f"Unknown surrogate gradient type: {gradient_type}")


def spike_function_with_surrogate(v_mem: jnp.ndarray, 
                                 threshold: float,
                                 surrogate_fn: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """
    Spike function with surrogate gradient.
    
    Forward pass: Heaviside step function
    Backward pass: Surrogate gradient
    
    Args:
        v_mem: Membrane potential
        threshold: Spike threshold
        surrogate_fn: Surrogate gradient function
        
    Returns:
        Spikes with surrogate gradient
    """
    # Forward pass: step function
    spikes = (v_mem > threshold).astype(jnp.float32)
    
    # Backward pass: surrogate gradient
    # Use stop_gradient to prevent gradients through forward pass
    spikes_sg = jax.lax.stop_gradient(spikes)
    
    # Add surrogate gradient
    v_shifted = v_mem - threshold
    surrogate_grad = surrogate_fn(v_shifted)
    
    # Combine: forward pass result + surrogate gradient
    return spikes_sg + surrogate_grad - jax.lax.stop_gradient(surrogate_grad)


class VectorizedLIFLayer(nn.Module):
    """
    Vectorized LIF layer optimized for single fused kernel.
    
    All operations are vectorized across batch and time dimensions
    for maximum GPU/TPU efficiency.
    """
    
    config: SNNConfig
    
    def setup(self):
        """Initialize surrogate gradient function."""
        self.surrogate_fn = create_surrogate_gradient_fn(
            self.config.surrogate_type, 
            self.config.surrogate_beta
        )
        
        # Precompute decay factors
        self.alpha_mem = jnp.exp(-self.config.dt / self.config.tau_mem)
        self.alpha_syn = jnp.exp(-self.config.dt / self.config.tau_syn)
    
    @nn.compact
    def __call__(self, spikes: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Vectorized LIF forward pass.
        
        Args:
            spikes: Input spikes [batch, time, input_dim]
            training: Training mode flag
            
        Returns:
            output_spikes: Output spikes [batch, time, hidden_size]
        """
        batch_size, time_steps, input_dim = spikes.shape
        
        # Synaptic weights
        W = self.param('kernel', nn.initializers.xavier_uniform(), (input_dim, self.config.hidden_size))
        b = self.param('bias', nn.initializers.zeros, (self.config.hidden_size,))
        
        # Always use optimized, adaptive strategy
        return self._optimized_lif_forward(spikes, W, b, training)
    
    def _optimized_lif_forward(self, spikes: jnp.ndarray, W: jnp.ndarray, b: jnp.ndarray, training: bool) -> jnp.ndarray:
        """
        Optimized LIF implementation with adaptive strategy selection.
        
        Uses different strategies based on sequence length:
        - Short sequences (< 32): Unrolled loop
        - Medium sequences (32-128): lax.scan
        - Long sequences (> 128): Chunked processing
        """
        batch_size, time_steps, input_dim = spikes.shape
        
        # Adaptive strategy selection
        if time_steps < 32:
            # For very short sequences, unrolled loop can be faster
            return self._unrolled_lif_forward(spikes, W, b, training)
        elif time_steps <= 128:
            # Use lax.scan for medium sequences
            return self._scan_lif_forward(spikes, W, b, training)
        else:
            # For very long sequences, use chunked processing
            return self._chunked_lif_forward(spikes, W, b, training)
    
    def _unrolled_lif_forward(self, spikes: jnp.ndarray, W: jnp.ndarray, b: jnp.ndarray, training: bool) -> jnp.ndarray:
        """
        Unrolled LIF implementation for very short sequences.
        """
        batch_size, time_steps, input_dim = spikes.shape
        
        # Compute synaptic inputs for all time steps
        synaptic_inputs = jnp.dot(spikes, W) + b
        
        # Initialize states
        v_mem = jnp.zeros((batch_size, self.config.hidden_size))
        i_syn = jnp.zeros((batch_size, self.config.hidden_size))
        
        # Create output array
        output_spikes = jnp.zeros((batch_size, time_steps, self.config.hidden_size))
        
        # Unroll manually for very short sequences (JIT will optimize this)
        def single_step(t, carry):
            v_mem_t, i_syn_t, outputs = carry
            
            # Update synaptic current
            i_syn_new = self.alpha_syn * i_syn_t + synaptic_inputs[:, t, :]
            
            # Update membrane potential
            v_mem_new = self.alpha_mem * v_mem_t + i_syn_new
            
            # Generate spikes
            spikes_out = spike_function_with_surrogate(v_mem_new, self.config.threshold, self.surrogate_fn)
            
            # Reset membrane potential
            v_mem_reset = jnp.where(spikes_out > 0.5, 0.0, v_mem_new)
            
            # Update outputs
            outputs = outputs.at[:, t, :].set(spikes_out)
            
            return (v_mem_reset, i_syn_new, outputs)
        
        # Run unrolled loop
        initial_carry = (v_mem, i_syn, output_spikes)
        _, _, final_outputs = jax.lax.fori_loop(0, time_steps, single_step, initial_carry)
        
        return final_outputs
    
    def _chunked_lif_forward(self, spikes: jnp.ndarray, W: jnp.ndarray, b: jnp.ndarray, training: bool) -> jnp.ndarray:
        """
        Chunked LIF implementation for very long sequences.
        """
        batch_size, time_steps, input_dim = spikes.shape
        chunk_size = 64  # Process in chunks of 64 time steps
        
        # Initialize states
        v_mem = jnp.zeros((batch_size, self.config.hidden_size))
        i_syn = jnp.zeros((batch_size, self.config.hidden_size))
        
        # Process in chunks
        output_chunks = []
        
        for start_idx in range(0, time_steps, chunk_size):
            end_idx = min(start_idx + chunk_size, time_steps)
            chunk_spikes = spikes[:, start_idx:end_idx, :]
            
            # Process chunk using scan
            chunk_synaptic_inputs = jnp.dot(chunk_spikes, W) + b
            
            def lif_step(carry, synaptic_input_t):
                v_mem_t, i_syn_t = carry
                
                # Update synaptic current
                i_syn_new = self.alpha_syn * i_syn_t + synaptic_input_t
                
                # Update membrane potential
                v_mem_new = self.alpha_mem * v_mem_t + i_syn_new
                
                # Generate spikes
                spikes_out = spike_function_with_surrogate(v_mem_new, self.config.threshold, self.surrogate_fn)
                
                # Reset membrane potential
                v_mem_reset = jnp.where(spikes_out > 0.5, 0.0, v_mem_new)
                
                return (v_mem_reset, i_syn_new), spikes_out
            
            # Run scan on chunk
            synaptic_inputs_t = jnp.transpose(chunk_synaptic_inputs, (1, 0, 2))
            initial_carry = (v_mem, i_syn)
            (v_mem, i_syn), chunk_outputs = jax.lax.scan(lif_step, initial_carry, synaptic_inputs_t)
            
            # Transpose back and append
            chunk_outputs = jnp.transpose(chunk_outputs, (1, 0, 2))
            output_chunks.append(chunk_outputs)
        
        # Concatenate all chunks
        output_spikes = jnp.concatenate(output_chunks, axis=1)
        
        return output_spikes
    
    def _scan_lif_forward(self, spikes: jnp.ndarray, W: jnp.ndarray, b: jnp.ndarray, training: bool) -> jnp.ndarray:
        """
        Scan-based LIF implementation (fallback for very long sequences).
        """
        batch_size, time_steps, input_dim = spikes.shape
        
        # Initial states
        v_mem = jnp.zeros((batch_size, self.config.hidden_size))
        i_syn = jnp.zeros((batch_size, self.config.hidden_size))
        
        def lif_step(carry, spike_t):
            v_mem_t, i_syn_t = carry
            
            # Update synaptic current
            i_syn_new = self.alpha_syn * i_syn_t + jnp.dot(spike_t, W) + b
            
            # Update membrane potential
            v_mem_new = self.alpha_mem * v_mem_t + i_syn_new
            
            # Generate spikes with surrogate gradient
            spikes_out = spike_function_with_surrogate(v_mem_new, self.config.threshold, self.surrogate_fn)
            
            # Reset membrane potential
            v_mem_reset = jnp.where(spikes_out > 0.5, 0.0, v_mem_new)
            
            return (v_mem_reset, i_syn_new), spikes_out
        
        # Apply over time dimension
        spikes_transposed = jnp.transpose(spikes, (1, 0, 2))
        carry_init = (v_mem, i_syn)
        
        final_carry, output_spikes = jax.lax.scan(lif_step, carry_init, spikes_transposed)
        
        # Transpose back to [batch, time, hidden_size]
        return jnp.transpose(output_spikes, (1, 0, 2))


class EnhancedSNNClassifier(nn.Module):
    """
    Enhanced SNN classifier with vectorized LIF and surrogate gradients.
    
    Features:
    - Vectorized LIF layers for optimal GPU/TPU performance
    - Configurable surrogate gradient methods
    - Batch normalization and dropout support
    - Memory-efficient implementations
    """
    
    config: SNNConfig
    
    @nn.compact
    def __call__(self, spikes: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Enhanced SNN forward pass.
        
        Args:
            spikes: Input spike trains [batch, time, input_dim]
            training: Training mode flag
            
        Returns:
            logits: Classification logits [batch, num_classes]
        """
        x = spikes
        
        # Multiple LIF layers
        for i in range(self.config.num_layers):
            hidden_size = self.config.hidden_size // (2**i) if i > 0 else self.config.hidden_size
            
            # Create layer-specific config
            layer_config = SNNConfig(
                hidden_size=hidden_size,
                tau_mem=self.config.tau_mem,
                tau_syn=self.config.tau_syn,
                threshold=self.config.threshold,
                dt=self.config.dt,
                surrogate_type=self.config.surrogate_type,
                surrogate_beta=self.config.surrogate_beta,
                use_fused_kernel=self.config.use_fused_kernel,
                memory_efficient=self.config.memory_efficient
            )
            
            # LIF layer
            x = VectorizedLIFLayer(config=layer_config)(x, training=training)
            
            # Optional batch normalization
            if self.config.use_batch_norm:
                x = nn.BatchNorm(use_running_average=not training)(x)
            
            # Optional dropout
            if self.config.dropout_rate > 0.0 and training:
                x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=not training)
        
        # Global pooling (multiple strategies)
        if self.config.memory_efficient:
            # Simple mean pooling
            x_pooled = jnp.mean(x, axis=1)
        else:
            # More sophisticated pooling
            x_mean = jnp.mean(x, axis=1)
            x_max = jnp.max(x, axis=1)
            x_pooled = jnp.concatenate([x_mean, x_max], axis=-1)
        
        # Final classification layer
        logits = nn.Dense(self.config.num_classes, kernel_init=nn.initializers.xavier_uniform())(x_pooled)
        
        return logits


class BatchedSNNValidator:
    """
    Batched validation for SNN with comprehensive metrics.
    
    Computes F1, AUROC, confusion matrix on GPU/TPU without host sync.
    """
    
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
    
    def compute_metrics(self, logits: jnp.ndarray, labels: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Compute comprehensive metrics in batched fashion.
        
        Args:
            logits: Model predictions [batch, num_classes]
            labels: Ground truth labels [batch]
            
        Returns:
            Dictionary of metrics
        """
        predictions = jnp.argmax(logits, axis=1)
        probabilities = nn.softmax(logits, axis=1)
        
        metrics = {}
        
        # Basic accuracy
        metrics['accuracy'] = jnp.mean(predictions == labels)
        
        # Confusion matrix
        metrics['confusion_matrix'] = self._compute_confusion_matrix(predictions, labels)
        
        # Per-class metrics
        if self.num_classes == 2:
            # Binary classification metrics
            metrics.update(self._compute_binary_metrics(predictions, labels, probabilities))
        else:
            # Multi-class metrics
            metrics.update(self._compute_multiclass_metrics(predictions, labels, probabilities))
        
        return metrics
    
    def _compute_confusion_matrix(self, predictions: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        """Compute confusion matrix without host sync using vectorized operations."""
        # ✅ Single vectorized operation instead of double loop
        cm = jnp.zeros((self.num_classes, self.num_classes), dtype=jnp.int32)
        indices = (labels, predictions)
        return cm.at[indices].add(1)
    
    def _compute_binary_metrics(self, predictions: jnp.ndarray, labels: jnp.ndarray, probabilities: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Compute binary classification metrics."""
        # True positives, false positives, etc.
        tp = jnp.sum((predictions == 1) & (labels == 1))
        fp = jnp.sum((predictions == 1) & (labels == 0))
        tn = jnp.sum((predictions == 0) & (labels == 0))
        fn = jnp.sum((predictions == 0) & (labels == 1))
        
        # Precision, recall, F1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # AUROC approximation (trapezoidal rule)
        auroc = self._compute_auroc(probabilities[:, 1], labels)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auroc': auroc,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
    
    def _compute_multiclass_metrics(self, predictions: jnp.ndarray, labels: jnp.ndarray, probabilities: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Compute multi-class metrics."""
        # Macro-averaged F1
        f1_scores = []
        
        for class_idx in range(self.num_classes):
            # One-vs-rest for each class
            class_predictions = (predictions == class_idx)
            class_labels = (labels == class_idx)
            
            tp = jnp.sum(class_predictions & class_labels)
            fp = jnp.sum(class_predictions & ~class_labels)
            fn = jnp.sum(~class_predictions & class_labels)
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            f1_scores.append(f1)
        
        macro_f1 = jnp.mean(jnp.array(f1_scores))
        
        return {
            'macro_f1': macro_f1,
            'per_class_f1': jnp.array(f1_scores)
        }
    
    def _compute_auroc(self, scores: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        """
        Compute AUROC using trapezoidal rule approximation.
        
        This is an approximation that works on GPU/TPU without host sync.
        """
        # Sort scores and labels
        sorted_indices = jnp.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        # Compute TPR and FPR at different thresholds
        n_pos = jnp.sum(sorted_labels)
        n_neg = len(sorted_labels) - n_pos
        
        # Cumulative sums
        tp_cumsum = jnp.cumsum(sorted_labels)
        fp_cumsum = jnp.cumsum(1 - sorted_labels)
        
        # TPR and FPR
        tpr = tp_cumsum / (n_pos + 1e-8)
        fpr = fp_cumsum / (n_neg + 1e-8)
        
        # Trapezoidal integration
        auroc = jnp.trapz(tpr, fpr)
        
        return auroc
    
    def validation_step(self, model: nn.Module, params: Dict, batch: jnp.ndarray, labels: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Complete validation step with all metrics.
        
        Args:
            model: SNN model
            params: Model parameters
            batch: Input batch
            labels: Ground truth labels
            
        Returns:
            Dictionary of validation metrics
        """
        # Forward pass
        logits = model.apply({'params': params}, batch, training=False)
        
        # Compute loss
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        
        # Compute metrics
        metrics = self.compute_metrics(logits, labels)
        metrics['loss'] = loss
        
        return metrics


# Enhanced factory functions
def create_enhanced_snn_classifier(config: Optional[SNNConfig] = None) -> EnhancedSNNClassifier:
    """Create enhanced SNN classifier with full configuration."""
    if config is None:
        config = SNNConfig()
    
    return EnhancedSNNClassifier(config=config)


def create_snn_config(
    hidden_size: int = 128,
    num_classes: int = 2,
    surrogate_type: SurrogateGradientType = SurrogateGradientType.FAST_SIGMOID,
    **kwargs
) -> SNNConfig:
    """Create SNN configuration with common overrides."""
    config = SNNConfig(
            hidden_size=hidden_size,
        num_classes=num_classes,
        surrogate_type=surrogate_type
    )
    
    # Apply any additional overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config parameter: {key}")
    
    return config


# Backward compatibility
class LIFLayer(nn.Module):
    """Backward compatible LIF layer."""
    hidden_size: int
    tau_mem: float = 20e-3
    tau_syn: float = 5e-3
    threshold: float = 1.0
    dt: float = 1e-3
    
    def setup(self):
        """Convert to enhanced implementation."""
        self.config = SNNConfig(
            hidden_size=self.hidden_size,
            tau_mem=self.tau_mem,
            tau_syn=self.tau_syn,
            threshold=self.threshold,
            dt=self.dt
        )
        self.enhanced_layer = VectorizedLIFLayer(config=self.config)
    
    @nn.compact
    def __call__(self, spikes: jnp.ndarray) -> jnp.ndarray:
        """Forward pass using enhanced layer."""
        return self.enhanced_layer(spikes, training=False)


class SNNClassifier(nn.Module):
    """Backward compatible SNN classifier."""
    hidden_size: int = 128
    num_classes: int = 2
    tau_mem: float = 20e-3
    tau_syn: float = 5e-3
    threshold: float = 1.0
    
    def setup(self):
        """Convert to enhanced implementation."""
        self.config = SNNConfig(
            hidden_size=self.hidden_size,
            num_classes=self.num_classes,
            tau_mem=self.tau_mem,
            tau_syn=self.tau_syn,
            threshold=self.threshold,
            surrogate_type=SurrogateGradientType.FAST_SIGMOID
        )
        self.enhanced_classifier = EnhancedSNNClassifier(config=self.config)
    
    @nn.compact
    def __call__(self, spikes: jnp.ndarray) -> jnp.ndarray:
        """Forward pass using enhanced classifier."""
        return self.enhanced_classifier(spikes, training=False)


def create_snn_classifier(hidden_size: int = 128, num_classes: int = 2) -> SNNClassifier:
    """Create backward compatible SNN classifier."""
    return SNNClassifier(hidden_size=hidden_size, num_classes=num_classes)


class SNNTrainer:
    """Enhanced SNN trainer with batched validation."""
    
    def __init__(self, 
                 snn_model: nn.Module,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 num_classes: int = 2):
        self.snn_model = snn_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        
        # Create optimizer
        self.optimizer = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create validator
        self.validator = BatchedSNNValidator(num_classes=num_classes)
    
    def create_train_state(self, key: jax.random.PRNGKey, sample_input: jnp.ndarray):
        """Initialize training state."""
        variables = self.snn_model.init(key, sample_input, training=True)
        state = self.optimizer.init(variables['params'])
        return variables['params'], state
    
    def train_step(self, params, opt_state, batch, labels):
        """Enhanced training step."""
        def loss_fn(params):
            logits = self.snn_model.apply({'params': params}, batch, training=True)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            return jnp.mean(loss), logits
        
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        # Calculate accuracy
        predictions = jnp.argmax(logits, axis=1)
        accuracy = jnp.mean(predictions == labels)
        
        return params, opt_state, loss, accuracy
    
    def validation_step(self, params, batch, labels):
        """Enhanced validation step with comprehensive metrics."""
        return self.validator.validation_step(self.snn_model, params, batch, labels)


# Enhanced test functions
def test_vectorized_lif_layer():
    """Test vectorized LIF layer."""
    print("Testing Vectorized LIF Layer...")
    
    try:
        # Create test configuration
        config = create_snn_config(
            hidden_size=32,
            surrogate_type=SurrogateGradientType.FAST_SIGMOID
        )
        
        # Create test data
        key = jax.random.PRNGKey(42)
        batch_size, time_steps, input_dim = 4, 50, 64
        spikes = jax.random.bernoulli(key, 0.1, (batch_size, time_steps, input_dim)).astype(jnp.float32)
        
        # Create vectorized LIF layer
        lif = VectorizedLIFLayer(config=config)
        
        # Initialize and test
        params = lif.init(key, spikes, training=True)
        output = lif.apply(params, spikes, training=True)
        
        print(f"✅ Input shape: {spikes.shape}")
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Output spikes mean: {jnp.mean(output):.4f}")
        print(f"✅ Surrogate gradient: {config.surrogate_type.value}")
        
        # Test gradient computation
        def loss_fn(params):
            out = lif.apply(params, spikes, training=True)
            return jnp.mean(out**2)
        
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)
        
        print(f"✅ Gradient computation successful")
        print(f"✅ Gradient norm: {jnp.linalg.norm(jax.tree_flatten(grads)[0][0]):.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vectorized LIF layer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_snn_classifier():
    """Test enhanced SNN classifier."""
    print("Testing Enhanced SNN Classifier...")
    
    try:
        # Create test configuration
        config = create_snn_config(
            hidden_size=64,
            num_classes=3,
            num_layers=2,
            surrogate_type=SurrogateGradientType.ATAN,
            memory_efficient=True
        )
        
        # Create test data
        key = jax.random.PRNGKey(42)
        batch_size, time_steps, input_dim = 4, 50, 64
        spikes = jax.random.bernoulli(key, 0.1, (batch_size, time_steps, input_dim)).astype(jnp.float32)
        labels = jax.random.randint(key, (batch_size,), 0, 3)
        
        # Create enhanced SNN classifier
        snn = create_enhanced_snn_classifier(config)
        
        # Initialize and test
        params = snn.init(key, spikes, training=True)
        logits = snn.apply(params, spikes, training=True)
        
        print(f"✅ Input shape: {spikes.shape}")
        print(f"✅ Logits shape: {logits.shape}")
        print(f"✅ Config: {config.surrogate_type.value}, memory_efficient={config.memory_efficient}")
        print(f"✅ Logits range: [{jnp.min(logits):.3f}, {jnp.max(logits):.3f}]")
        
        # Test enhanced trainer
        trainer = SNNTrainer(snn, num_classes=3)
        params, opt_state = trainer.create_train_state(key, spikes)
        params, opt_state, loss, accuracy = trainer.train_step(params, opt_state, spikes, labels)
        
        print(f"✅ Training loss: {loss:.4f}")
        print(f"✅ Training accuracy: {accuracy:.4f}")
        
        # Test validation metrics
        val_metrics = trainer.validation_step(params, spikes, labels)
        print(f"✅ Validation metrics: {list(val_metrics.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced SNN classifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_surrogate_gradients():
    """Test surrogate gradient functions."""
    print("Testing Surrogate Gradients...")
    
    try:
        # Test all surrogate gradient types
        x = jnp.linspace(-2, 2, 100)
        
        for grad_type in SurrogateGradientType:
            surrogate_fn = create_surrogate_gradient_fn(grad_type, beta=10.0)
            grad_values = surrogate_fn(x)
            
            print(f"✅ {grad_type.value}: range [{jnp.min(grad_values):.3f}, {jnp.max(grad_values):.3f}]")
            
            # Test with spike function
            v_mem = jnp.array([0.5, 1.5, -0.5, 2.0])
            spikes = spike_function_with_surrogate(v_mem, 1.0, surrogate_fn)
            
            print(f"   Spike output: {spikes}")
        
        return True
        
    except Exception as e:
        print(f"❌ Surrogate gradients test failed: {e}")
        return False


def test_batched_validation():
    """Test batched validation metrics."""
    print("Testing Batched Validation...")
    
    try:
        # Create test data
        key = jax.random.PRNGKey(42)
        batch_size, num_classes = 100, 3
        
        logits = jax.random.normal(key, (batch_size, num_classes))
        labels = jax.random.randint(key, (batch_size,), 0, num_classes)
        
        # Create validator
        validator = BatchedSNNValidator(num_classes=num_classes)
        
        # Compute metrics
        metrics = validator.compute_metrics(logits, labels)
        
        print(f"✅ Computed metrics: {list(metrics.keys())}")
        print(f"✅ Accuracy: {metrics['accuracy']:.4f}")
        print(f"✅ Confusion matrix shape: {metrics['confusion_matrix'].shape}")
        
        if 'macro_f1' in metrics:
            print(f"✅ Macro F1: {metrics['macro_f1']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batched validation test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧠 Testing Enhanced SNN Classifier Implementation...")
    print()
    
    success1 = test_vectorized_lif_layer()
    print()
    success2 = test_enhanced_snn_classifier()
    print()
    success3 = test_surrogate_gradients()
    print()
    success4 = test_batched_validation()
    print()
    
    overall_success = success1 and success2 and success3 and success4
    print(f"🎯 Overall: {'SUCCESS' if overall_success else 'FAILED'}")
    
    if overall_success:
        print("🎉 All enhanced SNN classifier tests passed!")
    else:
        print("❌ Some tests failed - check implementation") 