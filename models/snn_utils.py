"""
SNN Utilities: Surrogate Gradients and Validation Metrics

Utility functions for Spiking Neural Networks:
- Enhanced adaptive surrogate gradient functions for improved backpropagation
- Batched validation metrics (F1, AUROC, confusion matrix)
- Performance optimization utilities
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Dict, Optional
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
    # ✅ NEW: Enhanced adaptive surrogate
    ADAPTIVE_MULTI_SCALE = "adaptive_multi_scale"


def create_enhanced_surrogate_gradient_fn(membrane_potential: Optional[jnp.ndarray] = None,
                                        training_progress: float = 0.0) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Create enhanced adaptive surrogate gradient function.
    
    This improves upon static surrogate gradients by:
    1. Adapting to membrane potential dynamics
    2. Progressive difficulty during training (curriculum learning)
    3. Multi-scale gradient combination
    
    Args:
        membrane_potential: Current membrane potential for adaptive scaling
        training_progress: Progress through training (0.0 to 1.0)
        
    Returns:
        Enhanced adaptive surrogate gradient function
    """
    def adaptive_multi_scale_surrogate(x: jnp.ndarray) -> jnp.ndarray:
        """
        🚀 ENHANCED: Multi-scale adaptive surrogate gradient.
        
        Combines multiple surrogate types with adaptive weighting based on:
        - Training progress (curriculum learning)
        - Membrane potential dynamics (biological realism) 
        - Multi-scale temporal features
        """
        # Base surrogate gradients with different characteristics
        sigmoid_grad = 10.0 / (1.0 + jnp.abs(10.0 * x))  # Smooth, wide
        triangular_grad = jnp.maximum(0.0, 1.0 - jnp.abs(4.0 * x))  # Sharp, localized
        exponential_grad = 3.0 * jnp.exp(-3.0 * jnp.abs(x))  # Biological-like decay
        
        # 🎯 CURRICULUM LEARNING: Adaptive weighting based on training progress
        # Early training: Favor wide, smooth gradients for exploration
        early_weight = jnp.maximum(0.0, 1.0 - 2.0 * training_progress)
        # Mid training: Balanced combination
        mid_weight = 4.0 * training_progress * (1.0 - training_progress)  # Bell curve
        # Late training: Favor sharp, precise gradients
        late_weight = jnp.maximum(0.0, 2.0 * training_progress - 1.0)
        
        # 🧠 MEMBRANE-POTENTIAL ADAPTIVE SCALING
        if membrane_potential is not None:
            # Scale based on membrane potential dynamics
            membrane_scale = jnp.tanh(jnp.abs(membrane_potential.mean()))
            # Near threshold: favor precise gradients
            # Far from threshold: favor exploratory gradients
            precision_factor = 1.0 + membrane_scale
        else:
            precision_factor = 1.0
        
        # 🔄 MULTI-SCALE COMBINATION with adaptive weights
        combined_gradient = (
            early_weight * sigmoid_grad +           # Exploration phase
            mid_weight * triangular_grad +          # Balanced phase  
            late_weight * exponential_grad +       # Precision phase
            0.1 * (sigmoid_grad * triangular_grad)  # Nonlinear interaction
        ) * precision_factor
        
        return combined_gradient
    
    return adaptive_multi_scale_surrogate


def create_surrogate_gradient_fn(gradient_type: SurrogateGradientType, 
                                beta: float = 10.0,
                                membrane_potential: Optional[jnp.ndarray] = None,
                                training_progress: float = 0.0) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Create surrogate gradient function with enhanced adaptive option.
    
    Args:
        gradient_type: Type of surrogate gradient
        beta: Steepness parameter for traditional methods
        membrane_potential: Current membrane potential for adaptive methods
        training_progress: Training progress for curriculum learning
        
    Returns:
        Surrogate gradient function
    """
    if gradient_type == SurrogateGradientType.ADAPTIVE_MULTI_SCALE:
        return create_enhanced_surrogate_gradient_fn(membrane_potential, training_progress)
    
    elif gradient_type == SurrogateGradientType.FAST_SIGMOID:
        def fast_sigmoid(x):
            sigmoid_x = 1.0 / (1.0 + jnp.exp(-beta * x))
            return beta * sigmoid_x * (1.0 - sigmoid_x)
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
            return jnp.exp(-beta * jnp.abs(x))
        return exponential_surrogate
    
    else:
        raise ValueError(f"Unknown surrogate gradient type: {gradient_type}")


def spike_function_with_surrogate(v_mem: jnp.ndarray, 
                                 threshold: float,
                                 surrogate_fn: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """
    Spike function with surrogate gradient for backpropagation.
    
    Forward pass: Heaviside step function
    Backward pass: Smooth surrogate gradient
    
    Args:
        v_mem: Membrane potential
        threshold: Spike threshold
        surrogate_fn: Surrogate gradient function
        
    Returns:
        Binary spike output with surrogate gradients
    """
    # Forward pass: threshold crossing detection
    spikes = (v_mem >= threshold).astype(jnp.float32)
    
    # Backward pass: use surrogate gradient
    surrogate_grad = surrogate_fn(v_mem - threshold)
    
    # Straight-through estimator: forward spikes, backward surrogate
    return spikes + jax.lax.stop_gradient(spikes - surrogate_grad)


def spike_function_with_enhanced_surrogate(v_mem: jnp.ndarray,
                                         threshold: float,
                                         training_progress: float = 0.0) -> jnp.ndarray:
    """
    🚀 ENHANCED: Spike function with adaptive multi-scale surrogate gradients.
    
    This version automatically adapts the surrogate gradient based on:
    - Current membrane potential dynamics
    - Training progress for curriculum learning
    
    Args:
        v_mem: Membrane potential
        threshold: Spike threshold  
        training_progress: Current training progress (0.0 to 1.0)
        
    Returns:
        Binary spike output with enhanced adaptive surrogate gradients
    """
    # Create adaptive surrogate function based on current membrane state
    surrogate_fn = create_enhanced_surrogate_gradient_fn(
        membrane_potential=v_mem,
        training_progress=training_progress
    )
    
    return spike_function_with_surrogate(v_mem, threshold, surrogate_fn)


class BatchedSNNValidator:
    """
    Batched validation for SNN with comprehensive metrics.
    
    Computes F1, AUROC, confusion matrix on GPU/TPU without host sync.
    """
    
    def __init__(self, num_classes: int = 3):
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
            # Multi-class metrics (simplified)
            metrics['macro_f1'] = self._compute_macro_f1(predictions, labels)
        
        return metrics
    
    def _compute_confusion_matrix(self, predictions: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        """Compute confusion matrix without host sync using vectorized operations."""
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
        
        # AUROC approximation (simplified)
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
    
    def _compute_macro_f1(self, predictions: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        """Compute macro-averaged F1 score."""
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
        
        return jnp.mean(jnp.array(f1_scores))
    
    def _compute_auroc(self, scores: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        """
        Compute AUROC using trapezoidal approximation.
        
        Simplified implementation for efficiency.
        """
        # Sort by scores (descending)
        sorted_indices = jnp.argsort(-scores)
        sorted_labels = labels[sorted_indices]
        
        # Compute TPR and FPR at different thresholds
        num_positive = jnp.sum(labels)
        num_negative = len(labels) - num_positive
        
        # Cumulative sums
        true_positives = jnp.cumsum(sorted_labels)
        false_positives = jnp.cumsum(1 - sorted_labels)
        
        # TPR and FPR
        tpr = true_positives / (num_positive + 1e-8)
        fpr = false_positives / (num_negative + 1e-8)
        
        # Trapezoidal rule approximation
        return jnp.trapz(tpr, fpr)
    
    def validation_step(self, model: nn.Module, params: Dict, batch: jnp.ndarray, labels: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Single validation step with comprehensive metrics.
        
        Args:
            model: SNN model
            params: Model parameters
            batch: Input batch [batch, time, features]
            labels: Ground truth labels [batch]
            
        Returns:
            Dictionary of validation metrics
        """
        logits = model.apply(params, batch, training=False)
        
        # Compute loss
        loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        )
        
        # Compute metrics
        metrics = self.compute_metrics(logits, labels)
        metrics['loss'] = loss
        
        return metrics 