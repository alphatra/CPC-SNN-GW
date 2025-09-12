"""
SNN-specific loss functions and regularization.

This module contains loss functions for spiking neural networks
created for enhanced modularity according to the refactoring plan.

Created for finer modularity beyond the initial split.
"""

import logging
from typing import Optional
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def spike_reg_loss(spike_trains: jnp.ndarray, 
                  target_rate: float = 0.1,
                  regularization_weight: float = 0.01) -> jnp.ndarray:
    """
    Spike rate regularization loss.
    
    Encourages reasonable spike rates in SNN layers.
    
    Args:
        spike_trains: Spike trains [batch, time, neurons]
        target_rate: Target spike rate
        regularization_weight: Regularization strength
        
    Returns:
        Spike rate regularization loss
    """
    # Calculate actual spike rate
    actual_rate = jnp.mean(spike_trains)
    
    # L2 penalty for deviation from target
    rate_penalty = (actual_rate - target_rate) ** 2
    
    return regularization_weight * rate_penalty


def focal_loss(logits: jnp.ndarray, 
               labels: jnp.ndarray,
               alpha: float = 0.25,
               gamma: float = 2.0) -> jnp.ndarray:
    """
    Focal loss for addressing class imbalance in SNN classification.
    
    Args:
        logits: Model logits [batch, num_classes]
        labels: True labels [batch]
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
        
    Returns:
        Focal loss value
    """
    # Convert to probabilities
    probs = jax.nn.softmax(logits, axis=-1)
    
    # Get probability of true class
    true_class_probs = probs[jnp.arange(len(labels)), labels]
    
    # Focal loss formula: -α(1-p)^γ log(p)
    focal_weight = alpha * jnp.power(1 - true_class_probs, gamma)
    cross_entropy = -jnp.log(true_class_probs + 1e-8)
    
    return jnp.mean(focal_weight * cross_entropy)


def membrane_potential_regularization(membrane_potentials: jnp.ndarray,
                                    target_mean: float = 0.5,
                                    target_std: float = 0.2) -> jnp.ndarray:
    """
    Regularization for membrane potential distributions.
    
    Encourages healthy membrane potential statistics.
    
    Args:
        membrane_potentials: Membrane potentials [batch, time, neurons]
        target_mean: Target mean membrane potential
        target_std: Target standard deviation
        
    Returns:
        Membrane potential regularization loss
    """
    # Calculate statistics
    actual_mean = jnp.mean(membrane_potentials)
    actual_std = jnp.std(membrane_potentials)
    
    # L2 penalties
    mean_penalty = (actual_mean - target_mean) ** 2
    std_penalty = (actual_std - target_std) ** 2
    
    return mean_penalty + std_penalty


def temporal_consistency_loss(spike_trains: jnp.ndarray,
                            consistency_weight: float = 0.01) -> jnp.ndarray:
    """
    Temporal consistency loss for spike trains.
    
    Encourages smooth temporal evolution of spike patterns.
    
    Args:
        spike_trains: Spike trains [batch, time, neurons]
        consistency_weight: Consistency regularization weight
        
    Returns:
        Temporal consistency loss
    """
    if spike_trains.shape[1] < 2:
        return jnp.array(0.0)  # Need at least 2 time steps
    
    # Calculate temporal differences
    temporal_diffs = jnp.diff(spike_trains, axis=1)  # [batch, time-1, neurons]
    
    # L2 penalty for large temporal changes
    consistency_penalty = jnp.mean(temporal_diffs ** 2)
    
    return consistency_weight * consistency_penalty


def snn_combined_loss(logits: jnp.ndarray,
                     labels: jnp.ndarray,
                     spike_trains: Optional[jnp.ndarray] = None,
                     membrane_potentials: Optional[jnp.ndarray] = None,
                     loss_weights: Optional[Dict[str, float]] = None) -> jnp.ndarray:
    """
    Combined loss function for SNN training.
    
    Combines classification loss with SNN-specific regularization terms.
    
    Args:
        logits: Model logits
        labels: True labels
        spike_trains: Spike trains for regularization
        membrane_potentials: Membrane potentials for regularization
        loss_weights: Weights for different loss components
        
    Returns:
        Combined loss value
    """
    if loss_weights is None:
        loss_weights = {
            'classification': 1.0,
            'spike_reg': 0.01,
            'membrane_reg': 0.005,
            'temporal_consistency': 0.01
        }
    
    # Classification loss (focal loss for imbalance)
    classification_loss = focal_loss(logits, labels)
    total_loss = loss_weights['classification'] * classification_loss
    
    # Spike rate regularization
    if spike_trains is not None:
        spike_loss = spike_reg_loss(spike_trains)
        total_loss += loss_weights['spike_reg'] * spike_loss
    
    # Membrane potential regularization
    if membrane_potentials is not None:
        membrane_loss = membrane_potential_regularization(membrane_potentials)
        total_loss += loss_weights['membrane_reg'] * membrane_loss
        
        # Temporal consistency
        temporal_loss = temporal_consistency_loss(spike_trains if spike_trains is not None else membrane_potentials)
        total_loss += loss_weights['temporal_consistency'] * temporal_loss
    
    return total_loss


# Export SNN loss functions
__all__ = [
    "spike_reg_loss",
    "focal_loss",
    "membrane_potential_regularization",
    "temporal_consistency_loss",
    "snn_combined_loss"
]
