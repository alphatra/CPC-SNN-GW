"""
InfoNCE and contrastive loss implementations for CPC.

This module contains all loss functions extracted from
cpc_losses.py for better modularity:
- info_nce_loss: Basic InfoNCE implementation
- enhanced_info_nce_loss: Enhanced version with stability
- temporal_info_nce_loss: Temporal-aware InfoNCE
- advanced_info_nce_loss_with_momentum: Advanced version with momentum

Split from cpc_losses.py for better maintainability.
"""

import logging
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def info_nce_loss(z_context: jnp.ndarray, z_target: jnp.ndarray, temperature: float = 0.1) -> jnp.ndarray:
    """
    Basic InfoNCE (Noise Contrastive Estimation) loss for CPC.
    
    Args:
        z_context: Context embeddings [batch_size, context_dim]
        z_target: Target embeddings [batch_size, target_dim]  
        temperature: Temperature parameter for softmax
        
    Returns:
        InfoNCE loss value
    """
    batch_size = z_context.shape[0]
    
    # L2 normalize embeddings
    z_context = z_context / (jnp.linalg.norm(z_context, axis=-1, keepdims=True) + 1e-8)
    z_target = z_target / (jnp.linalg.norm(z_target, axis=-1, keepdims=True) + 1e-8)
    
    # Compute similarity matrix
    similarity_matrix = jnp.dot(z_context, z_target.T) / temperature
    
    # Labels for positive pairs (diagonal)
    labels = jnp.arange(batch_size)
    
    # Cross-entropy loss
    log_softmax = jax.nn.log_softmax(similarity_matrix, axis=1)
    loss = -jnp.mean(log_softmax[labels, labels])
    
    return loss


def enhanced_info_nce_loss(z_context: jnp.ndarray, 
                          z_target: jnp.ndarray, 
                          temperature: float = 0.1,
                          eps: float = 1e-8) -> jnp.ndarray:
    """
    Enhanced InfoNCE loss with improved numerical stability and handling.
    
    Key improvements:
    - Better handling of small batch sizes
    - Numerical stability with epsilon
    - Gradient flow optimization
    - NaN/Inf protection
    
    Args:
        z_context: Context embeddings [batch_size, context_dim]
        z_target: Target embeddings [batch_size, target_dim]
        temperature: Temperature for softmax scaling
        eps: Epsilon for numerical stability
        
    Returns:
        Enhanced InfoNCE loss value
    """
    # ✅ INPUT VALIDATION
    if z_context.size == 0 or z_target.size == 0:
        logger.warning("Empty input to enhanced_info_nce_loss")
        return jnp.array(0.0)
    
    batch_size = z_context.shape[0]
    
    # Handle single sample case
    if batch_size == 1:
        logger.debug("Single sample batch - using temporal InfoNCE")
        return temporal_info_nce_loss(
            jnp.expand_dims(z_context, axis=1),  # Add time dimension
            temperature=temperature
        )
    
    # ✅ NORMALIZATION: Stable L2 normalization
    context_norm = jnp.linalg.norm(z_context, axis=-1, keepdims=True)
    target_norm = jnp.linalg.norm(z_target, axis=-1, keepdims=True)
    
    # Prevent division by zero
    context_norm = jnp.maximum(context_norm, eps)
    target_norm = jnp.maximum(target_norm, eps)
    
    z_context_normalized = z_context / context_norm
    z_target_normalized = z_target / target_norm
    
    # ✅ SIMILARITY: Compute similarity matrix with stability
    similarity_matrix = jnp.dot(z_context_normalized, z_target_normalized.T)
    
    # Scale by temperature with clipping to prevent overflow
    scaled_similarity = jnp.clip(similarity_matrix / temperature, -50.0, 50.0)
    
    # ✅ LOSS COMPUTATION: Stable InfoNCE
    labels = jnp.arange(batch_size)
    
    # Log-sum-exp trick for numerical stability
    max_sim = jnp.max(scaled_similarity, axis=1, keepdims=True)
    exp_sim = jnp.exp(scaled_similarity - max_sim)
    log_sum_exp = jnp.log(jnp.sum(exp_sim, axis=1) + eps) + jnp.squeeze(max_sim, axis=1)
    
    # InfoNCE loss
    positive_similarities = scaled_similarity[labels, labels]
    loss = jnp.mean(log_sum_exp - positive_similarities)
    
    # ✅ VALIDATION: Check for NaN/Inf
    if jnp.isnan(loss) or jnp.isinf(loss):
        logger.error("NaN or Inf detected in enhanced_info_nce_loss")
        return jnp.array(0.0)
    
    return loss


def temporal_info_nce_loss(cpc_features: jnp.ndarray,
                          temperature: float = 0.1,
                          max_prediction_steps: int = 12) -> jnp.ndarray:
    """
    Temporal InfoNCE loss for sequences with temporal prediction.
    
    Optimized for temporal sequences where positive pairs are temporally related.
    
    Args:
        cpc_features: CPC features [batch_size, time_steps, feature_dim]
        temperature: Temperature parameter
        max_prediction_steps: Maximum steps to predict ahead
        
    Returns:
        Temporal InfoNCE loss
    """
    if cpc_features.shape[1] <= max_prediction_steps:
        logger.warning("Sequence too short for temporal InfoNCE")
        return jnp.array(0.0)
    
    batch_size, time_steps, feature_dim = cpc_features.shape
    total_loss = 0.0
    num_predictions = 0
    
    # ✅ TEMPORAL PREDICTION: Predict multiple steps ahead
    for k in range(1, min(max_prediction_steps + 1, time_steps)):
        if time_steps - k <= 0:
            continue
        
        # Context and target sequences
        context_seq = cpc_features[:, :-k, :]  # [batch, time-k, features]
        target_seq = cpc_features[:, k:, :]    # [batch, time-k, features]
        
        # Flatten temporal dimension for contrastive learning
        context_flat = context_seq.reshape(-1, feature_dim)
        target_flat = target_seq.reshape(-1, feature_dim)
        
        if context_flat.shape[0] > 1:  # Need multiple samples for contrastive learning
            # Enhanced InfoNCE for this temporal offset
            step_loss = enhanced_info_nce_loss(
                context_flat,
                target_flat,
                temperature=temperature
            )
            
            # Weight by prediction difficulty (farther = harder)
            weight = 1.0 / k  # Give more weight to closer predictions
            total_loss += weight * step_loss
            num_predictions += weight
    
    # Average over prediction steps
    if num_predictions > 0:
        return total_loss / num_predictions
    else:
        return jnp.array(0.0)


def advanced_info_nce_loss_with_momentum(z_context: jnp.ndarray, 
                                        z_target: jnp.ndarray,
                                        momentum_miner,  # MomentumHardNegativeMiner
                                        temperature: float = 0.1,
                                        use_hard_negatives: bool = True) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Advanced InfoNCE loss with momentum-based hard negative mining.
    
    Args:
        z_context: Context embeddings
        z_target: Target embeddings
        momentum_miner: Momentum-based hard negative miner
        temperature: Temperature parameter
        use_hard_negatives: Whether to use hard negative mining
        
    Returns:
        Tuple of (loss_value, mining_statistics)
    """
    batch_size = z_context.shape[0]
    
    # Basic InfoNCE computation
    base_loss = enhanced_info_nce_loss(z_context, z_target, temperature)
    
    if not use_hard_negatives or batch_size < 4:
        # Return basic loss if hard negative mining not applicable
        return base_loss, {'hard_negatives_used': False}
    
    # ✅ HARD NEGATIVE MINING: Use momentum miner
    try:
        # Mine hard negatives using momentum miner
        hard_negatives, mining_stats = momentum_miner.mine_hard_negatives(
            z_context, z_target, temperature
        )
        
        if hard_negatives is not None:
            # Compute loss with hard negatives
            hard_negative_loss = enhanced_info_nce_loss(
                z_context, hard_negatives, temperature
            )
            
            # Combine base loss and hard negative loss
            combined_loss = 0.7 * base_loss + 0.3 * hard_negative_loss
            
            mining_stats.update({
                'hard_negatives_used': True,
                'base_loss': float(base_loss),
                'hard_negative_loss': float(hard_negative_loss),
                'combined_loss': float(combined_loss)
            })
            
            return combined_loss, mining_stats
        else:
            # Fall back to base loss
            return base_loss, {'hard_negatives_used': False, 'mining_failed': True}
            
    except Exception as e:
        logger.warning(f"Hard negative mining failed: {e}")
        return base_loss, {'hard_negatives_used': False, 'error': str(e)}


def momentum_enhanced_info_nce_loss(context_embeddings: jnp.ndarray,
                                   target_embeddings: jnp.ndarray,  
                                   momentum_queue: jnp.ndarray,
                                   temperature: float = 0.1,
                                   momentum_coefficient: float = 0.999) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    InfoNCE loss enhanced with momentum-based negative sampling.
    
    Args:
        context_embeddings: Context embeddings [batch_size, feature_dim]
        target_embeddings: Target embeddings [batch_size, feature_dim]
        momentum_queue: Queue of momentum embeddings [queue_size, feature_dim]
        temperature: Temperature parameter
        momentum_coefficient: Momentum update coefficient
        
    Returns:
        Tuple of (loss, updated_momentum_queue)
    """
    batch_size, feature_dim = context_embeddings.shape
    
    # ✅ NORMALIZATION
    context_norm = context_embeddings / (jnp.linalg.norm(context_embeddings, axis=-1, keepdims=True) + 1e-8)
    target_norm = target_embeddings / (jnp.linalg.norm(target_embeddings, axis=-1, keepdims=True) + 1e-8)
    momentum_norm = momentum_queue / (jnp.linalg.norm(momentum_queue, axis=-1, keepdims=True) + 1e-8)
    
    # ✅ SIMILARITY: Compute similarities
    # Positive similarities (context vs target)
    pos_similarities = jnp.sum(context_norm * target_norm, axis=-1)  # [batch_size]
    
    # Negative similarities (context vs momentum queue)
    neg_similarities = jnp.dot(context_norm, momentum_norm.T)  # [batch_size, queue_size]
    
    # ✅ CONTRASTIVE: Combine positive and negative similarities
    # Create logits: [batch_size, 1 + queue_size]
    logits = jnp.concatenate([
        pos_similarities[:, None] / temperature,  # Positive logits
        neg_similarities / temperature            # Negative logits
    ], axis=1)
    
    # Labels: positive pairs are at index 0
    labels = jnp.zeros(batch_size, dtype=jnp.int32)
    
    # Cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    
    # ✅ MOMENTUM UPDATE: Update momentum queue
    # Dequeue oldest, enqueue newest
    queue_size = momentum_queue.shape[0]
    
    if batch_size >= queue_size:
        # Replace entire queue
        updated_queue = target_norm[-queue_size:]
    else:
        # Update with momentum
        # Remove oldest batch_size elements, add new ones
        old_queue = momentum_queue[batch_size:]
        new_embeddings = momentum_coefficient * old_queue + (1 - momentum_coefficient) * target_norm
        updated_queue = jnp.concatenate([old_queue, new_embeddings], axis=0)
        
        # Ensure queue size doesn't exceed limit
        if updated_queue.shape[0] > queue_size:
            updated_queue = updated_queue[-queue_size:]
    
    return loss, updated_queue


# Export loss functions
__all__ = [
    "info_nce_loss",
    "enhanced_info_nce_loss",
    "temporal_info_nce_loss", 
    "advanced_info_nce_loss_with_momentum",
    "momentum_enhanced_info_nce_loss"
]

