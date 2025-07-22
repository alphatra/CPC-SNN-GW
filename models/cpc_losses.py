"""
CPC Loss Functions: Contrastive Learning Objectives

Loss functions for Contrastive Predictive Coding:
- enhanced_info_nce_loss: Advanced InfoNCE with hard negatives and numerical stability
- info_nce_loss: Standard InfoNCE implementation
- Additional contrastive learning utilities
"""

import jax
import jax.numpy as jnp
import optax  # ✅ Added for cross_entropy function
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def enhanced_info_nce_loss(z_context: jnp.ndarray, 
                          z_target: jnp.ndarray, 
                          temperature: float = 0.1,
                          num_negatives: int = 8,
                          use_hard_negatives: bool = False) -> jnp.ndarray:
    """
    Enhanced InfoNCE loss with improved numerical stability and vectorized computation.
    
    Improvements:
    - Better numerical stability with eps guards
    - Cosine similarity computation
    - Optional hard negative mining
    - Vectorized implementation with jax.vmap for efficiency
    - Gradient-friendly implementation
    
    Args:
        z_context: Context representations [batch, time, features]
        z_target: Target representations [batch, time, features]
        temperature: Temperature scaling parameter
        num_negatives: Number of hard negatives (if enabled)
        use_hard_negatives: Whether to use hard negative mining
        
    Returns:
        Scalar loss value
    """
    batch_size, context_len, feature_dim = z_context.shape
    _, target_len, _ = z_target.shape
    
    # Ensure equal lengths for proper alignment
    min_len = min(context_len, target_len)
    z_context = z_context[:, :min_len, :]
    z_target = z_target[:, :min_len, :]
    
    # Normalize for cosine similarity (should already be normalized)
    z_context_norm = z_context / (jnp.linalg.norm(z_context, axis=-1, keepdims=True) + 1e-8)
    z_target_norm = z_target / (jnp.linalg.norm(z_target, axis=-1, keepdims=True) + 1e-8)
    
    # Prepare data for vmap: [time, batch, features]
    z_context_T = jnp.transpose(z_context_norm, (1, 0, 2))
    z_target_T = jnp.transpose(z_target_norm, (1, 0, 2))
    
    def loss_for_single_timestep(context_t, target_t):
        """Compute loss for single timestep - this will be vectorized."""
        # Compute similarity matrix
        similarity_matrix = jnp.dot(context_t, target_t.T)  # [batch, batch]
        
        # Apply temperature scaling
        logits = similarity_matrix / temperature
        
        # Optional hard negative mining
        if use_hard_negatives:
            # Find hardest negatives (highest similarity but wrong pairs)
            mask = jnp.eye(batch_size)  # Positive pairs
            negative_similarities = jnp.where(mask, -jnp.inf, similarity_matrix)
            
            # Keep only top-k hardest negatives
            hard_negatives = jnp.argsort(negative_similarities, axis=1)[:, -num_negatives:]
            
            # Create masked logits with hard negatives
            hard_mask = jnp.zeros_like(logits)
            
            # Vectorized hard negative mask creation
            indices = jnp.arange(batch_size)[:, None]
            hard_mask = hard_mask.at[indices, hard_negatives].set(1.0)
            
            # Keep positive pairs + hard negatives
            pos_mask = jnp.eye(batch_size)
            final_mask = pos_mask + hard_mask
            
            # Apply mask to logits
            logits = jnp.where(final_mask > 0, logits, -jnp.inf)
        
        # Labels for positive pairs (diagonal)
        labels = jnp.arange(batch_size)
        
        # Compute cross-entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        )
        
        return jnp.mean(loss)
    
    # Vectorize over time dimension
    losses = jax.vmap(loss_for_single_timestep)(z_context_T, z_target_T)
    
    # Return mean loss across time
    return jnp.mean(losses)


def info_nce_loss(z_context: jnp.ndarray, z_target: jnp.ndarray, temperature: float = 0.1) -> jnp.ndarray:
    """
    Standard InfoNCE loss implementation.
    
    Simpler version for backward compatibility and baseline comparisons.
    
    Args:
        z_context: Context representations [batch, time, features]
        z_target: Target representations [batch, time, features]
        temperature: Temperature scaling parameter
        
    Returns:
        Scalar loss value
    """
    # Use enhanced version without hard negatives
    return enhanced_info_nce_loss(
        z_context=z_context,
        z_target=z_target, 
        temperature=temperature,
        use_hard_negatives=False
    )


def contrastive_accuracy(z_context: jnp.ndarray, z_target: jnp.ndarray, temperature: float = 0.1) -> jnp.ndarray:
    """
    Compute contrastive accuracy for monitoring training progress.
    
    Args:
        z_context: Context representations [batch, time, features]
        z_target: Target representations [batch, time, features]
        temperature: Temperature scaling parameter
        
    Returns:
        Accuracy score (fraction of correct positive pairs)
    """
    batch_size = z_context.shape[0]
    min_len = min(z_context.shape[1], z_target.shape[1])
    
    # Align sequences
    z_context = z_context[:, :min_len, :]
    z_target = z_target[:, :min_len, :]
    
    # Normalize
    z_context_norm = z_context / (jnp.linalg.norm(z_context, axis=-1, keepdims=True) + 1e-8)
    z_target_norm = z_target / (jnp.linalg.norm(z_target, axis=-1, keepdims=True) + 1e-8)
    
    # Compute similarities and predictions
    similarities = jnp.einsum('btf,bsf->bts', z_context_norm, z_target_norm)
    logits = similarities / temperature
    
    # Get predictions (argmax along target dimension)
    predictions = jnp.argmax(logits, axis=-1)
    
    # True labels are diagonal (same timestep)
    true_labels = jnp.arange(min_len)[None, :]  # [1, time]
    true_labels = jnp.broadcast_to(true_labels, (batch_size, min_len))
    
    # Compute accuracy
    correct = (predictions == true_labels)
    accuracy = jnp.mean(correct)
    
    return accuracy


def cosine_similarity_matrix(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute cosine similarity matrix between two sets of vectors.
    
    Args:
        x: First set of vectors [batch, features]
        y: Second set of vectors [batch, features]
        
    Returns:
        Cosine similarity matrix [batch, batch]
    """
    # Normalize vectors
    x_norm = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    y_norm = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity
    similarity = jnp.dot(x_norm, y_norm.T)
    
    return similarity 