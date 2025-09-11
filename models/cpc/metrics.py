"""
Contrastive learning evaluation metrics for CPC.

This module contains metric functions extracted from
cpc_losses.py for better modularity:
- contrastive_accuracy: Accuracy in contrastive learning setting
- cosine_similarity_matrix: Cosine similarity computation utilities

Split from cpc_losses.py for better maintainability.
"""

import logging
from typing import Optional

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def contrastive_accuracy(z_context: jnp.ndarray, z_target: jnp.ndarray, temperature: float = 0.1) -> jnp.ndarray:
    """
    Compute contrastive accuracy for CPC evaluation.
    
    Measures how often the model correctly identifies positive pairs
    in the contrastive learning setting.
    
    Args:
        z_context: Context embeddings [batch_size, feature_dim]
        z_target: Target embeddings [batch_size, feature_dim]
        temperature: Temperature parameter for scaling
        
    Returns:
        Contrastive accuracy (fraction of correct positive pair predictions)
    """
    batch_size = z_context.shape[0]
    
    if batch_size < 2:
        logger.warning("Need at least 2 samples for contrastive accuracy")
        return jnp.array(1.0)  # Trivial case
    
    # ✅ NORMALIZATION: L2 normalize embeddings
    z_context_norm = z_context / (jnp.linalg.norm(z_context, axis=-1, keepdims=True) + 1e-8)
    z_target_norm = z_target / (jnp.linalg.norm(z_target, axis=-1, keepdims=True) + 1e-8)
    
    # ✅ SIMILARITY: Compute similarity matrix
    similarity_matrix = jnp.dot(z_context_norm, z_target_norm.T) / temperature
    
    # ✅ PREDICTION: Get predicted positive pairs
    predicted_pairs = jnp.argmax(similarity_matrix, axis=1)
    
    # ✅ ACCURACY: True positive pairs are diagonal (i -> i)
    true_pairs = jnp.arange(batch_size)
    correct_predictions = (predicted_pairs == true_pairs)
    
    accuracy = jnp.mean(correct_predictions.astype(jnp.float32))
    
    return accuracy


def cosine_similarity_matrix(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute cosine similarity matrix between two sets of vectors.
    
    Args:
        x: First set of vectors [batch_size_x, feature_dim]
        y: Second set of vectors [batch_size_y, feature_dim]
        
    Returns:
        Cosine similarity matrix [batch_size_x, batch_size_y]
    """
    # ✅ NORMALIZATION: L2 normalize both sets
    x_norm = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    y_norm = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + 1e-8)
    
    # ✅ SIMILARITY: Compute cosine similarity
    similarity_matrix = jnp.dot(x_norm, y_norm.T)
    
    # ✅ CLIPPING: Ensure values are in valid range [-1, 1]
    similarity_matrix = jnp.clip(similarity_matrix, -1.0, 1.0)
    
    return similarity_matrix


def compute_contrastive_metrics(z_context: jnp.ndarray, 
                               z_target: jnp.ndarray,
                               temperature: float = 0.1) -> dict:
    """
    Compute comprehensive contrastive learning metrics.
    
    Args:
        z_context: Context embeddings
        z_target: Target embeddings  
        temperature: Temperature parameter
        
    Returns:
        Dictionary of contrastive metrics
    """
    metrics = {}
    
    try:
        # Basic metrics
        metrics['contrastive_accuracy'] = float(contrastive_accuracy(z_context, z_target, temperature))
        metrics['info_nce_loss'] = float(enhanced_info_nce_loss(z_context, z_target, temperature))
        
        # Similarity analysis
        similarity_matrix = cosine_similarity_matrix(z_context, z_target)
        
        # Diagonal (positive pairs) vs off-diagonal (negative pairs)
        batch_size = min(z_context.shape[0], z_target.shape[0])
        if batch_size > 1:
            diagonal_sim = jnp.diag(similarity_matrix[:batch_size, :batch_size])
            off_diagonal_mask = ~jnp.eye(batch_size, dtype=bool)
            off_diagonal_sim = similarity_matrix[:batch_size, :batch_size][off_diagonal_mask]
            
            metrics['positive_similarity_mean'] = float(jnp.mean(diagonal_sim))
            metrics['negative_similarity_mean'] = float(jnp.mean(off_diagonal_sim))
            metrics['similarity_separation'] = float(jnp.mean(diagonal_sim) - jnp.mean(off_diagonal_sim))
        
        # Feature statistics
        metrics['context_norm_mean'] = float(jnp.mean(jnp.linalg.norm(z_context, axis=-1)))
        metrics['target_norm_mean'] = float(jnp.mean(jnp.linalg.norm(z_target, axis=-1)))
        
        # Representation quality
        context_var = jnp.var(z_context, axis=0)
        target_var = jnp.var(z_target, axis=0)
        metrics['context_representation_variance'] = float(jnp.mean(context_var))
        metrics['target_representation_variance'] = float(jnp.mean(target_var))
        
    except Exception as e:
        logger.error(f"Error computing contrastive metrics: {e}")
        metrics['error'] = str(e)
    
    return metrics


def evaluate_representation_quality(embeddings: jnp.ndarray) -> dict:
    """
    Evaluate quality of learned representations.
    
    Args:
        embeddings: Learned embeddings [batch_size, feature_dim]
        
    Returns:
        Dictionary of representation quality metrics
    """
    metrics = {}
    
    try:
        # Basic statistics
        metrics['mean_norm'] = float(jnp.mean(jnp.linalg.norm(embeddings, axis=-1)))
        metrics['std_norm'] = float(jnp.std(jnp.linalg.norm(embeddings, axis=-1)))
        
        # Feature diversity
        feature_vars = jnp.var(embeddings, axis=0)
        metrics['feature_variance_mean'] = float(jnp.mean(feature_vars))
        metrics['feature_variance_std'] = float(jnp.std(feature_vars))
        metrics['dead_features'] = int(jnp.sum(feature_vars < 1e-6))
        
        # Representation spread
        pairwise_distances = jnp.linalg.norm(
            embeddings[:, None, :] - embeddings[None, :, :], axis=-1
        )
        metrics['mean_pairwise_distance'] = float(jnp.mean(pairwise_distances))
        metrics['min_pairwise_distance'] = float(jnp.min(pairwise_distances[pairwise_distances > 0]))
        
        # Dimensional utilization
        embedding_range = jnp.max(embeddings, axis=0) - jnp.min(embeddings, axis=0)
        metrics['dimensional_utilization'] = float(jnp.mean(embedding_range))
        
    except Exception as e:
        logger.error(f"Error evaluating representation quality: {e}")
        metrics['error'] = str(e)
    
    return metrics


# Export metric functions
__all__ = [
    "contrastive_accuracy",
    "cosine_similarity_matrix",
    "compute_contrastive_metrics",
    "evaluate_representation_quality"
]

