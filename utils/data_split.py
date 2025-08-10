"""
Data Split Utilities

Migrated from real_ligo_test.py - provides stratified train/test split
functionality for the main CLI and pipeline.

Key Features:
- create_stratified_split(): Ensures balanced train/test sets
- Handles edge cases (single class, small datasets)
- Comprehensive logging and validation
"""

import logging
import jax
import jax.numpy as jnp
from typing import Tuple

logger = logging.getLogger(__name__)

def create_stratified_split(signals: jnp.ndarray, 
                           labels: jnp.ndarray,
                           train_ratio: float = 0.8,
                           random_seed: int = 42) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], 
                                                          Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Create stratified train/test split ensuring balanced representation of classes
    
    Args:
        signals: Input signal data
        labels: Corresponding labels
        train_ratio: Fraction of data for training (0.8 = 80%)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of ((train_signals, train_labels), (test_signals, test_labels))
    """
    if len(signals) <= 1:
        logger.warning("⚠️ Dataset too small for proper split - using same data for train/test")
        return (signals, labels), (signals, labels)
    
    # ✅ STRATIFIED SPLIT: Ensure both classes in train and test
    class_0_indices = jnp.where(labels == 0)[0]
    class_1_indices = jnp.where(labels == 1)[0]
    
    # Calculate split for each class
    n_class_0 = len(class_0_indices)
    n_class_1 = len(class_1_indices)
    
    logger.info(f"📊 Creating stratified split (train: {train_ratio:.1%}):")
    logger.info(f"   Class 0 samples: {n_class_0}")
    logger.info(f"   Class 1 samples: {n_class_1}")
    
    # ✅ FALLBACK: If one class is missing, use random split
    if n_class_0 == 0 or n_class_1 == 0:
        logger.warning(f"⚠️ Only one class present (0: {n_class_0}, 1: {n_class_1}) - using random split")
        n_train = max(1, int(train_ratio * len(signals)))
        indices = jax.random.permutation(jax.random.PRNGKey(random_seed), len(signals))
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
    else:
        # Calculate samples per class for training
        n_train_0 = max(1, int(train_ratio * n_class_0))
        n_train_1 = max(1, int(train_ratio * n_class_1))
    
        # Shuffle each class separately
        shuffled_0 = jax.random.permutation(jax.random.PRNGKey(random_seed), class_0_indices)
        shuffled_1 = jax.random.permutation(jax.random.PRNGKey(random_seed + 1), class_1_indices)
        
        # Split each class
        train_indices_0 = shuffled_0[:n_train_0]
        test_indices_0 = shuffled_0[n_train_0:]
        train_indices_1 = shuffled_1[:n_train_1] 
        test_indices_1 = shuffled_1[n_train_1:]
        
        # Combine indices
        train_indices = jnp.concatenate([train_indices_0, train_indices_1])
        test_indices = jnp.concatenate([test_indices_0, test_indices_1])
        
        # Final shuffle to mix classes
        train_indices = jax.random.permutation(jax.random.PRNGKey(random_seed + 2), train_indices)
        test_indices = jax.random.permutation(jax.random.PRNGKey(random_seed + 3), test_indices)
    
    # Extract splits
    train_signals = signals[train_indices]
    train_labels = labels[train_indices]
    test_signals = signals[test_indices] 
    test_labels = labels[test_indices]
    
    # Log split results
    logger.info(f"📊 Dataset split completed:")
    logger.info(f"   Train: {len(train_signals)} samples")
    logger.info(f"   Test: {len(test_signals)} samples")
    logger.info(f"   Train class balance: {jnp.mean(train_labels):.1%} positive")
    logger.info(f"   Test class balance: {jnp.mean(test_labels):.1%} positive")
    
    # ✅ CRITICAL: Validate test set quality
    if len(test_signals) > 0:
        if jnp.all(test_labels == 0):
            logger.error("🚨 ALL TEST LABELS ARE 0 - This will give fake accuracy!")
            logger.error("   Stratified split failed - dataset too small or imbalanced")
        elif jnp.all(test_labels == 1):
            logger.error("🚨 ALL TEST LABELS ARE 1 - This will give fake accuracy!")
            logger.error("   Stratified split failed - dataset too small or imbalanced")
        else:
            logger.info(f"✅ Balanced test set: {jnp.mean(test_labels):.1%} positive")
    
    return (train_signals, train_labels), (test_signals, test_labels)

def validate_split_quality(train_labels: jnp.ndarray, 
                          test_labels: jnp.ndarray,
                          min_samples_per_class: int = 1) -> bool:
    """
    Validate the quality of train/test split
    
    Args:
        train_labels: Training labels
        test_labels: Test labels  
        min_samples_per_class: Minimum samples per class required
        
    Returns:
        True if split is valid, False otherwise
    """
    # Check training set
    train_class_0 = jnp.sum(train_labels == 0)
    train_class_1 = jnp.sum(train_labels == 1)
    
    # Check test set
    test_class_0 = jnp.sum(test_labels == 0)
    test_class_1 = jnp.sum(test_labels == 1)
    
    logger.info(f"🔍 Split validation:")
    logger.info(f"   Train - Class 0: {train_class_0}, Class 1: {train_class_1}")
    logger.info(f"   Test - Class 0: {test_class_0}, Class 1: {test_class_1}")
    
    # Validate minimum samples per class
    if (train_class_0 < min_samples_per_class or train_class_1 < min_samples_per_class or
        test_class_0 < min_samples_per_class or test_class_1 < min_samples_per_class):
        logger.error(f"❌ Split validation failed: Need at least {min_samples_per_class} samples per class")
        return False
    
    # Check for single-class test set (leads to fake accuracy)
    if test_class_0 == 0 or test_class_1 == 0:
        logger.error("❌ Split validation failed: Test set has only one class")
        return False
    
    logger.info("✅ Split validation passed")
    return True 