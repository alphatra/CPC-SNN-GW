"""
Test Evaluation Module

Migrated from real_ligo_test.py - provides proper test set evaluation
with real accuracy calculation to avoid fake accuracy issues.

Key Features:
- evaluate_on_test_set(): Proper test evaluation with detailed analysis
- detect_model_collapse(): Identifies when model always predicts same class
- Comprehensive logging and validation
"""

import logging
import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

def evaluate_on_test_set(trainer_state, 
                        test_signals: jnp.ndarray,
                        test_labels: jnp.ndarray,
                        train_signals: jnp.ndarray = None,
                        verbose: bool = True) -> Dict[str, Any]:
    """
    Evaluate model on test set with comprehensive analysis
    
    Args:
        trainer_state: Training state with model parameters
        test_signals: Test signal data
        test_labels: Test labels
        train_signals: Training signals (to check for data leakage)
        verbose: Whether to log detailed analysis
        
    Returns:
        Dictionary with test evaluation results
    """
    if verbose:
        logger.info("\n🧪 Evaluating on test set...")
    
    # Check for proper test set
    if len(test_signals) == 0:
        logger.warning("⚠️ Empty test set - cannot evaluate")
        return {
            'test_accuracy': 0.0,
            'has_proper_test_set': False,
            'error': 'Empty test set'
        }
    
    # Check for data leakage (same data used for train and test)
    if train_signals is not None and jnp.array_equal(test_signals, train_signals):
        logger.warning("⚠️ Test set identical to training set - accuracy may be inflated")
        data_leakage = True
    else:
        data_leakage = False
    
    # Get predictions for each test sample
    test_predictions = []
    test_logits_all = []
    
    for i in range(len(test_signals)):
        test_signal = test_signals[i:i+1]
        
        # Forward pass
        test_logits = trainer_state.apply_fn(
            trainer_state.params,
            test_signal,
            train=False
        )
        
        test_pred = jnp.argmax(test_logits, axis=-1)[0]
        test_predictions.append(int(test_pred))
        test_logits_all.append(test_logits[0])
    
    test_predictions = jnp.array(test_predictions)
    test_accuracy = jnp.mean(test_predictions == test_labels)
    
    # Detailed test analysis
    class_counts_true = {
        0: int(jnp.sum(test_labels == 0)),
        1: int(jnp.sum(test_labels == 1))
    }
    
    class_counts_pred = {
        0: int(jnp.sum(test_predictions == 0)),
        1: int(jnp.sum(test_predictions == 1))
    }
    
    if verbose:
        logger.info(f"📊 TEST SET ANALYSIS:")
        logger.info(f"   Test samples: {len(test_predictions)}")
        logger.info(f"   True labels - Class 0: {class_counts_true[0]}, Class 1: {class_counts_true[1]}")
        logger.info(f"   Predictions - Class 0: {class_counts_pred[0]}, Class 1: {class_counts_pred[1]}")
        logger.info(f"   Test accuracy: {test_accuracy:.1%}")
    
    # Check for model collapse (always predicts same class)
    unique_test_preds = jnp.unique(test_predictions)
    model_collapse = len(unique_test_preds) == 1
    
    if model_collapse:
        collapsed_class = int(unique_test_preds[0])
        if verbose:
            logger.warning(f"🚨 MODEL ALWAYS PREDICTS CLASS {collapsed_class} ON TEST SET!")
            logger.warning("   This suggests the model didn't learn properly")
    
    # Show individual predictions if dataset is small
    if verbose and len(test_predictions) <= 20:
        logger.info(f"🔍 TEST PREDICTIONS vs LABELS:")
        for i in range(len(test_predictions)):
            match = "✅" if test_predictions[i] == test_labels[i] else "❌"
            logger.info(f"   Test {i}: Pred={test_predictions[i]}, True={test_labels[i]} {match}")
    elif verbose:
        # Show summary for larger datasets
        correct = jnp.sum(test_predictions == test_labels)
        logger.info(f"🔍 TEST SUMMARY: {correct}/{len(test_predictions)} correct predictions")
    
    # Detect suspicious patterns
    suspicious_patterns = []
    
    if test_accuracy > 0.95:
        suspicious_patterns.append("suspiciously_high_accuracy")
        if verbose:
            logger.warning("🚨 SUSPICIOUSLY HIGH TEST ACCURACY!")
            logger.warning("   Please investigate for data leakage or bugs")
    
    if model_collapse:
        suspicious_patterns.append("model_collapse")
    
    if data_leakage:
        suspicious_patterns.append("data_leakage")
    
    # Calculate additional metrics
    if class_counts_true[1] > 0 and class_counts_true[0] > 0:
        # True positive rate (sensitivity)
        true_positives = int(jnp.sum((test_predictions == 1) & (test_labels == 1)))
        false_negatives = int(jnp.sum((test_predictions == 0) & (test_labels == 1)))
        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        # True negative rate (specificity)
        true_negatives = int(jnp.sum((test_predictions == 0) & (test_labels == 0)))
        false_positives = int(jnp.sum((test_predictions == 1) & (test_labels == 0)))
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
        
        # Precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = sensitivity  # Same as sensitivity
        
        # F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        sensitivity = specificity = precision = recall = f1_score = 0.0
    
    return {
        'test_accuracy': float(test_accuracy),
        'has_proper_test_set': not data_leakage,
        'data_leakage': data_leakage,
        'model_collapse': model_collapse,
        'collapsed_class': int(unique_test_preds[0]) if model_collapse else None,
        'class_counts_true': class_counts_true,
        'class_counts_pred': class_counts_pred,
        'suspicious_patterns': suspicious_patterns,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'predictions': test_predictions.tolist(),
        'true_labels': test_labels.tolist()
    }

def create_test_evaluation_summary(train_accuracy: float,
                                 test_results: Dict[str, Any],
                                 data_source: str = "Unknown",
                                 num_epochs: int = 0) -> str:
    """
    Create a comprehensive test evaluation summary
    
    Args:
        train_accuracy: Training accuracy
        test_results: Results from evaluate_on_test_set
        data_source: Source of training data
        num_epochs: Number of training epochs
        
    Returns:
        Formatted summary string
    """
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("🧪 TEST EVALUATION SUMMARY 🧪")
    summary_lines.append("=" * 70)
    
    # Basic metrics
    summary_lines.append(f"Training Accuracy: {train_accuracy:.1%}")
    
    if test_results['has_proper_test_set']:
        test_acc = test_results['test_accuracy']
        summary_lines.append(f"Test Accuracy: {test_acc:.1%} (This is the REAL accuracy!)")
        
        # Quality assessment
        if test_acc < 0.7:
            summary_lines.append("✅ Realistic accuracy for this challenging task!")
        elif test_acc > 0.95:
            summary_lines.append("🚨 SUSPICIOUSLY HIGH TEST ACCURACY!")
            summary_lines.append("   Please investigate for data leakage or bugs")
        else:
            summary_lines.append("🔬 Good performance - verify results!")
    else:
        summary_lines.append("⚠️ No proper test set - accuracy may be inflated")
    
    # Model behavior analysis
    if test_results.get('model_collapse', False):
        collapsed_class = test_results.get('collapsed_class', 'Unknown')
        summary_lines.append(f"🚨 MODEL COLLAPSE: Always predicts class {collapsed_class}")
    
    if test_results.get('data_leakage', False):
        summary_lines.append("⚠️ DATA LEAKAGE: Test set identical to training set")
    
    # Detailed metrics (if available)
    if test_results.get('f1_score', 0) > 0:
        summary_lines.append(f"F1 Score: {test_results['f1_score']:.3f}")
        summary_lines.append(f"Precision: {test_results['precision']:.3f}")
        summary_lines.append(f"Recall: {test_results['recall']:.3f}")
    
    # Training info
    summary_lines.append(f"Data Source: {data_source}")
    summary_lines.append(f"Epochs: {num_epochs}")
    
    summary_lines.append("=" * 70)
    
    return "\n".join(summary_lines)

def validate_test_set_quality(test_labels: jnp.ndarray, 
                             min_samples_per_class: int = 1) -> Dict[str, Any]:
    """
    Validate test set quality for reliable evaluation
    
    Args:
        test_labels: Test set labels
        min_samples_per_class: Minimum samples required per class
        
    Returns:
        Validation results dictionary
    """
    class_counts = {i: int(jnp.sum(test_labels == i)) for i in jnp.unique(test_labels)}
    
    issues = []
    
    # Check minimum samples per class
    for class_id, count in class_counts.items():
        if count < min_samples_per_class:
            issues.append(f"Class {class_id} has only {count} samples (need ≥{min_samples_per_class})")
    
    # Check for single class
    if len(class_counts) == 1:
        issues.append("Test set contains only one class - cannot evaluate properly")
    
    # Check class balance
    if len(class_counts) == 2:
        counts = list(class_counts.values())
        imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        if imbalance_ratio > 10:
            issues.append(f"Severe class imbalance: {imbalance_ratio:.1f}:1 ratio")
    
    is_valid = len(issues) == 0
    
    return {
        'is_valid': is_valid,
        'class_counts': class_counts,
        'issues': issues,
        'total_samples': len(test_labels)
    } 