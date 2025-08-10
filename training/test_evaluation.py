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
import numpy as np
try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, roc_curve,
        precision_recall_curve, confusion_matrix, classification_report
    )
    _SKLEARN = True
except Exception:
    _SKLEARN = False

logger = logging.getLogger(__name__)

def evaluate_on_test_set(trainer_state, 
                        test_signals: jnp.ndarray,
                        test_labels: jnp.ndarray,
                        train_signals: jnp.ndarray = None,
                        verbose: bool = True,
                        batch_size: int = 32,
                        optimize_threshold: bool = False,
                        event_ids: Optional[List[str]] = None) -> Dict[str, Any]:
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
    
    # Batched predictions for efficiency
    num_samples = len(test_signals)
    bs = max(1, int(batch_size))
    preds_list = []
    prob_list = []
    for start in range(0, num_samples, bs):
        end = min(start + bs, num_samples)
        batch_x = test_signals[start:end]
        logits = trainer_state.apply_fn(
            trainer_state.params,
            batch_x,
            train=False,
            rngs={'spike_noise': jax.random.PRNGKey(0)}
        )
        preds_list.append(jnp.argmax(logits, axis=-1))
        probs = jax.nn.softmax(logits, axis=-1)
        prob_list.append(probs[:, 1])
    test_predictions = jnp.concatenate(preds_list, axis=0)
    test_prob_class1 = jnp.concatenate(prob_list, axis=0)
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
        # ROC/PR / AUC
        y_true_np = np.array(test_labels)
        y_score_np = np.array(test_prob_class1)
        if _SKLEARN:
            try:
                auc_roc = roc_auc_score(y_true_np, y_score_np)
                auc_pr = average_precision_score(y_true_np, y_score_np)
                fpr, tpr, roc_th = roc_curve(y_true_np, y_score_np)
                prec, rec, pr_th = precision_recall_curve(y_true_np, y_score_np)
            except Exception:
                auc_roc = auc_pr = 0.0
                fpr = tpr = prec = rec = pr_th = roc_th = np.array([])
        else:
            auc_roc = auc_pr = 0.0
            fpr = tpr = prec = rec = pr_th = roc_th = np.array([])
        # Threshold optimization (optional)
        opt_threshold = 0.5
        if _SKLEARN and optimize_threshold and len(roc_th) > 0:
            candidates = np.unique(np.concatenate([roc_th, pr_th]))
            best_f1, best_bacc = -1.0, -1.0
            for th in candidates:
                pred_bin = (y_score_np >= th).astype(int)
                if _SKLEARN:
                    tn, fp, fn, tp = confusion_matrix(y_true_np, pred_bin).ravel()
                else:
                    tp = int(((pred_bin == 1) & (y_true_np == 1)).sum())
                    tn = int(((pred_bin == 0) & (y_true_np == 0)).sum())
                    fp = int(((pred_bin == 1) & (y_true_np == 0)).sum())
                    fn = int(((pred_bin == 0) & (y_true_np == 1)).sum())
                prec_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_c = 2 * prec_c * rec_c / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0.0
                spec_c = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                bacc_c = 0.5 * (rec_c + spec_c)
                if f1_c > best_f1:
                    best_f1 = f1_c
                    opt_threshold = th
                if bacc_c > best_bacc:
                    best_bacc = bacc_c
        # Expected Calibration Error (ECE) with equal-width bins
        try:
            num_bins = 15
            bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
            bin_indices = np.digitize(y_score_np, bin_edges[:-1], right=False) - 1
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)
            ece_accum = 0.0
            total = len(y_true_np)
            for b in range(num_bins):
                mask = (bin_indices == b)
                if np.any(mask):
                    conf = float(np.mean(y_score_np[mask]))
                    acc = float(np.mean((y_true_np[mask] == 1).astype(np.float32)))
                    weight = float(np.mean(mask))
                    ece_accum += weight * abs(acc - conf)
            ece = float(ece_accum)
        except Exception:
            ece = 0.0
    else:
        sensitivity = specificity = precision = recall = f1_score = 0.0
        auc_roc = auc_pr = 0.0
        fpr = tpr = prec = rec = pr_th = roc_th = np.array([])
        opt_threshold = 0.5
        ece = 0.0
    
    # Optional event-level aggregation
    event_metrics: Dict[str, Any] = {}
    try:
        if event_ids is not None and len(event_ids) == len(test_labels):
            event_metrics = _aggregate_event_level_metrics(event_ids, np.array(test_labels), np.array(test_prob_class1))
    except Exception:
        event_metrics = {}

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
        'true_labels': test_labels.tolist(),
        'probabilities': test_prob_class1.tolist() if 'test_prob_class1' in locals() else [],
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': roc_th.tolist()} if len(roc_th) else {},
        'pr_curve': {'precision': prec.tolist(), 'recall': rec.tolist(), 'thresholds': pr_th.tolist()} if len(pr_th) else {},
        'opt_threshold': float(opt_threshold),
        'ece': float(ece),
        'event_level': event_metrics
    }


def _aggregate_event_level_metrics(event_ids: List[str], labels: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    """Aggregate window-level predictions to event-level using mean probability and majority vote."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for eid, lbl, pr in zip(event_ids, labels, probs):
        buckets[eid].append((int(lbl), float(pr)))
    event_true: List[int] = []
    event_pred_mean: List[int] = []
    event_pred_vote: List[int] = []
    for eid, items in buckets.items():
        lbls, prs = zip(*items)
        mean_prob = float(np.mean(prs))
        vote = int(np.round(np.mean([1 if p >= 0.5 else 0 for p in prs])))
        event_true.append(int(np.round(np.mean(lbls))))
        event_pred_mean.append(1 if mean_prob >= 0.5 else 0)
        event_pred_vote.append(vote)
    acc_mean = float(np.mean(np.array(event_pred_mean) == np.array(event_true))) if event_true else 0.0
    acc_vote = float(np.mean(np.array(event_pred_vote) == np.array(event_true))) if event_true else 0.0
    return {
        'num_events': len(buckets),
        'accuracy_meanprob': acc_mean,
        'accuracy_vote': acc_vote
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