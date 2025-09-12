"""
Confusion matrix and binary classification metrics.

This module contains confusion matrix functionality extracted from
real_metrics_evaluator.py for better modularity.

Split from real_metrics_evaluator.py for better maintainability.
"""

import logging
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from sklearn.metrics import confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available - confusion matrix will use fallback")


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Compute confusion matrix with detailed statistics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with confusion matrix and derived metrics
    """
    if not SKLEARN_AVAILABLE:
        return _compute_confusion_matrix_fallback(y_true, y_pred)
    
    try:
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get unique classes
        classes = np.unique(np.concatenate([y_true, y_pred]))
        
        results = {
            'confusion_matrix': cm.tolist(),
            'classes': classes.tolist(),
            'shape': cm.shape
        }
        
        # Add class-wise metrics for binary classification
        if len(classes) == 2:
            tn, fp, fn, tp = cm.ravel()
            
            results.update({
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'total_samples': int(tn + fp + fn + tp)
            })
            
            # Derived metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            results.update({
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'specificity': float(specificity),
                'f1_score': float(f1_score),
                'balanced_accuracy': float((recall + specificity) / 2)
            })
        
        logger.debug(f"Confusion matrix computed: shape={cm.shape}")
        return results
        
    except Exception as e:
        logger.error(f"Confusion matrix computation failed: {e}")
        return {'error': str(e)}


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Compute comprehensive binary classification metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        
    Returns:
        Dictionary with binary classification metrics
    """
    try:
        # Get confusion matrix results
        cm_results = compute_confusion_matrix(y_true, y_pred)
        
        if 'error' in cm_results:
            return cm_results
        
        # Extract binary metrics (already computed in confusion matrix function)
        if 'accuracy' in cm_results:  # Binary classification
            binary_metrics = {
                'accuracy': cm_results['accuracy'],
                'precision': cm_results['precision'],
                'recall': cm_results['recall'],
                'specificity': cm_results['specificity'],
                'f1_score': cm_results['f1_score'],
                'balanced_accuracy': cm_results['balanced_accuracy'],
                'sensitivity': cm_results['recall'],  # Alias
                'true_positive_rate': cm_results['recall'],  # Alias
                'false_positive_rate': 1.0 - cm_results['specificity']
            }
            
            # Additional binary metrics
            tn = cm_results['true_negatives']
            fp = cm_results['false_positives'] 
            fn = cm_results['false_negatives']
            tp = cm_results['true_positives']
            
            # Positive/Negative Predictive Values
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Same as precision
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            
            # Likelihood ratios
            lr_positive = binary_metrics['sensitivity'] / binary_metrics['false_positive_rate'] if binary_metrics['false_positive_rate'] > 0 else float('inf')
            lr_negative = (1 - binary_metrics['sensitivity']) / binary_metrics['specificity'] if binary_metrics['specificity'] > 0 else float('inf')
            
            binary_metrics.update({
                'positive_predictive_value': float(ppv),
                'negative_predictive_value': float(npv),
                'likelihood_ratio_positive': float(lr_positive) if lr_positive != float('inf') else 999.0,
                'likelihood_ratio_negative': float(lr_negative) if lr_negative != float('inf') else 999.0
            })
            
            return binary_metrics
        else:
            return {'error': 'Not binary classification or computation failed'}
            
    except Exception as e:
        logger.error(f"Binary metrics computation failed: {e}")
        return {'error': str(e)}


def _compute_confusion_matrix_fallback(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Fallback confusion matrix computation without sklearn."""
    try:
        # Get unique classes
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        
        # Initialize confusion matrix
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        # Fill confusion matrix
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
        
        results = {
            'confusion_matrix': cm.tolist(),
            'classes': classes.tolist(),
            'shape': cm.shape,
            'fallback_computation': True
        }
        
        # Binary classification metrics
        if n_classes == 2:
            tn, fp, fn, tp = cm.ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            results.update({
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'specificity': float(specificity),
                'f1_score': float(f1_score),
                'balanced_accuracy': float((recall + specificity) / 2)
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Fallback confusion matrix computation failed: {e}")
        return {'error': str(e)}


def create_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                               class_names: Optional[list] = None) -> Dict[str, Any]:
    """
    Create comprehensive classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names for report
        
    Returns:
        Classification report dictionary
    """
    if not SKLEARN_AVAILABLE:
        return _create_classification_report_fallback(y_true, y_pred, class_names)
    
    try:
        # Use sklearn classification report
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        return {
            'classification_report': report,
            'sklearn_available': True
        }
        
    except Exception as e:
        logger.error(f"Classification report failed: {e}")
        return {'error': str(e)}


def _create_classification_report_fallback(y_true: np.ndarray, y_pred: np.ndarray, 
                                         class_names: Optional[list] = None) -> Dict[str, Any]:
    """Fallback classification report without sklearn."""
    try:
        classes = np.unique(np.concatenate([y_true, y_pred]))
        
        if class_names is None:
            class_names = [f'class_{c}' for c in classes]
        
        report = {}
        
        for i, cls in enumerate(classes):
            cls_name = class_names[i] if i < len(class_names) else f'class_{cls}'
            
            # Class-specific metrics
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = np.sum(y_true == cls)
            
            report[cls_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1-score': float(f1),
                'support': int(support)
            }
        
        # Overall metrics
        accuracy = np.mean(y_true == y_pred)
        
        report['accuracy'] = float(accuracy)
        report['macro avg'] = {
            'precision': float(np.mean([report[name]['precision'] for name in class_names])),
            'recall': float(np.mean([report[name]['recall'] for name in class_names])),
            'f1-score': float(np.mean([report[name]['f1-score'] for name in class_names])),
            'support': len(y_true)
        }
        
        return {
            'classification_report': report,
            'fallback_computation': True
        }
        
    except Exception as e:
        logger.error(f"Fallback classification report failed: {e}")
        return {'error': str(e)}


# Export confusion matrix utilities
__all__ = [
    "compute_confusion_matrix",
    "compute_binary_metrics", 
    "create_classification_report"
]
