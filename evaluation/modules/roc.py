"""
ROC-AUC and TPR@FAR calculation utilities.

This module contains ROC curve analysis functionality extracted from
real_metrics_evaluator.py for better modularity.

Split from real_metrics_evaluator.py for better maintainability.
"""

import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available - ROC calculations will use fallback")


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
    """
    Compute ROC-AUC with comprehensive statistics.
    
    Args:
        y_true: True binary labels
        y_scores: Prediction scores/probabilities
        
    Returns:
        Dictionary with ROC-AUC results
    """
    if not SKLEARN_AVAILABLE:
        return _compute_roc_auc_fallback(y_true, y_scores)
    
    try:
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's index)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        
        results = {
            'roc_auc': float(roc_auc),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(), 
            'thresholds': thresholds.tolist(),
            'optimal_threshold': float(optimal_threshold),
            'optimal_tpr': float(optimal_tpr),
            'optimal_fpr': float(optimal_fpr),
            'youden_index': float(optimal_tpr - optimal_fpr)
        }
        
        logger.debug(f"ROC-AUC computed: {roc_auc:.4f}, optimal_threshold: {optimal_threshold:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"ROC-AUC computation failed: {e}")
        return {'error': str(e)}


def compute_tpr_at_far(y_true: np.ndarray, y_scores: np.ndarray, far_threshold: float = 1e-3) -> Dict[str, Any]:
    """
    Compute True Positive Rate at specified False Alarm Rate.
    
    Args:
        y_true: True binary labels
        y_scores: Prediction scores/probabilities  
        far_threshold: Target false alarm rate
        
    Returns:
        Dictionary with TPR@FAR results
    """
    if not SKLEARN_AVAILABLE:
        return _compute_tpr_at_far_fallback(y_true, y_scores, far_threshold)
    
    try:
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Find closest FPR to target FAR
        far_idx = np.argmin(np.abs(fpr - far_threshold))
        achieved_far = fpr[far_idx]
        achieved_tpr = tpr[far_idx]
        threshold_at_far = thresholds[far_idx]
        
        # Interpolate for exact FAR if possible
        if far_idx > 0 and far_idx < len(fpr) - 1:
            # Linear interpolation between adjacent points
            fpr_low, fpr_high = fpr[far_idx], fpr[far_idx + 1]
            tpr_low, tpr_high = tpr[far_idx], tpr[far_idx + 1]
            
            if fpr_high != fpr_low:
                alpha = (far_threshold - fpr_low) / (fpr_high - fpr_low)
                interpolated_tpr = tpr_low + alpha * (tpr_high - tpr_low)
            else:
                interpolated_tpr = achieved_tpr
        else:
            interpolated_tpr = achieved_tpr
        
        results = {
            'tpr_at_far': float(interpolated_tpr),
            'target_far': far_threshold,
            'achieved_far': float(achieved_far),
            'threshold_at_far': float(threshold_at_far),
            'interpolated': far_idx > 0 and far_idx < len(fpr) - 1
        }
        
        logger.debug(f"TPR@FAR computed: TPR={interpolated_tpr:.4f} at FAR={far_threshold:.2e}")
        return results
        
    except Exception as e:
        logger.error(f"TPR@FAR computation failed: {e}")
        return {'error': str(e)}


def _compute_roc_auc_fallback(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
    """Fallback ROC-AUC computation without sklearn."""
    try:
        # Simple manual ROC curve computation
        thresholds = np.unique(y_scores)
        thresholds = np.sort(thresholds)[::-1]  # Descending order
        
        fprs = []
        tprs = []
        
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        if n_pos == 0 or n_neg == 0:
            return {'error': 'Single class in y_true'}
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            
            tpr = tp / n_pos
            fpr = fp / n_neg
            
            tprs.append(tpr)
            fprs.append(fpr)
        
        # Compute AUC using trapezoidal rule
        fprs = np.array(fprs)
        tprs = np.array(tprs)
        
        # Sort by FPR
        sort_idx = np.argsort(fprs)
        fprs = fprs[sort_idx]
        tprs = tprs[sort_idx]
        
        auc_value = np.trapz(tprs, fprs)
        
        return {
            'roc_auc': float(auc_value),
            'fpr': fprs.tolist(),
            'tpr': tprs.tolist(),
            'thresholds': thresholds.tolist(),
            'fallback_computation': True
        }
        
    except Exception as e:
        logger.error(f"Fallback ROC-AUC computation failed: {e}")
        return {'error': str(e)}


def _compute_tpr_at_far_fallback(y_true: np.ndarray, y_scores: np.ndarray, far_threshold: float) -> Dict[str, Any]:
    """Fallback TPR@FAR computation without sklearn."""
    try:
        # Manual computation
        roc_results = _compute_roc_auc_fallback(y_true, y_scores)
        
        if 'error' in roc_results:
            return roc_results
        
        fprs = np.array(roc_results['fpr'])
        tprs = np.array(roc_results['tpr'])
        
        # Find closest FPR to target
        far_idx = np.argmin(np.abs(fprs - far_threshold))
        achieved_tpr = tprs[far_idx]
        achieved_far = fprs[far_idx]
        
        return {
            'tpr_at_far': float(achieved_tpr),
            'target_far': far_threshold,
            'achieved_far': float(achieved_far),
            'fallback_computation': True
        }
        
    except Exception as e:
        logger.error(f"Fallback TPR@FAR computation failed: {e}")
        return {'error': str(e)}


# Export ROC utilities
__all__ = [
    "compute_roc_auc",
    "compute_tpr_at_far"
]
