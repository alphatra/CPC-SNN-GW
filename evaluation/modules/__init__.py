"""
Evaluation Modules

Modular implementation of evaluation components
split from real_metrics_evaluator.py for better maintainability.

Components:
- roc: ROC-AUC and TPR@FAR calculations
- confusion: Confusion matrix and binary metrics
- evaluator: Main evaluator class and API
"""

from .roc import compute_roc_auc, compute_tpr_at_far
from .confusion import compute_confusion_matrix, compute_binary_metrics
from .evaluator import RealMetricsEvaluator, create_evaluator

__all__ = [
    # ROC utilities
    "compute_roc_auc",
    "compute_tpr_at_far",
    
    # Confusion matrix utilities
    "compute_confusion_matrix",
    "compute_binary_metrics",
    
    # Main evaluator
    "RealMetricsEvaluator",
    "create_evaluator"
]
