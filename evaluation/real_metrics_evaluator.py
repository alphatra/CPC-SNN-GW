"""
Real Metrics Evaluator (MODULAR)

This file delegates to modular evaluation components for better maintainability.
The actual implementation has been split into:
- modules/evaluator.py: Main RealMetricsEvaluator class
- modules/roc.py: ROC-AUC and TPR@FAR calculations
- modules/confusion.py: Confusion matrix and binary metrics

This file maintains backward compatibility through delegation.

Real Metrics Evaluator for CPC-SNN-GW Pipeline
"""

import logging
import warnings

# Import from new modular components
from .modules import (
    RealMetricsEvaluator,
    EvaluationMetrics,
    create_evaluator,
    evaluate_model_predictions,
    create_evaluation_report,
    compute_roc_auc,
    compute_tpr_at_far,
    compute_confusion_matrix,
    compute_binary_metrics
)

logger = logging.getLogger(__name__)

# ===== BACKWARD COMPATIBILITY EXPORTS =====
# All classes and functions are now imported from modular components

# Export everything for backward compatibility
__all__ = [
    # Main evaluator (now modular)
    "RealMetricsEvaluator",
    "EvaluationMetrics",
    "create_evaluator",
    "evaluate_model_predictions", 
    "create_evaluation_report",
    
    # Utility functions (now modular)
    "compute_roc_auc",
    "compute_tpr_at_far",
    "compute_confusion_matrix",
    "compute_binary_metrics"
]

# ===== DEPRECATION NOTICE =====
def _show_migration_notice():
    """Show migration notice for direct imports."""
    warnings.warn(
        "Direct imports from real_metrics_evaluator.py are deprecated. "
        "Use modular imports: from evaluation.modules import RealMetricsEvaluator, compute_roc_auc",
        DeprecationWarning,
        stacklevel=3
    )

# Show notice when module is imported directly
logger.info("📦 Using modular evaluation components (real_metrics_evaluator.py → modules/)")
_show_migration_notice()
