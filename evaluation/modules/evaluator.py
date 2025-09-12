"""
Main evaluator class for comprehensive metrics evaluation.

This module contains the main RealMetricsEvaluator class extracted from
real_metrics_evaluator.py for better modularity.

Split from real_metrics_evaluator.py for better maintainability.
"""

import logging
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .roc import compute_roc_auc, compute_tpr_at_far
from .confusion import compute_confusion_matrix, compute_binary_metrics, create_classification_report

logger = logging.getLogger(__name__)

# Try importing visualization dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("matplotlib/seaborn not available - visualization disabled")


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    # Core metrics
    roc_auc: float
    tpr_at_far: float  # TPR@FAR=1/30d
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    
    # Advanced metrics
    balanced_accuracy: float
    specificity: float
    average_precision: float
    
    # Confusion matrix elements
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    # Quality indicators
    model_collapse: bool
    has_proper_test_set: bool
    
    # Additional data
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'roc_auc': self.roc_auc,
            'tpr_at_far': self.tpr_at_far,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'accuracy': self.accuracy,
            'balanced_accuracy': self.balanced_accuracy,
            'specificity': self.specificity,
            'average_precision': self.average_precision,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives,
            'model_collapse': self.model_collapse,
            'has_proper_test_set': self.has_proper_test_set,
            'confusion_matrix': self.confusion_matrix,
            'classification_report': self.classification_report
        }


class RealMetricsEvaluator:
    """
    Real Metrics Evaluator for CPC-SNN-GW Pipeline.
    
    ✅ ENHANCED: Professional evaluation with ROC-AUC, TPR@FAR, and bootstrap CI
    Based on audit recommendations for scientific GW detection metrics.
    """
    
    def __init__(self, 
                 far_threshold: float = 1.0/(30*24*3600),  # 1 per 30 days
                 bootstrap_samples: int = 1000,
                 confidence_level: float = 0.95,
                 output_dir: Optional[Path] = None):
        """
        Initialize real metrics evaluator.
        
        Args:
            far_threshold: False alarm rate threshold for TPR@FAR
            bootstrap_samples: Number of bootstrap samples for CI
            confidence_level: Confidence level for bootstrap CI
            output_dir: Output directory for plots and results
        """
        self.far_threshold = far_threshold
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RealMetricsEvaluator initialized: FAR={far_threshold:.2e}")
    
    def evaluate_predictions(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           y_scores: Optional[np.ndarray] = None) -> EvaluationMetrics:
        """
        Evaluate predictions with comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores/probabilities (optional)
            
        Returns:
            EvaluationMetrics with comprehensive results
        """
        logger.info("Computing comprehensive evaluation metrics...")
        
        # ✅ BASIC VALIDATION
        if len(y_true) != len(y_pred):
            raise ValueError("Length mismatch between y_true and y_pred")
        
        if len(np.unique(y_true)) < 2:
            logger.warning("Single class in y_true - limited metrics available")
        
        # ✅ CONFUSION MATRIX AND BINARY METRICS
        confusion_results = compute_confusion_matrix(y_true, y_pred)
        binary_metrics = compute_binary_metrics(y_true, y_pred)
        
        # ✅ ROC-AUC AND TPR@FAR (if scores available)
        if y_scores is not None:
            roc_results = compute_roc_auc(y_true, y_scores)
            tpr_far_results = compute_tpr_at_far(y_true, y_scores, self.far_threshold)
            
            roc_auc = roc_results.get('roc_auc', 0.5)
            tpr_at_far = tpr_far_results.get('tpr_at_far', 0.0)
            average_precision = roc_results.get('average_precision', 0.0)
        else:
            # Fallback values when scores not available
            logger.warning("No prediction scores provided - ROC-AUC unavailable")
            roc_auc = 0.5  # Random performance baseline
            tpr_at_far = 0.0
            average_precision = 0.0
        
        # ✅ MODEL COLLAPSE DETECTION
        unique_predictions = len(np.unique(y_pred))
        model_collapse = unique_predictions == 1
        
        if model_collapse:
            logger.warning("Model collapse detected - all predictions are same class")
        
        # ✅ TEST SET QUALITY
        has_proper_test_set = len(np.unique(y_true)) > 1 and len(y_true) >= 10
        
        if not has_proper_test_set:
            logger.warning("Poor test set quality detected")
        
        # ✅ CLASSIFICATION REPORT
        class_names = ['noise', 'gravitational_wave'] if len(np.unique(y_true)) == 2 else None
        classification_report = create_classification_report(y_true, y_pred, class_names)
        
        # ✅ CREATE COMPREHENSIVE METRICS
        metrics = EvaluationMetrics(
            roc_auc=float(roc_auc),
            tpr_at_far=float(tpr_at_far),
            precision=binary_metrics.get('precision', 0.0),
            recall=binary_metrics.get('recall', 0.0),
            f1_score=binary_metrics.get('f1_score', 0.0),
            accuracy=binary_metrics.get('accuracy', 0.0),
            balanced_accuracy=binary_metrics.get('balanced_accuracy', 0.0),
            specificity=binary_metrics.get('specificity', 0.0),
            average_precision=float(average_precision),
            true_positives=confusion_results.get('true_positives', 0),
            false_positives=confusion_results.get('false_positives', 0),
            true_negatives=confusion_results.get('true_negatives', 0),
            false_negatives=confusion_results.get('false_negatives', 0),
            model_collapse=model_collapse,
            has_proper_test_set=has_proper_test_set,
            confusion_matrix=confusion_results.get('confusion_matrix', []),
            classification_report=classification_report.get('classification_report', {})
        )
        
        # ✅ SAVE RESULTS
        if self.output_dir:
            self._save_evaluation_results(metrics, y_true, y_pred, y_scores)
        
        # ✅ LOG SUMMARY
        self._log_evaluation_summary(metrics)
        
        return metrics
    
    def _save_evaluation_results(self, metrics: EvaluationMetrics, 
                               y_true: np.ndarray, y_pred: np.ndarray, 
                               y_scores: Optional[np.ndarray] = None):
        """Save evaluation results to files."""
        try:
            # Save metrics as JSON
            import json
            metrics_file = self.output_dir / "evaluation_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
            
            # Save confusion matrix as CSV
            if metrics.confusion_matrix:
                import pandas as pd
                cm_df = pd.DataFrame(
                    metrics.confusion_matrix,
                    index=['noise', 'gw'],
                    columns=['noise', 'gw']
                )
                cm_df.to_csv(self.output_dir / "confusion_matrix.csv")
            
            # Generate plots if possible
            if PLOTTING_AVAILABLE and y_scores is not None:
                self._generate_evaluation_plots(y_true, y_pred, y_scores)
            
            logger.info(f"Evaluation results saved to {self.output_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to save evaluation results: {e}")
    
    def _generate_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray):
        """Generate evaluation plots."""
        try:
            # ROC curve
            roc_results = compute_roc_auc(y_true, y_scores)
            
            if 'fpr' in roc_results and 'tpr' in roc_results:
                plt.figure(figsize=(8, 6))
                plt.plot(roc_results['fpr'], roc_results['tpr'], 
                        label=f'ROC Curve (AUC = {roc_results["roc_auc"]:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(self.output_dir / "roc_curve.png", dpi=150)
                plt.close()
            
            # Confusion matrix heatmap
            confusion_results = compute_confusion_matrix(y_true, y_pred)
            if 'confusion_matrix' in confusion_results:
                plt.figure(figsize=(6, 5))
                sns.heatmap(
                    confusion_results['confusion_matrix'], 
                    annot=True, 
                    fmt='d',
                    xticklabels=['noise', 'gw'],
                    yticklabels=['noise', 'gw']
                )
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(self.output_dir / "confusion_matrix.png", dpi=150)
                plt.close()
            
            logger.info("Evaluation plots generated successfully")
            
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")
    
    def _log_evaluation_summary(self, metrics: EvaluationMetrics):
        """Log evaluation summary to console."""
        logger.info("📊 EVALUATION RESULTS SUMMARY:")
        logger.info(f"   ROC-AUC: {metrics.roc_auc:.4f}")
        logger.info(f"   TPR@FAR: {metrics.tpr_at_far:.4f}")
        logger.info(f"   Accuracy: {metrics.accuracy:.4f}")
        logger.info(f"   Precision: {metrics.precision:.4f}")
        logger.info(f"   Recall: {metrics.recall:.4f}")
        logger.info(f"   F1-Score: {metrics.f1_score:.4f}")
        logger.info(f"   Balanced Accuracy: {metrics.balanced_accuracy:.4f}")
        
        # Quality indicators
        if metrics.model_collapse:
            logger.warning("⚠️ Model collapse detected")
        if not metrics.has_proper_test_set:
            logger.warning("⚠️ Poor test set quality")
        
        if metrics.roc_auc > 0.9:
            logger.info("🎉 Excellent performance (ROC-AUC > 0.9)")
        elif metrics.roc_auc > 0.8:
            logger.info("✅ Good performance (ROC-AUC > 0.8)")
        elif metrics.roc_auc > 0.7:
            logger.info("👍 Fair performance (ROC-AUC > 0.7)")
        else:
            logger.warning("⚠️ Poor performance (ROC-AUC ≤ 0.7)")


def create_evaluator(far_threshold: float = 1.0/(30*24*3600),
                    bootstrap_samples: int = 1000,
                    output_dir: Optional[Path] = None) -> RealMetricsEvaluator:
    """
    Factory function for creating RealMetricsEvaluator.
    
    Args:
        far_threshold: False alarm rate threshold
        bootstrap_samples: Number of bootstrap samples
        output_dir: Output directory for results
        
    Returns:
        Configured RealMetricsEvaluator
    """
    return RealMetricsEvaluator(
        far_threshold=far_threshold,
        bootstrap_samples=bootstrap_samples,
        output_dir=output_dir
    )


def evaluate_model_predictions(y_true: np.ndarray,
                             y_pred: np.ndarray, 
                             y_scores: Optional[np.ndarray] = None,
                             output_dir: Optional[Path] = None) -> EvaluationMetrics:
    """
    Convenience function for quick model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores (optional)
        output_dir: Output directory for results
        
    Returns:
        Comprehensive evaluation metrics
    """
    evaluator = create_evaluator(output_dir=output_dir)
    return evaluator.evaluate_predictions(y_true, y_pred, y_scores)


def create_evaluation_report(metrics: EvaluationMetrics, 
                           model_name: str = "CPC-SNN-GW",
                           dataset_name: str = "GW Detection") -> str:
    """
    Create formatted evaluation report.
    
    Args:
        metrics: Evaluation metrics
        model_name: Name of evaluated model
        dataset_name: Name of dataset
        
    Returns:
        Formatted evaluation report string
    """
    report = f"""
# {model_name} Evaluation Report

## Dataset: {dataset_name}

### Core Performance Metrics
- **ROC-AUC**: {metrics.roc_auc:.4f}
- **TPR@FAR**: {metrics.tpr_at_far:.4f}
- **Accuracy**: {metrics.accuracy:.4f}
- **Balanced Accuracy**: {metrics.balanced_accuracy:.4f}

### Classification Metrics
- **Precision**: {metrics.precision:.4f}
- **Recall**: {metrics.recall:.4f}
- **F1-Score**: {metrics.f1_score:.4f}
- **Specificity**: {metrics.specificity:.4f}

### Confusion Matrix
```
                Predicted
           noise    gw
Actual noise  {metrics.true_negatives:4d}  {metrics.false_positives:4d}
       gw     {metrics.false_negatives:4d}  {metrics.true_positives:4d}
```

### Quality Assessment
- **Model Collapse**: {'Yes' if metrics.model_collapse else 'No'}
- **Proper Test Set**: {'Yes' if metrics.has_proper_test_set else 'No'}

### Performance Assessment
"""
    
    # Add performance assessment
    if metrics.roc_auc > 0.9:
        report += "- **Overall**: 🎉 Excellent performance\n"
    elif metrics.roc_auc > 0.8:
        report += "- **Overall**: ✅ Good performance\n"
    elif metrics.roc_auc > 0.7:
        report += "- **Overall**: 👍 Fair performance\n"
    else:
        report += "- **Overall**: ⚠️ Poor performance\n"
    
    return report


# Export evaluator components
__all__ = [
    "EvaluationMetrics",
    "RealMetricsEvaluator",
    "create_evaluator",
    "evaluate_model_predictions",
    "create_evaluation_report"
]
