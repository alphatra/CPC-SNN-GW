"""
Real Metrics Evaluator for CPC-SNN-GW Pipeline

✅ ENHANCED: Professional evaluation with ROC-AUC, TPR@FAR, and bootstrap CI
Based on audit recommendations for scientific GW detection metrics.

Key metrics:
- ROC-AUC: Overall classification performance
- TPR@FAR=1/30d: True positive rate at 1 false alarm per 30 days
- Precision-Recall curves with bootstrap confidence intervals
- Confusion matrices and F1 scores
- Model collapse detection
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Try importing sklearn for metrics
try:
    from sklearn.metrics import (
        roc_auc_score, roc_curve,
        precision_recall_curve, average_precision_score,
        confusion_matrix, f1_score,
        classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("scikit-learn not available, using JAX-based metrics")

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    # Core metrics
    roc_auc: float
    tpr_at_far: float  # TPR@FAR=1/30d
    precision: float
    recall: float
    f1: float
    
    # Additional metrics
    average_precision: float
    accuracy: float
    specificity: float
    
    # Confidence intervals (from bootstrap)
    roc_auc_ci: Tuple[float, float]
    tpr_at_far_ci: Tuple[float, float]
    f1_ci: Tuple[float, float]
    
    # Model health
    model_collapse: bool
    collapsed_class: Optional[int]
    
    # Curves for plotting
    roc_curve: Optional[Dict]
    pr_curve: Optional[Dict]
    confusion_matrix: Optional[np.ndarray]


class RealMetricsEvaluator:
    """
    ✅ ENHANCED: Professional metrics evaluator for GW detection.
    
    Implements:
    - ROC-AUC with bootstrap confidence intervals
    - TPR@FAR=1/30d (critical for GW detection)
    - Precision-Recall analysis
    - Model collapse detection
    - PyCBC baseline comparison
    """
    
    def __init__(self,
                 far_threshold: float = 1.0 / (30 * 24 * 3600),  # 1 per 30 days
                 bootstrap_iterations: int = 1000,
                 confidence_level: float = 0.95,
                 random_seed: int = 42):
        """
        Initialize evaluator with GW-specific parameters.
        
        Args:
            far_threshold: False alarm rate threshold (default: 1/30 days)
            bootstrap_iterations: Number of bootstrap samples for CI
            confidence_level: Confidence level for intervals (0.95 = 95% CI)
            random_seed: Random seed for reproducibility
        """
        self.far_threshold = far_threshold
        self.bootstrap_iterations = bootstrap_iterations
        self.confidence_level = confidence_level
        self.rng = np.random.RandomState(random_seed)
        
        logger.info(f"✅ Real Metrics Evaluator initialized")
        logger.info(f"   FAR threshold: 1/{30*24*3600/far_threshold:.1f} days")
        logger.info(f"   Bootstrap iterations: {bootstrap_iterations}")
        logger.info(f"   Confidence level: {confidence_level*100:.0f}%")
    
    def compute_roc_auc(self, 
                       y_true: np.ndarray,
                       y_scores: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        """
        Compute ROC-AUC with bootstrap confidence interval.
        
        Args:
            y_true: True labels (0 or 1)
            y_scores: Predicted scores/probabilities
            
        Returns:
            (roc_auc, (ci_lower, ci_upper))
        """
        if SKLEARN_AVAILABLE:
            # Primary ROC-AUC
            roc_auc = roc_auc_score(y_true, y_scores)
            
            # Bootstrap for confidence interval
            bootstrap_aucs = []
            n_samples = len(y_true)
            
            for _ in range(self.bootstrap_iterations):
                # Resample with replacement
                indices = self.rng.choice(n_samples, n_samples, replace=True)
                y_boot = y_true[indices]
                scores_boot = y_scores[indices]
                
                # Skip if only one class in bootstrap sample
                if len(np.unique(y_boot)) < 2:
                    continue
                
                try:
                    auc_boot = roc_auc_score(y_boot, scores_boot)
                    bootstrap_aucs.append(auc_boot)
                except:
                    continue
            
            # Calculate confidence interval
            if bootstrap_aucs:
                alpha = (1 - self.confidence_level) / 2
                ci_lower = np.percentile(bootstrap_aucs, alpha * 100)
                ci_upper = np.percentile(bootstrap_aucs, (1 - alpha) * 100)
            else:
                ci_lower = ci_upper = roc_auc
            
            return roc_auc, (ci_lower, ci_upper)
        else:
            # JAX-based fallback
            return self._compute_roc_auc_jax(y_true, y_scores)
    
    def _compute_roc_auc_jax(self,
                            y_true: jnp.ndarray,
                            y_scores: jnp.ndarray) -> Tuple[float, Tuple[float, float]]:
        """
        JAX-based ROC-AUC computation (fallback).
        
        Args:
            y_true: True labels
            y_scores: Predicted scores
            
        Returns:
            (roc_auc, (ci_lower, ci_upper))
        """
        # Simple approximation using trapezoid rule
        thresholds = jnp.linspace(0, 1, 100)
        tprs = []
        fprs = []
        
        for threshold in thresholds:
            predictions = (y_scores >= threshold).astype(jnp.float32)
            
            tp = jnp.sum((predictions == 1) & (y_true == 1))
            fp = jnp.sum((predictions == 1) & (y_true == 0))
            tn = jnp.sum((predictions == 0) & (y_true == 0))
            fn = jnp.sum((predictions == 0) & (y_true == 1))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tprs.append(tpr)
            fprs.append(fpr)
        
        # Compute AUC using trapezoid rule
        tprs = jnp.array(tprs)
        fprs = jnp.array(fprs)
        
        # Sort by FPR
        sort_indices = jnp.argsort(fprs)
        fprs_sorted = fprs[sort_indices]
        tprs_sorted = tprs[sort_indices]
        
        # Trapezoid rule
        roc_auc = jnp.trapz(tprs_sorted, fprs_sorted)
        
        # Simple CI estimate (±2 standard errors)
        se = jnp.sqrt(roc_auc * (1 - roc_auc) / len(y_true))
        ci_lower = max(0, roc_auc - 2 * se)
        ci_upper = min(1, roc_auc + 2 * se)
        
        return float(roc_auc), (float(ci_lower), float(ci_upper))
    
    def compute_tpr_at_far(self,
                          y_true: np.ndarray,
                          y_scores: np.ndarray,
                          far_threshold: Optional[float] = None) -> Tuple[float, Tuple[float, float]]:
        """
        Compute TPR at specified FAR threshold.
        
        ✅ CRITICAL METRIC: TPR@FAR=1/30d for GW detection.
        
        Args:
            y_true: True labels
            y_scores: Predicted scores
            far_threshold: FAR threshold (default: 1/30 days)
            
        Returns:
            (tpr_at_far, (ci_lower, ci_upper))
        """
        if far_threshold is None:
            far_threshold = self.far_threshold
        
        if SKLEARN_AVAILABLE:
            # Get ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            
            # Interpolate TPR at target FAR
            tpr_at_far = np.interp(far_threshold, fpr, tpr)
            
            # Bootstrap for confidence interval
            bootstrap_tprs = []
            n_samples = len(y_true)
            
            for _ in range(self.bootstrap_iterations):
                indices = self.rng.choice(n_samples, n_samples, replace=True)
                y_boot = y_true[indices]
                scores_boot = y_scores[indices]
                
                if len(np.unique(y_boot)) < 2:
                    continue
                
                try:
                    fpr_boot, tpr_boot, _ = roc_curve(y_boot, scores_boot)
                    tpr_at_far_boot = np.interp(far_threshold, fpr_boot, tpr_boot)
                    bootstrap_tprs.append(tpr_at_far_boot)
                except:
                    continue
            
            # Calculate CI
            if bootstrap_tprs:
                alpha = (1 - self.confidence_level) / 2
                ci_lower = np.percentile(bootstrap_tprs, alpha * 100)
                ci_upper = np.percentile(bootstrap_tprs, (1 - alpha) * 100)
            else:
                ci_lower = ci_upper = tpr_at_far
            
            logger.info(f"✅ TPR@FAR=1/30d: {tpr_at_far:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
            
            return tpr_at_far, (ci_lower, ci_upper)
        else:
            # JAX fallback
            return self._compute_tpr_at_far_jax(y_true, y_scores, far_threshold)
    
    def _compute_tpr_at_far_jax(self,
                               y_true: jnp.ndarray,
                               y_scores: jnp.ndarray,
                               far_threshold: float) -> Tuple[float, Tuple[float, float]]:
        """JAX-based TPR@FAR computation."""
        # Find threshold that gives desired FAR
        thresholds = jnp.linspace(0, 1, 1000)
        
        for threshold in thresholds:
            predictions = (y_scores >= threshold).astype(jnp.float32)
            
            fp = jnp.sum((predictions == 1) & (y_true == 0))
            tn = jnp.sum((predictions == 0) & (y_true == 0))
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            if fpr <= far_threshold:
                # Calculate TPR at this threshold
                tp = jnp.sum((predictions == 1) & (y_true == 1))
                fn = jnp.sum((predictions == 0) & (y_true == 1))
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # Simple CI
                se = jnp.sqrt(tpr * (1 - tpr) / len(y_true))
                ci_lower = max(0, tpr - 2 * se)
                ci_upper = min(1, tpr + 2 * se)
                
                return float(tpr), (float(ci_lower), float(ci_upper))
        
        return 0.0, (0.0, 0.0)
    
    def detect_model_collapse(self,
                             predictions: np.ndarray) -> Tuple[bool, Optional[int]]:
        """
        Detect if model always predicts the same class.
        
        Args:
            predictions: Model predictions
            
        Returns:
            (is_collapsed, collapsed_class)
        """
        unique_preds = np.unique(predictions)
        
        if len(unique_preds) == 1:
            logger.warning(f"⚠️ Model collapse detected! Always predicts class {unique_preds[0]}")
            return True, int(unique_preds[0])
        
        # Check for severe imbalance (>95% one class)
        pred_counts = np.bincount(predictions.astype(int))
        max_ratio = np.max(pred_counts) / len(predictions)
        
        if max_ratio > 0.95:
            dominant_class = np.argmax(pred_counts)
            logger.warning(f"⚠️ Near collapse: {max_ratio*100:.1f}% predictions are class {dominant_class}")
            return True, int(dominant_class)
        
        return False, None
    
    def evaluate(self,
                y_true: np.ndarray,
                y_scores: np.ndarray,
                y_pred: Optional[np.ndarray] = None,
                threshold: float = 0.5) -> EvaluationMetrics:
        """
        Complete evaluation with all metrics.
        
        ✅ MAIN ENTRY POINT: Computes all evaluation metrics.
        
        Args:
            y_true: True labels
            y_scores: Predicted scores/probabilities
            y_pred: Binary predictions (optional, computed from scores if None)
            threshold: Threshold for binary predictions
            
        Returns:
            EvaluationMetrics object with all metrics
        """
        # Convert to numpy if needed
        if hasattr(y_true, 'numpy'):
            y_true = y_true.numpy()
        if hasattr(y_scores, 'numpy'):
            y_scores = y_scores.numpy()
        
        # Generate binary predictions if not provided
        if y_pred is None:
            y_pred = (y_scores >= threshold).astype(int)
        
        logger.info("=" * 60)
        logger.info("🔬 Computing Real Evaluation Metrics")
        logger.info("=" * 60)
        
        # 1. ROC-AUC with CI
        roc_auc, roc_auc_ci = self.compute_roc_auc(y_true, y_scores)
        logger.info(f"ROC-AUC: {roc_auc:.3f} [{roc_auc_ci[0]:.3f}, {roc_auc_ci[1]:.3f}]")
        
        # 2. TPR@FAR=1/30d with CI
        tpr_at_far, tpr_at_far_ci = self.compute_tpr_at_far(y_true, y_scores)
        
        # 3. Precision, Recall, F1
        if SKLEARN_AVAILABLE:
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            avg_precision = average_precision_score(y_true, y_scores)
            f1 = f1_score(y_true, y_pred)
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            # Get curves for plotting
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
            roc_curve_data = {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds}
            pr_curve_data = {'precision': precision, 'recall': recall}
        else:
            # JAX fallback
            tp = jnp.sum((y_pred == 1) & (y_true == 1))
            fp = jnp.sum((y_pred == 1) & (y_true == 0))
            tn = jnp.sum((y_pred == 0) & (y_true == 0))
            fn = jnp.sum((y_pred == 0) & (y_true == 1))
            
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val) \
                if (precision_val + recall_val) > 0 else 0
            
            avg_precision = precision_val  # Simplified
            conf_matrix = np.array([[tn, fp], [fn, tp]])
            roc_curve_data = None
            pr_curve_data = None
        
        # 4. Additional metrics
        accuracy = np.mean(y_pred == y_true)
        
        # Specificity
        tn_total = np.sum((y_pred == 0) & (y_true == 0))
        fp_total = np.sum((y_pred == 1) & (y_true == 0))
        specificity = tn_total / (tn_total + fp_total) if (tn_total + fp_total) > 0 else 0
        
        # 5. F1 with bootstrap CI
        f1_scores_boot = []
        for _ in range(self.bootstrap_iterations):
            indices = self.rng.choice(len(y_true), len(y_true), replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            if len(np.unique(y_true_boot)) < 2:
                continue
            
            if SKLEARN_AVAILABLE:
                f1_boot = f1_score(y_true_boot, y_pred_boot)
            else:
                tp_b = np.sum((y_pred_boot == 1) & (y_true_boot == 1))
                fp_b = np.sum((y_pred_boot == 1) & (y_true_boot == 0))
                fn_b = np.sum((y_pred_boot == 0) & (y_true_boot == 1))
                
                prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
                rec_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
                f1_boot = 2 * (prec_b * rec_b) / (prec_b + rec_b) if (prec_b + rec_b) > 0 else 0
            
            f1_scores_boot.append(f1_boot)
        
        if f1_scores_boot:
            alpha = (1 - self.confidence_level) / 2
            f1_ci = (np.percentile(f1_scores_boot, alpha * 100),
                    np.percentile(f1_scores_boot, (1 - alpha) * 100))
        else:
            f1_ci = (f1, f1)
        
        # 6. Model collapse detection
        model_collapse, collapsed_class = self.detect_model_collapse(y_pred)
        
        # Log summary
        logger.info(f"Accuracy: {accuracy:.3f}")
        logger.info(f"Precision: {precision_val if not SKLEARN_AVAILABLE else np.mean(precision):.3f}")
        logger.info(f"Recall: {recall_val if not SKLEARN_AVAILABLE else np.mean(recall):.3f}")
        logger.info(f"F1 Score: {f1:.3f} [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]")
        logger.info(f"Specificity: {specificity:.3f}")
        
        if model_collapse:
            logger.warning(f"⚠️ Model health issue: collapse to class {collapsed_class}")
        else:
            logger.info("✅ Model health: No collapse detected")
        
        logger.info("=" * 60)
        
        # Create metrics object
        metrics = EvaluationMetrics(
            roc_auc=float(roc_auc),
            tpr_at_far=float(tpr_at_far),
            precision=float(precision_val if not SKLEARN_AVAILABLE else np.mean(precision)),
            recall=float(recall_val if not SKLEARN_AVAILABLE else np.mean(recall)),
            f1=float(f1),
            average_precision=float(avg_precision),
            accuracy=float(accuracy),
            specificity=float(specificity),
            roc_auc_ci=roc_auc_ci,
            tpr_at_far_ci=tpr_at_far_ci,
            f1_ci=f1_ci,
            model_collapse=model_collapse,
            collapsed_class=collapsed_class,
            roc_curve=roc_curve_data,
            pr_curve=pr_curve_data,
            confusion_matrix=conf_matrix
        )
        
        return metrics
    
    def plot_metrics(self, metrics: EvaluationMetrics, save_path: Optional[Path] = None):
        """
        Plot evaluation metrics.
        
        Args:
            metrics: Evaluation metrics object
            save_path: Path to save plots (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. ROC Curve
        if metrics.roc_curve:
            axes[0, 0].plot(metrics.roc_curve['fpr'], metrics.roc_curve['tpr'])
            axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0, 0].set_xlabel('False Positive Rate')
            axes[0, 0].set_ylabel('True Positive Rate')
            axes[0, 0].set_title(f'ROC Curve (AUC={metrics.roc_auc:.3f})')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curve
        if metrics.pr_curve:
            axes[0, 1].plot(metrics.pr_curve['recall'], metrics.pr_curve['precision'])
            axes[0, 1].set_xlabel('Recall')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_title(f'Precision-Recall Curve (AP={metrics.average_precision:.3f})')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confusion Matrix
        if metrics.confusion_matrix is not None:
            sns.heatmap(metrics.confusion_matrix, annot=True, fmt='d', 
                       ax=axes[1, 0], cmap='Blues')
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_ylabel('True')
            axes[1, 0].set_title('Confusion Matrix')
        
        # 4. Metrics Summary
        axes[1, 1].axis('off')
        summary_text = f"""
        Performance Metrics:
        
        ROC-AUC: {metrics.roc_auc:.3f} [{metrics.roc_auc_ci[0]:.3f}, {metrics.roc_auc_ci[1]:.3f}]
        TPR@FAR=1/30d: {metrics.tpr_at_far:.3f} [{metrics.tpr_at_far_ci[0]:.3f}, {metrics.tpr_at_far_ci[1]:.3f}]
        
        Accuracy: {metrics.accuracy:.3f}
        Precision: {metrics.precision:.3f}
        Recall: {metrics.recall:.3f}
        F1 Score: {metrics.f1:.3f} [{metrics.f1_ci[0]:.3f}, {metrics.f1_ci[1]:.3f}]
        Specificity: {metrics.specificity:.3f}
        
        Model Health: {'⚠️ COLLAPSED' if metrics.model_collapse else '✅ OK'}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, 
                       verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"✅ Plots saved to {save_path}")
        
        plt.show()


def create_evaluator(far_threshold: float = 1.0 / (30 * 24 * 3600)) -> RealMetricsEvaluator:
    """
    Factory function to create evaluator.
    
    Args:
        far_threshold: FAR threshold for TPR calculation
        
    Returns:
        Configured evaluator
    """
    return RealMetricsEvaluator(far_threshold=far_threshold)


if __name__ == "__main__":
    # Test evaluation
    logging.basicConfig(level=logging.INFO)
    
    # Generate test data
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.randint(0, 2, n_samples)
    y_scores = np.random.beta(2, 5, n_samples)  # Skewed scores
    y_scores[y_true == 1] += 0.3  # Make positive class have higher scores
    y_scores = np.clip(y_scores, 0, 1)
    
    # Create evaluator and compute metrics
    evaluator = create_evaluator()
    metrics = evaluator.evaluate(y_true, y_scores)
    
    # Plot results
    evaluator.plot_metrics(metrics)
    
    print("\n✅ Evaluation complete!")
