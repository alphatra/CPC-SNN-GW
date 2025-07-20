"""
Label Analytics and Statistics

Comprehensive analytics system for gravitational wave detection labels with:
- Class distribution analysis and visualization
- Dataset summary generation and reporting  
- Class weight calculation for balanced training
- Visualization configuration and color schemes
- Detailed reporting and logging

Features:
- JAX-optimized class weight calculations
- Comprehensive dataset summaries
- Visualization configuration support
- Detailed logging and reporting
- Export capabilities for reports
"""

from typing import Dict, List, Union, Optional, Tuple, Any
import jax
import jax.numpy as jnp
import numpy as np
import logging
from dataclasses import dataclass

from .label_enums import (
    GWSignalType,
    CANONICAL_LABELS,
    LABEL_NAMES,
    LABEL_DESCRIPTIONS,
    COLOR_SCHEMES,
    NUM_CLASSES,
    get_cmap_colors,
    get_label_info
)

from .label_validation import _fast_class_distribution

logger = logging.getLogger(__name__)


@dataclass
class DatasetStatistics:
    """Comprehensive statistics for a label dataset."""
    total_samples: int
    class_distribution: Dict[int, int]
    class_percentages: Dict[int, float]
    class_weights: Dict[int, float]
    imbalance_ratio: float
    entropy: float
    most_common_class: int
    least_common_class: int
    missing_classes: List[int]


def get_class_weights(labels: jnp.ndarray,
                     method: str = 'balanced',
                     class_weight: Optional[Dict] = None) -> jnp.ndarray:
    """
    Calculate class weights for balanced training with JAX optimization.
    
    Args:
        labels: Array of integer labels
        method: Weight calculation method ('balanced', 'inverse', 'log', 'custom')
        class_weight: Custom weights dictionary (for method='custom')
        
    Returns:
        JAX array of class weights
    """
    if len(labels) == 0:
        return jnp.ones(NUM_CLASSES)
    
    # Fast class distribution using JAX
    class_counts = _fast_class_distribution(labels, NUM_CLASSES)
    total_samples = len(labels)
    
    if method == 'balanced':
        # Sklearn-style balanced weights: n_samples / (n_classes * class_count)
        weights = total_samples / (NUM_CLASSES * (class_counts + 1e-8))  # Add epsilon to avoid division by zero
        
    elif method == 'inverse':
        # Simple inverse frequency
        weights = 1.0 / (class_counts + 1e-8)
        
    elif method == 'log':
        # Logarithmic scaling to reduce extreme weights
        weights = jnp.log(total_samples / (class_counts + 1e-8) + 1)
        
    elif method == 'custom':
        if class_weight is None:
            raise ValueError("class_weight dictionary required for method='custom'")
        
        # Convert custom weights to JAX array
        weights = jnp.array([class_weight.get(i, 1.0) for i in range(NUM_CLASSES)])
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'balanced', 'inverse', 'log', or 'custom'.")
    
    # Normalize weights to sum to number of classes (sklearn convention)
    normalized_weights = weights * NUM_CLASSES / jnp.sum(weights)
    
    logger.info(f"Calculated class weights using {method} method")
    logger.debug(f"Class weights: {[f'{i}:{w:.3f}' for i, w in enumerate(normalized_weights)]}")
    
    return normalized_weights


def calculate_dataset_statistics(labels: jnp.ndarray) -> DatasetStatistics:
    """
    Calculate comprehensive statistics for a dataset.
    
    Args:
        labels: Array of integer labels
        
    Returns:
        DatasetStatistics object with comprehensive metrics
    """
    if len(labels) == 0:
        return DatasetStatistics(
            total_samples=0, class_distribution={}, class_percentages={},
            class_weights={}, imbalance_ratio=1.0, entropy=0.0,
            most_common_class=0, least_common_class=0, missing_classes=list(range(NUM_CLASSES))
        )
    
    total_samples = len(labels)
    
    # Fast class distribution
    class_counts = _fast_class_distribution(labels, NUM_CLASSES)
    
    # Convert to Python dict
    class_distribution = {i: int(count) for i, count in enumerate(class_counts)}
    
    # Calculate percentages
    class_percentages = {i: count / total_samples * 100 for i, count in class_distribution.items()}
    
    # Calculate class weights
    weights = get_class_weights(labels, method='balanced')
    class_weights = {i: float(weight) for i, weight in enumerate(weights)}
    
    # Find present classes
    present_classes = [i for i, count in class_distribution.items() if count > 0]
    missing_classes = [i for i in range(NUM_CLASSES) if i not in present_classes]
    
    # Imbalance ratio (max/min class frequency)
    if present_classes:
        present_counts = [class_distribution[i] for i in present_classes]
        max_count = max(present_counts)
        min_count = min(present_counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        most_common_class = max(present_classes, key=lambda i: class_distribution[i])
        least_common_class = min(present_classes, key=lambda i: class_distribution[i])
    else:
        imbalance_ratio = 1.0
        most_common_class = 0
        least_common_class = 0
    
    # Calculate entropy
    probabilities = jnp.array([class_distribution[i] / total_samples for i in range(NUM_CLASSES)])
    # Add small epsilon to avoid log(0)
    probabilities = jnp.where(probabilities > 0, probabilities, 1e-8)
    entropy = float(-jnp.sum(probabilities * jnp.log2(probabilities)))
    
    return DatasetStatistics(
        total_samples=total_samples,
        class_distribution=class_distribution,
        class_percentages=class_percentages,
        class_weights=class_weights,
        imbalance_ratio=imbalance_ratio,
        entropy=entropy,
        most_common_class=most_common_class,
        least_common_class=least_common_class,
        missing_classes=missing_classes
    )


def create_dataset_summary(labels: jnp.ndarray,
                         dataset_name: str = "Dataset",
                         include_details: bool = True) -> Dict[str, Any]:
    """
    Create comprehensive dataset summary with statistics and recommendations.
    
    Args:
        labels: Array of integer labels
        dataset_name: Name of the dataset for reporting
        include_details: Whether to include detailed class information
        
    Returns:
        Dictionary with complete dataset summary
    """
    stats = calculate_dataset_statistics(labels)
    
    summary = {
        'dataset_name': dataset_name,
        'total_samples': stats.total_samples,
        'num_classes': NUM_CLASSES,
        'present_classes': len([c for c in stats.class_distribution.values() if c > 0]),
        'missing_classes': len(stats.missing_classes),
        'imbalance_ratio': stats.imbalance_ratio,
        'entropy': stats.entropy,
        'timestamp': logging.Formatter().formatTime(logging.LogRecord(
            '', 0, '', 0, '', (), None
        ))
    }
    
    if include_details:
        summary.update({
            'class_distribution': stats.class_distribution,
            'class_percentages': stats.class_percentages,
            'class_weights': stats.class_weights,
            'most_common_class': {
                'class_id': stats.most_common_class,
                'class_name': LABEL_NAMES.get(GWSignalType(stats.most_common_class), 'Unknown'),
                'count': stats.class_distribution.get(stats.most_common_class, 0),
                'percentage': stats.class_percentages.get(stats.most_common_class, 0.0)
            },
            'least_common_class': {
                'class_id': stats.least_common_class,
                'class_name': LABEL_NAMES.get(GWSignalType(stats.least_common_class), 'Unknown'),
                'count': stats.class_distribution.get(stats.least_common_class, 0),
                'percentage': stats.class_percentages.get(stats.least_common_class, 0.0)
            },
            'missing_classes_details': [
                {
                    'class_id': class_id,
                    'class_name': LABEL_NAMES.get(GWSignalType(class_id), 'Unknown'),
                    'canonical': CANONICAL_LABELS.get(GWSignalType(class_id), 'unknown')
                }
                for class_id in stats.missing_classes
            ]
        })
    
    # Add recommendations
    recommendations = []
    
    if stats.imbalance_ratio > 10:
        recommendations.append("Severe class imbalance detected - consider class balancing techniques")
    elif stats.imbalance_ratio > 3:
        recommendations.append("Moderate class imbalance - consider using class weights")
    
    if stats.missing_classes:
        recommendations.append(f"Missing {len(stats.missing_classes)} classes - check data collection")
    
    if stats.entropy < 1.0:
        recommendations.append("Low entropy - dataset may lack diversity")
    
    if stats.total_samples < 1000:
        recommendations.append("Small dataset - consider data augmentation")
    
    summary['recommendations'] = recommendations
    
    return summary


def log_dataset_info(labels: jnp.ndarray,
                    dataset_name: str = "Dataset",
                    log_level: int = logging.INFO) -> None:
    """
    Log comprehensive dataset information at specified level.
    
    Args:
        labels: Array of integer labels
        dataset_name: Name of the dataset
        log_level: Logging level for output
    """
    stats = calculate_dataset_statistics(labels)
    
    logger.log(log_level, f"=== {dataset_name} Statistics ===")
    logger.log(log_level, f"Total samples: {stats.total_samples:,}")
    logger.log(log_level, f"Number of classes: {NUM_CLASSES}")
    logger.log(log_level, f"Present classes: {NUM_CLASSES - len(stats.missing_classes)}/{NUM_CLASSES}")
    
    if stats.total_samples > 0:
        logger.log(log_level, f"Imbalance ratio: {stats.imbalance_ratio:.2f}:1")
        logger.log(log_level, f"Entropy: {stats.entropy:.3f}")
        
        # Log class distribution
        logger.log(log_level, "Class distribution:")
        for class_id in range(NUM_CLASSES):
            count = stats.class_distribution.get(class_id, 0)
            percentage = stats.class_percentages.get(class_id, 0.0)
            weight = stats.class_weights.get(class_id, 0.0)
            class_name = LABEL_NAMES.get(GWSignalType(class_id), 'Unknown')
            
            if count > 0:
                logger.log(log_level, f"  {class_id} ({class_name}): {count:,} ({percentage:.1f}%) [weight: {weight:.3f}]")
            else:
                logger.log(log_level, f"  {class_id} ({class_name}): MISSING")
        
        # Log warnings for issues
        if stats.imbalance_ratio > 10:
            logger.warning(f"Severe class imbalance in {dataset_name}: {stats.imbalance_ratio:.1f}:1")
        
        if stats.missing_classes:
            logger.warning(f"Missing classes in {dataset_name}: {stats.missing_classes}")


def create_label_visualization_config(scheme: str = 'default') -> Dict[str, Any]:
    """
    Create configuration for label visualization including colors and legends.
    
    Args:
        scheme: Color scheme name
        
    Returns:
        Dictionary with visualization configuration
    """
    colors = get_cmap_colors(scheme)
    
    config = {
        'color_scheme': scheme,
        'colors': colors,
        'num_classes': NUM_CLASSES,
        'class_labels': [LABEL_NAMES[GWSignalType(i)] for i in range(NUM_CLASSES)],
        'canonical_labels': [CANONICAL_LABELS[GWSignalType(i)] for i in range(NUM_CLASSES)],
        'descriptions': [LABEL_DESCRIPTIONS[GWSignalType(i)] for i in range(NUM_CLASSES)],
        'colormap_name': f'gw_labels_{scheme}',
        'legend_config': {
            'title': 'GW Signal Types',
            'labels': [f"{i}: {LABEL_NAMES[GWSignalType(i)]}" for i in range(NUM_CLASSES)],
            'colors': colors
        },
        'matplotlib_config': {
            'cmap': {
                'colors': colors,
                'N': NUM_CLASSES,
                'name': f'gw_labels_{scheme}'
            },
            'norm': {
                'vmin': 0,
                'vmax': NUM_CLASSES - 1
            }
        }
    }
    
    return config


def dataset_to_canonical(dataset: Dict,
                        label_key: str = 'labels',
                        inplace: bool = False) -> Dict:
    """
    Convert dataset labels to canonical string format.
    
    Args:
        dataset: Dataset dictionary
        label_key: Key for labels in dataset
        inplace: Whether to modify dataset in place
        
    Returns:
        Dataset with canonical string labels
    """
    if not inplace:
        dataset = dataset.copy()
    
    if label_key in dataset:
        labels = dataset[label_key]
        
        if isinstance(labels, (list, np.ndarray, jnp.ndarray)):
            canonical_labels = []
            
            for label in labels:
                try:
                    if isinstance(label, (int, np.integer)):
                        if label in CANONICAL_LABELS:
                            canonical_labels.append(CANONICAL_LABELS[GWSignalType(label)])
                        else:
                            logger.warning(f"Invalid label {label}, using 'unknown'")
                            canonical_labels.append('unknown')
                    else:
                        # Already string, validate and normalize
                        from .label_validation import normalize_labels
                        normalized, _ = normalize_labels([label], expected_type='categorical')
                        canonical_labels.append(normalized[0])
                        
                except Exception as e:
                    logger.error(f"Failed to convert label {label}: {e}")
                    canonical_labels.append('unknown')
            
            dataset[label_key] = canonical_labels
            
    # Handle nested datasets (e.g., train/test splits)
    for key, value in dataset.items():
        if isinstance(value, dict) and label_key in value:
            value_copy = value.copy() if not inplace else value
            labels = value_copy[label_key]
            
            canonical_labels = []
            for label in labels:
                try:
                    if isinstance(label, (int, np.integer)):
                        if label in CANONICAL_LABELS:
                            canonical_labels.append(CANONICAL_LABELS[GWSignalType(label)])
                        else:
                            canonical_labels.append('unknown')
                    else:
                        from .label_validation import normalize_labels
                        normalized, _ = normalize_labels([label], expected_type='categorical')
                        canonical_labels.append(normalized[0])
                except Exception:
                    canonical_labels.append('unknown')
            
            value_copy[label_key] = canonical_labels
            if not inplace:
                dataset[key] = value_copy
    
    return dataset


def create_label_report(labels: jnp.ndarray,
                       dataset_name: str = "Dataset",
                       output_format: str = 'dict') -> Union[Dict[str, Any], str]:
    """
    Generate comprehensive label analysis report.
    
    Args:
        labels: Array of integer labels
        dataset_name: Name of the dataset
        output_format: Output format ('dict', 'text', 'markdown')
        
    Returns:
        Report in specified format
    """
    summary = create_dataset_summary(labels, dataset_name, include_details=True)
    stats = calculate_dataset_statistics(labels)
    
    if output_format == 'dict':
        return summary
    
    elif output_format == 'text':
        report_lines = [
            f"=== {dataset_name} Label Analysis Report ===",
            f"Generated: {summary['timestamp']}",
            "",
            f"Dataset Overview:",
            f"  Total samples: {summary['total_samples']:,}",
            f"  Classes present: {summary['present_classes']}/{summary['num_classes']}",
            f"  Missing classes: {summary['missing_classes']}",
            f"  Imbalance ratio: {summary['imbalance_ratio']:.2f}:1",
            f"  Entropy: {summary['entropy']:.3f}",
            "",
            f"Class Distribution:"
        ]
        
        for i in range(NUM_CLASSES):
            count = summary['class_distribution'].get(i, 0)
            percentage = summary['class_percentages'].get(i, 0.0)
            weight = summary['class_weights'].get(i, 0.0)
            class_name = LABEL_NAMES.get(GWSignalType(i), 'Unknown')
            
            if count > 0:
                report_lines.append(f"  {i} ({class_name}): {count:,} ({percentage:.1f}%) [weight: {weight:.3f}]")
            else:
                report_lines.append(f"  {i} ({class_name}): MISSING")
        
        if summary['recommendations']:
            report_lines.extend(["", "Recommendations:"])
            for rec in summary['recommendations']:
                report_lines.append(f"  - {rec}")
        
        return "\n".join(report_lines)
    
    elif output_format == 'markdown':
        report_lines = [
            f"# {dataset_name} Label Analysis Report",
            f"*Generated: {summary['timestamp']}*",
            "",
            "## Dataset Overview",
            f"- **Total samples:** {summary['total_samples']:,}",
            f"- **Classes present:** {summary['present_classes']}/{summary['num_classes']}",
            f"- **Missing classes:** {summary['missing_classes']}",
            f"- **Imbalance ratio:** {summary['imbalance_ratio']:.2f}:1",
            f"- **Entropy:** {summary['entropy']:.3f}",
            "",
            "## Class Distribution",
            "| Class | Name | Count | Percentage | Weight |",
            "|-------|------|-------|------------|--------|"
        ]
        
        for i in range(NUM_CLASSES):
            count = summary['class_distribution'].get(i, 0)
            percentage = summary['class_percentages'].get(i, 0.0)
            weight = summary['class_weights'].get(i, 0.0)
            class_name = LABEL_NAMES.get(GWSignalType(i), 'Unknown')
            
            if count > 0:
                report_lines.append(f"| {i} | {class_name} | {count:,} | {percentage:.1f}% | {weight:.3f} |")
            else:
                report_lines.append(f"| {i} | {class_name} | MISSING | 0.0% | {weight:.3f} |")
        
        if summary['recommendations']:
            report_lines.extend(["", "## Recommendations"])
            for rec in summary['recommendations']:
                report_lines.append(f"- {rec}")
        
        return "\n".join(report_lines)
    
    else:
        raise ValueError(f"Unknown output format: {output_format}")


# Export main functions
__all__ = [
    'DatasetStatistics',
    'get_class_weights',
    'calculate_dataset_statistics',
    'create_dataset_summary',
    'log_dataset_info',
    'create_label_visualization_config',
    'dataset_to_canonical',
    'create_label_report'
] 