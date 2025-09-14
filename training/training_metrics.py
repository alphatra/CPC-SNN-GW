"""
Training Metrics (MODULAR)

This file delegates to modular monitoring components for better maintainability.
The actual implementation has been split into:
- monitoring/core.py: TrainingMetrics + ExperimentTracker
- monitoring/stopping.py: EarlyStoppingMonitor
- monitoring/profiler.py: PerformanceProfiler + EnhancedMetricsLogger

This file maintains backward compatibility through delegation.

Training Metrics: Monitoring and Experiment Tracking
"""

import logging
import warnings

# Import from new modular components
from .monitoring import (
    TrainingMetrics,
    ExperimentTracker,
    create_training_metrics,
    EarlyStoppingMonitor,
    PerformanceProfiler,
    EnhancedMetricsLogger,
    create_enhanced_metrics_logger,
)

logger = logging.getLogger(__name__)

# ===== BACKWARD COMPATIBILITY EXPORTS =====
# All classes and functions are now imported from modular components

# Export everything for backward compatibility
__all__ = [
    # Core metrics (now modular)
    "TrainingMetrics",
    "ExperimentTracker",
    "create_training_metrics",

    # Early stopping (now modular)
    "EarlyStoppingMonitor",

    # Profiling and logging (now modular)
    "PerformanceProfiler",
    "EnhancedMetricsLogger",
    "create_enhanced_metrics_logger",
]

# ===== DEPRECATION NOTICE =====
def _show_migration_notice():
    """Show migration notice for direct imports."""
    warnings.warn(
        "Direct imports from training_metrics.py are deprecated. "
        "Use modular imports: from training.monitoring import TrainingMetrics, ExperimentTracker",
        DeprecationWarning,
        stacklevel=3,
    )


# Show notice when module is imported directly
logger.info("📦 Using modular monitoring components (training_metrics.py → monitoring/)")
_show_migration_notice()

"""
Training Metrics (MODULAR)

This file delegates to modular monitoring components for better maintainability.
The actual implementation has been split into:
- monitoring/core.py: TrainingMetrics + ExperimentTracker
- monitoring/stopping.py: EarlyStoppingMonitor
- monitoring/profiler.py: PerformanceProfiler + EnhancedMetricsLogger

This file maintains backward compatibility through delegation.

Training Metrics: Monitoring and Experiment Tracking
"""

import logging
import warnings

# Import from new modular components
from .monitoring import (
    TrainingMetrics,
    ExperimentTracker,
    create_training_metrics,
    EarlyStoppingMonitor,
    PerformanceProfiler,
    EnhancedMetricsLogger,
    create_enhanced_metrics_logger
)

logger = logging.getLogger(__name__)

# ===== BACKWARD COMPATIBILITY EXPORTS =====
# All classes and functions are now imported from modular components

# Export everything for backward compatibility
__all__ = [
    # Core metrics (now modular)
    "TrainingMetrics",
    "ExperimentTracker",
    "create_training_metrics",
    
    # Early stopping (now modular)
    "EarlyStoppingMonitor",
    
    # Profiling and logging (now modular)
    "PerformanceProfiler", 
    "EnhancedMetricsLogger",
    "create_enhanced_metrics_logger"
]

# ===== DEPRECATION NOTICE =====
def _show_migration_notice():
    """Show migration notice for direct imports."""
    warnings.warn(
        "Direct imports from training_metrics.py are deprecated. "
        "Use modular imports: from training.monitoring import TrainingMetrics, ExperimentTracker",
        DeprecationWarning,
        stacklevel=3
    )

# Show notice when module is imported directly
logger.info("📦 Using modular monitoring components (training_metrics.py → monitoring/)")
_show_migration_notice()
