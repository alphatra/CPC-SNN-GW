"""
Modular logging package for CPC-SNN-GW.

This module provides comprehensive logging capabilities with neuromorphic-specific
metrics, performance monitoring, and visualization support.
"""

# Main logger class
from .wandb_logger import EnhancedWandbLogger

# Metric classes
from .metrics import NeuromorphicMetrics, PerformanceMetrics, SystemInfo

# Visualization components  
from .visualizations import SpikeVisualizer, GradientVisualizer, DashboardCreator

# Factory functions
from .factories import (
    create_enhanced_wandb_logger,
    create_neuromorphic_metrics,
    create_performance_metrics
)

__all__ = [
    # Main classes
    'EnhancedWandbLogger',
    'NeuromorphicMetrics', 
    'PerformanceMetrics',
    'SystemInfo',
    'SpikeVisualizer',
    'GradientVisualizer', 
    'DashboardCreator',
    
    # Factory functions
    'create_enhanced_wandb_logger',
    'create_neuromorphic_metrics',
    'create_performance_metrics'
]
