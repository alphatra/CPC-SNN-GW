"""
Modular W&B logging package.

Re-exports split interfaces from the monolithic utils.wandb_enhanced_logger
to provide a clean public API without duplicating implementations.
"""

from ..wandb_enhanced_logger import (
    EnhancedWandbLogger,
    NeuromorphicMetrics,
    PerformanceMetrics,
    create_enhanced_wandb_logger,
)

__all__ = [
    "EnhancedWandbLogger",
    "NeuromorphicMetrics",
    "PerformanceMetrics",
    "create_enhanced_wandb_logger",
]


