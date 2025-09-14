"""
Factory functions for creating logging components.

Extracted from wandb_enhanced_logger.py for better modularity.
"""

from typing import Dict, List, Optional, Any
from .wandb_logger import EnhancedWandbLogger
from .metrics import NeuromorphicMetrics, PerformanceMetrics


def create_enhanced_wandb_logger(
    project: str = "neuromorphic-gw-detection",
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    output_dir: str = "wandb_outputs",
    **kwargs
) -> EnhancedWandbLogger:
    """
    Factory function to create enhanced W&B logger with sensible defaults.
    
    Args:
        project: W&B project name
        name: Run name (auto-generated if None)
        config: Configuration dictionary
        tags: List of tags for the run
        notes: Run description
        output_dir: Output directory for logs
        **kwargs: Additional arguments for EnhancedWandbLogger
        
    Returns:
        Configured EnhancedWandbLogger instance
    """
    return EnhancedWandbLogger(
        project=project,
        name=name,
        config=config,
        tags=tags or ["neuromorphic", "gravitational-waves", "cpc-snn"],
        notes=notes,
        output_dir=output_dir,
        **kwargs
    )


def create_neuromorphic_metrics(
    spike_rate: float = 0.0,
    encoding_fidelity: float = 0.0,
    contrastive_accuracy: float = 0.0,
    **kwargs
) -> NeuromorphicMetrics:
    """
    Factory function to create neuromorphic metrics with common values.
    
    Args:
        spike_rate: Average spike rate (Hz)
        encoding_fidelity: Reconstruction quality
        contrastive_accuracy: CPC contrastive task accuracy
        **kwargs: Additional metric values
        
    Returns:
        NeuromorphicMetrics instance
    """
    return NeuromorphicMetrics(
        spike_rate=spike_rate,
        encoding_fidelity=encoding_fidelity,
        contrastive_accuracy=contrastive_accuracy,
        **kwargs
    )


def create_performance_metrics(
    inference_latency_ms: float = 0.0,
    memory_usage_mb: float = 0.0,
    cpu_usage_percent: float = 0.0,
    **kwargs
) -> PerformanceMetrics:
    """
    Factory function to create performance metrics with common values.
    
    Args:
        inference_latency_ms: Inference latency in milliseconds
        memory_usage_mb: Memory usage in MB
        cpu_usage_percent: CPU usage percentage
        **kwargs: Additional metric values
        
    Returns:
        PerformanceMetrics instance
    """
    return PerformanceMetrics(
        inference_latency_ms=inference_latency_ms,
        memory_usage_mb=memory_usage_mb,
        cpu_usage_percent=cpu_usage_percent,
        **kwargs
    )
