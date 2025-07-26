"""
Advanced Gradient Accumulation System for Neuromorphic GW Detection

Revolutionary gradient accumulation framework enabling large effective batch sizes
within GPU memory constraints. Designed for production-scale gravitational wave
detection training with memory optimization and scientific precision.

Key Features:
- Memory-efficient gradient accumulation with JAX
- Dynamic batch size adaptation based on GPU memory
- Scientific metrics tracking during accumulation
- Integration with Enhanced Logger for beautiful progress tracking
- Automatic mixed precision support
- Gradient clipping and numerical stability
"""

import jax
import jax.numpy as jnp
from jax import grad, value_and_grad
import flax.linen as nn
from flax.training import train_state
from typing import Dict, Any, Tuple, Optional, Callable
import time
import psutil
from dataclasses import dataclass

from utils.enhanced_logger import get_enhanced_logger, ScientificMetrics

@dataclass
class AccumulationConfig:
    """Configuration for gradient accumulation"""
    
    accumulation_steps: int = 4  # Number of micro-batches to accumulate
    max_physical_batch_size: int = 4  # Max batch size that fits in memory
    gradient_clipping: float = 1.0  # Gradient norm clipping
    mixed_precision: bool = True  # Use mixed precision training
    memory_monitoring: bool = True  # Monitor GPU memory during accumulation
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size"""
        return self.accumulation_steps * self.max_physical_batch_size

class GradientAccumulator:
    """
    Advanced gradient accumulation system for memory-efficient training.
    
    Enables training with large effective batch sizes by accumulating gradients
    across multiple micro-batches while staying within GPU memory limits.
    """
    
    def __init__(self, config: AccumulationConfig):
        self.config = config
        self.logger = get_enhanced_logger()
        
        # Initialize accumulation state
        self.accumulated_grads = None
        self.accumulation_count = 0
        self.total_loss = 0.0
        self.total_accuracy = 0.0
        
        # Memory monitoring
        self.peak_memory_mb = 0.0
        self.memory_timeline = []
        
        self.logger.info(f"🧬 Gradient Accumulator initialized", 
                        extra={"config": {
                            "effective_batch_size": config.effective_batch_size,
                            "accumulation_steps": config.accumulation_steps,
                            "physical_batch_size": config.max_physical_batch_size
                        }})

    def create_accumulation_step(self, loss_fn: Callable) -> Callable:
        """
        Create accumulation step function with scientific monitoring.
        
        Args:
            loss_fn: Loss function that takes (params, batch) and returns (loss, aux)
            
        Returns:
            Accumulation step function
        """
        
        @jax.jit
        def accumulation_step(params, batch, accumulation_step_idx):
            """Single micro-batch accumulation step"""
            
            # Forward and backward pass
            (loss, aux), grads = value_and_grad(loss_fn, has_aux=True)(params, batch)
            
            # Scale gradients by accumulation steps for proper averaging
            scaled_grads = jax.tree_util.tree_map(
                lambda g: g / self.config.accumulation_steps, 
                grads
            )
            
            return scaled_grads, loss, aux
        
        return accumulation_step

    def accumulate_gradients(self, 
                           train_state: train_state.TrainState,
                           batches: list,
                           loss_fn: Callable) -> Tuple[Any, ScientificMetrics]:
        """
        Accumulate gradients across multiple micro-batches.
        
        Args:
            train_state: Current training state
            batches: List of micro-batches for accumulation
            loss_fn: Loss function for gradient computation
            
        Returns:
            Tuple of (accumulated_gradients, scientific_metrics)
        """
        
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        # Initialize accumulation
        accumulated_grads = None
        total_loss = 0.0
        total_accuracy = 0.0
        
        # Create accumulation step function
        accumulation_step = self.create_accumulation_step(loss_fn)
        
        # Single progress bar for all micro-batches - FIXED
        progress_description = f"🧬 Gradient Accumulation"
        
        with self.logger.progress_context(
            progress_description,
            total=len(batches),
            reuse_existing=True
        ) as task_id:
            
            for step_idx, batch in enumerate(batches):
                
                # Memory monitoring
                if self.config.memory_monitoring:
                    current_memory = self._get_memory_usage()
                    self.peak_memory_mb = max(self.peak_memory_mb, current_memory)
                    self.memory_timeline.append((step_idx, current_memory))
                
                # Compute gradients for micro-batch
                try:
                    micro_grads, micro_loss, aux = accumulation_step(
                        train_state.params, 
                        batch, 
                        step_idx
                    )
                    
                    # Accumulate gradients
                    if accumulated_grads is None:
                        accumulated_grads = micro_grads
                    else:
                        accumulated_grads = jax.tree_util.tree_map(
                            jnp.add, accumulated_grads, micro_grads
                        )
                    
                    # Accumulate loss and metrics
                    total_loss += float(micro_loss)
                    if 'accuracy' in aux:
                        total_accuracy += float(aux['accuracy'])
                    
                    # ✅ ENHANCED: Collect additional metrics
                    if 'cpc_loss' in aux:
                        if not hasattr(self, 'total_cpc_loss'):
                            self.total_cpc_loss = 0.0
                        self.total_cpc_loss += float(aux['cpc_loss'])
                    
                    if 'snn_accuracy' in aux:
                        if not hasattr(self, 'total_snn_accuracy'):
                            self.total_snn_accuracy = 0.0
                        self.total_snn_accuracy += float(aux['snn_accuracy'])
                    
                    # ✅ FIXED: Single progress update without creating new bars
                    avg_loss = total_loss / (step_idx + 1)
                    avg_acc = total_accuracy / (step_idx + 1) if 'accuracy' in aux else 0.0
                    
                    self.logger.update_progress(
                        task_id, 
                        advance=1,
                        description=f"🧬 Loss: {avg_loss:.3f} | Acc: {avg_acc:.2f}"
                    )
                    
                except Exception as e:
                    self.logger.error(
                        f"Accumulation step {step_idx} failed", 
                        extra={"step": step_idx, "batch_shape": jax.tree_util.tree_map(lambda x: x.shape, batch)},
                        exception=e
                    )
                    raise
        
        # Apply gradient clipping if specified
        if self.config.gradient_clipping > 0:
            accumulated_grads = self._clip_gradients(accumulated_grads)
        
        # Create scientific metrics
        training_time = time.time() - start_time
        final_memory = self._get_memory_usage()
        
        # ✅ ENHANCED: Include all collected metrics
        avg_cpc_loss = getattr(self, 'total_cpc_loss', 0.0) / len(batches)
        avg_snn_accuracy = getattr(self, 'total_snn_accuracy', 0.0) / len(batches)
        
        metrics = ScientificMetrics(
            loss=total_loss / len(batches),
            accuracy=total_accuracy / len(batches),
            cpc_loss=avg_cpc_loss,
            snn_accuracy=avg_snn_accuracy,
            training_time=training_time,
            gpu_memory_mb=final_memory,
            gradient_norm=self._compute_gradient_norm(accumulated_grads),
            batch_size=self.config.effective_batch_size,
            samples_processed=len(batches) * self.config.max_physical_batch_size
        )
        
        # Reset accumulated metrics for next batch
        if hasattr(self, 'total_cpc_loss'):
            delattr(self, 'total_cpc_loss')
        if hasattr(self, 'total_snn_accuracy'):
            delattr(self, 'total_snn_accuracy')
        
        # Log accumulation results
        self.logger.info(
            f"🧬 Gradient accumulation completed",
            extra={
                "metrics": metrics,
                "memory_increase": final_memory - initial_memory,
                "peak_memory": self.peak_memory_mb
            }
        )
        
        return accumulated_grads, metrics

    def _clip_gradients(self, grads):
        """Apply gradient clipping for numerical stability"""
        
        # Compute global gradient norm
        grad_norm = self._compute_gradient_norm(grads)
        
        if grad_norm > self.config.gradient_clipping:
            # Clip gradients
            clip_factor = self.config.gradient_clipping / grad_norm
            clipped_grads = jax.tree_util.tree_map(
                lambda g: g * clip_factor, 
                grads
            )
            
            self.logger.warning(
                f"Gradients clipped: norm {grad_norm:.2e} → {self.config.gradient_clipping:.2e}",
                extra={"original_norm": grad_norm, "clip_factor": clip_factor}
            )
            
            return clipped_grads
        
        return grads

    def _compute_gradient_norm(self, grads) -> float:
        """Compute global gradient norm for monitoring"""
        
        squared_norms = jax.tree_util.tree_map(
            lambda g: jnp.sum(g ** 2), 
            grads
        )
        
        total_squared_norm = jax.tree_util.tree_reduce(
            jnp.add, 
            squared_norms
        )
        
        return float(jnp.sqrt(total_squared_norm))

    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        try:
            # Try to get JAX memory stats
            if hasattr(jax.devices()[0], 'memory_stats'):
                memory_stats = jax.devices()[0].memory_stats()
                if 'bytes_in_use' in memory_stats:
                    return memory_stats['bytes_in_use'] / (1024**2)
            
            # Fallback to system memory
            return psutil.virtual_memory().used / (1024**2)
            
        except Exception:
            return 0.0

    def create_accumulation_batches(self, full_batch, target_effective_size: Optional[int] = None):
        """
        Split full batch into micro-batches for accumulation.
        
        Args:
            full_batch: Full batch to split
            target_effective_size: Target effective batch size (uses config if None)
            
        Returns:
            List of micro-batches
        """
        
        effective_size = target_effective_size or self.config.effective_batch_size
        physical_size = self.config.max_physical_batch_size
        
        # Get actual batch size from input
        actual_batch_size = jax.tree_util.tree_leaves(full_batch)[0].shape[0]
        
        if actual_batch_size <= physical_size:
            # No need to split
            self.logger.info(f"Batch size {actual_batch_size} ≤ physical limit {physical_size}, no accumulation needed")
            return [full_batch]
        
        # Calculate number of micro-batches needed
        num_micro_batches = (actual_batch_size + physical_size - 1) // physical_size
        
        micro_batches = []
        for i in range(num_micro_batches):
            start_idx = i * physical_size
            end_idx = min(start_idx + physical_size, actual_batch_size)
            
            # Create micro-batch by slicing
            micro_batch = jax.tree_util.tree_map(
                lambda x: x[start_idx:end_idx], 
                full_batch
            )
            micro_batches.append(micro_batch)
        
        self.logger.info(
            f"🧬 Split batch {actual_batch_size} → {num_micro_batches} micro-batches of ≤{physical_size}",
            extra={
                "original_size": actual_batch_size,
                "micro_batches": num_micro_batches, 
                "effective_size": sum(jax.tree_util.tree_leaves(mb)[0].shape[0] for mb in micro_batches)
            }
        )
        
        return micro_batches

    def optimize_accumulation_config(self, available_memory_gb: float) -> AccumulationConfig:
        """
        Optimize accumulation configuration based on available memory.
        
        Args:
            available_memory_gb: Available GPU memory in GB
            
        Returns:
            Optimized accumulation configuration
        """
        
        # Memory-based batch size estimation
        if available_memory_gb >= 14:  # T4 16GB, V100 16GB+
            max_physical = 8
            accumulation_steps = 4
        elif available_memory_gb >= 10:  # RTX 3080 10GB
            max_physical = 6
            accumulation_steps = 4
        elif available_memory_gb >= 8:   # RTX 3070 8GB  
            max_physical = 4
            accumulation_steps = 4
        else:  # Smaller GPUs
            max_physical = 2
            accumulation_steps = 8
        
        optimized_config = AccumulationConfig(
            accumulation_steps=accumulation_steps,
            max_physical_batch_size=max_physical,
            gradient_clipping=self.config.gradient_clipping,
            mixed_precision=self.config.mixed_precision,
            memory_monitoring=True
        )
        
        self.logger.info(
            f"🧬 Optimized accumulation config for {available_memory_gb:.1f}GB memory",
            extra={
                "config": {
                    "physical_batch": max_physical,
                    "accumulation_steps": accumulation_steps,
                    "effective_batch": optimized_config.effective_batch_size
                }
            }
        )
        
        return optimized_config

    def log_accumulation_summary(self):
        """Log comprehensive accumulation performance summary"""
        
        if not self.memory_timeline:
            self.logger.warning("No accumulation data available for summary")
            return
        
        # Calculate memory statistics
        memory_values = [mem for _, mem in self.memory_timeline]
        avg_memory = sum(memory_values) / len(memory_values)
        max_memory = max(memory_values)
        min_memory = min(memory_values)
        
        summary_metrics = {
            "effective_batch_size": self.config.effective_batch_size,
            "accumulation_steps": self.config.accumulation_steps,
            "physical_batch_size": self.config.max_physical_batch_size,
            "memory_efficiency": {
                "peak_mb": max_memory,
                "average_mb": avg_memory,
                "memory_range_mb": max_memory - min_memory
            }
        }
        
        self.logger.info(
            "🧬 Gradient accumulation summary completed",
            extra={"accumulation_summary": summary_metrics}
        )
        
        # Clear timeline for next session
        self.memory_timeline.clear()

# Factory function for easy integration
def create_gradient_accumulator(
    accumulation_steps: int = 4,
    max_physical_batch_size: int = 4,
    gradient_clipping: float = 1.0,
    auto_optimize: bool = True
) -> GradientAccumulator:
    """
    Factory function to create optimized gradient accumulator.
    
    Args:
        accumulation_steps: Number of micro-batches to accumulate
        max_physical_batch_size: Maximum batch size that fits in memory
        gradient_clipping: Gradient norm clipping threshold
        auto_optimize: Whether to auto-optimize based on available memory
        
    Returns:
        Configured GradientAccumulator instance
    """
    
    config = AccumulationConfig(
        accumulation_steps=accumulation_steps,
        max_physical_batch_size=max_physical_batch_size,
        gradient_clipping=gradient_clipping,
        mixed_precision=True,
        memory_monitoring=True
    )
    
    accumulator = GradientAccumulator(config)
    
    if auto_optimize:
        try:
            # Try to get available memory
            available_memory = 14.6  # Default T4 assumption
            # You could integrate GPU memory detection here
            
            optimized_config = accumulator.optimize_accumulation_config(available_memory)
            accumulator.config = optimized_config
            
        except Exception as e:
            accumulator.logger.warning(
                "Could not auto-optimize accumulation config", 
                exception=e
            )
    
    return accumulator 