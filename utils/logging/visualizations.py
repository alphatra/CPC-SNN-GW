"""
Visualization components for neuromorphic logging.

Extracted from wandb_enhanced_logger.py for better modularity.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from pathlib import Path

# Optional plotting dependencies
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

import jax.numpy as jnp

logger = logging.getLogger(__name__)


class SpikeVisualizer:
    """Visualization for spike patterns and raster plots."""
    
    def __init__(self, enable_wandb: bool = True):
        self.enable_wandb = enable_wandb and HAS_WANDB
        
    def create_raster_plot(self, spikes: np.ndarray, name: str, 
                          step: int = 0) -> Optional[plt.Figure]:
        """Create and log spike raster plot"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Sample subset for visualization (max 100 neurons, 1000 time steps)
            if spikes.shape[0] > 100:
                neuron_indices = np.random.choice(spikes.shape[0], 100, replace=False)
                spikes_sample = spikes[neuron_indices]
            else:
                spikes_sample = spikes
                
            if spikes_sample.shape[-1] > 1000:
                time_indices = np.random.choice(spikes_sample.shape[-1], 1000, replace=False)
                spikes_sample = spikes_sample[..., time_indices]
            
            # Create raster plot
            if len(spikes_sample.shape) == 2:  # [neurons, time]
                spike_times, spike_neurons = np.where(spikes_sample)
                ax.scatter(spike_times, spike_neurons, s=1, alpha=0.7, c='black')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Neuron Index')
                ax.set_title(f'{name} - Spike Raster Plot')
            
            elif len(spikes_sample.shape) == 3:  # [batch, neurons, time]
                # Show first batch
                spike_times, spike_neurons = np.where(spikes_sample[0])
                ax.scatter(spike_times, spike_neurons, s=1, alpha=0.7, c='black')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Neuron Index')
                ax.set_title(f'{name} - Spike Raster Plot (Batch 0)')
            
            plt.tight_layout()
            
            # Log to W&B if enabled
            if self.enable_wandb and wandb.run is not None:
                wandb.log({f"{name}_raster": wandb.Image(fig)}, step=step)
            
            return fig
            
        except Exception as e:
            logger.warning(f"Raster plot creation failed: {e}")
            return None
    
    def log_spike_patterns(self, spikes: jnp.ndarray, name: str = "spike_patterns",
                          step: int = 0, run=None):
        """Log spike patterns with statistics and visualizations"""
        try:
            # Convert to numpy for processing
            spikes_np = np.array(spikes)
            
            # Basic spike statistics
            spike_rate = float(np.mean(spikes_np))
            spike_std = float(np.std(spikes_np))
            spike_count = int(np.sum(spikes_np))
            
            # Log basic stats
            if run is not None:
                run.log({
                    f"{name}/spike_rate": spike_rate,
                    f"{name}/spike_std": spike_std, 
                    f"{name}/spike_count": spike_count,
                    f"{name}/sparsity": 1.0 - spike_rate
                }, step=step)
            
            # Create raster plot occasionally
            if len(spikes_np.shape) >= 2 and step % 25 == 0:  # Every 25 steps
                fig = self.create_raster_plot(spikes_np, name, step)
                if fig is not None:
                    plt.close(fig)
            
            logger.debug(f"🔥 Logged spike patterns: rate={spike_rate:.3f}")
            
            return {
                'spike_rate': spike_rate,
                'spike_count': spike_count,
                'sparsity': 1.0 - spike_rate
            }
            
        except Exception as e:
            logger.warning(f"Spike pattern logging failed: {e}")
            return {}


class GradientVisualizer:
    """Visualization for gradient statistics and distributions."""
    
    def __init__(self, enable_wandb: bool = True):
        self.enable_wandb = enable_wandb and HAS_WANDB
        
    def create_gradient_histogram(self, gradients: Dict[str, jnp.ndarray], 
                                prefix: str, step: int = 0) -> Optional[plt.Figure]:
        """Create gradient distribution histogram"""
        try:
            # Flatten all gradients
            all_grads = []
            for name, grad in gradients.items():
                grad_np = np.array(grad).flatten()
                all_grads.extend(grad_np.tolist())
            
            if not all_grads:
                return None
                
            # Create histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.hist(all_grads, bins=50, alpha=0.7, density=True)
            ax.set_xlabel('Gradient Value')
            ax.set_ylabel('Density')
            ax.set_title(f'{prefix} - Gradient Distribution')
            ax.set_yscale('log')  # Log scale for better visualization
            
            # Add statistics
            mean_grad = np.mean(all_grads)
            std_grad = np.std(all_grads)
            ax.axvline(mean_grad, color='red', linestyle='--', 
                      label=f'Mean: {mean_grad:.2e}')
            ax.axvline(mean_grad + std_grad, color='orange', linestyle='--', alpha=0.7)
            ax.axvline(mean_grad - std_grad, color='orange', linestyle='--', alpha=0.7)
            ax.legend()
            
            plt.tight_layout()
            
            # Log to W&B if enabled
            if self.enable_wandb and wandb.run is not None:
                wandb.log({f"{prefix}_histogram": wandb.Image(fig)}, step=step)
            
            return fig
            
        except Exception as e:
            logger.warning(f"Gradient histogram creation failed: {e}")
            return None
    
    def log_gradient_stats(self, gradients: Dict[str, jnp.ndarray], 
                          prefix: str = "gradients", step: int = 0, run=None):
        """Log gradient statistics with distributions"""
        try:
            gradient_stats = {}
            
            # Flatten all gradients
            all_grads = []
            for name, grad in gradients.items():
                grad_np = np.array(grad).flatten()
                all_grads.extend(grad_np.tolist())
                
                # Per-parameter statistics
                gradient_stats[f"{prefix}/{name}_norm"] = float(np.linalg.norm(grad_np))
                gradient_stats[f"{prefix}/{name}_mean"] = float(np.mean(grad_np))
                gradient_stats[f"{prefix}/{name}_std"] = float(np.std(grad_np))
                gradient_stats[f"{prefix}/{name}_max"] = float(np.max(np.abs(grad_np)))
            
            # Global gradient statistics
            if all_grads:
                gradient_stats[f"{prefix}/global_norm"] = float(np.linalg.norm(all_grads))
                gradient_stats[f"{prefix}/global_mean"] = float(np.mean(all_grads))
                gradient_stats[f"{prefix}/global_std"] = float(np.std(all_grads))
                gradient_stats[f"{prefix}/global_max"] = float(np.max(np.abs(all_grads)))
                
                # Gradient health indicators
                gradient_stats[f"{prefix}/vanishing_ratio"] = float(
                    np.sum(np.abs(all_grads) < 1e-7) / len(all_grads)
                )
                gradient_stats[f"{prefix}/exploding_ratio"] = float(
                    np.sum(np.abs(all_grads) > 1.0) / len(all_grads)
                )
            
            # Log stats
            if run is not None:
                run.log(gradient_stats, step=step)
            
            # Create histogram occasionally
            if step % 50 == 0:  # Every 50 steps
                fig = self.create_gradient_histogram(gradients, prefix, step)
                if fig is not None:
                    plt.close(fig)
            
            logger.debug(f"📊 Logged gradient stats: norm={gradient_stats.get(f'{prefix}/global_norm', 0):.2e}")
            
            return gradient_stats
            
        except Exception as e:
            logger.warning(f"Gradient stats logging failed: {e}")
            return {}


class DashboardCreator:
    """Creates summary dashboards and reports."""
    
    def __init__(self, enable_wandb: bool = True, output_dir: Optional[Path] = None):
        self.enable_wandb = enable_wandb and HAS_WANDB
        self.output_dir = output_dir or Path("./wandb_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
    def create_metrics_summary_plot(self, metrics_history: List[Dict],
                                  step: int = 0) -> Optional[plt.Figure]:
        """Create comprehensive metrics summary plot"""
        if not metrics_history:
            return None
            
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Extract data
            steps = [m.get('step', 0) for m in metrics_history]
            spike_rates = [m.get('spike_rate', 0) for m in metrics_history]
            gradient_norms = [m.get('gradient_norm', 0) for m in metrics_history]
            memory_usage = [m.get('memory_usage_mb', 0) for m in metrics_history]
            latencies = [m.get('inference_latency_ms', 0) for m in metrics_history]
            
            # Plot 1: Spike rates over time
            axes[0, 0].plot(steps, spike_rates, 'b-', alpha=0.7)
            axes[0, 0].set_title('Spike Rate Evolution')
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('Spike Rate (Hz)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Gradient norms
            axes[0, 1].plot(steps, gradient_norms, 'r-', alpha=0.7)
            axes[0, 1].set_title('Gradient Norm Evolution')
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Gradient Norm')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Memory usage
            axes[1, 0].plot(steps, memory_usage, 'g-', alpha=0.7)
            axes[1, 0].set_title('Memory Usage Evolution')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Memory Usage (MB)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Inference latency
            axes[1, 1].plot(steps, latencies, 'm-', alpha=0.7)
            axes[1, 1].set_title('Inference Latency Evolution')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Latency (ms)')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save locally
            summary_path = self.output_dir / f"metrics_summary_step_{step}.png"
            fig.savefig(summary_path, dpi=150, bbox_inches='tight')
            
            # Log to W&B if enabled
            if self.enable_wandb and wandb.run is not None:
                wandb.log({"metrics_summary": wandb.Image(fig)}, step=step)
            
            return fig
            
        except Exception as e:
            logger.warning(f"Metrics summary plot creation failed: {e}")
            return None
    
    def create_summary_dashboard(self, metrics_history: List[Dict], 
                               gradient_history: List[Dict],
                               spike_history: List[Dict],
                               step: int = 0):
        """Create comprehensive summary dashboard"""
        try:
            # Create metrics summary
            metrics_fig = self.create_metrics_summary_plot(metrics_history, step)
            if metrics_fig is not None:
                plt.close(metrics_fig)
            
            logger.info(f"📊 Created summary dashboard at step {step}")
            
        except Exception as e:
            logger.warning(f"Dashboard creation failed: {e}")
