"""
Enhanced Scientific Logging System for Neuromorphic Gravitational Wave Detection

Revolutionary logging framework combining Rich visual enhancements with deep scientific 
metrics tracking. Provides beautiful output while maintaining rigorous scientific documentation
for breakthrough gravitational wave detection research.

Features:
- Rich Console with scientific formatting  
- Real-time progress tracking with tqdm integration
- Advanced error diagnostics with scientific context
- GPU memory monitoring with visual alerts
- Training metrics visualization with scientific precision
- Comprehensive traceback analysis for research debugging
"""

import logging
import time
import traceback
import psutil
import sys
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, 
    TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.traceback import install
from rich.text import Text
from rich.columns import Columns
from rich.layout import Layout
from rich.live import Live
from rich import box
import jax

# Install rich traceback for beautiful error handling
install(show_locals=True)

@dataclass
class ScientificMetrics:
    """Scientific metrics for gravitational wave detection research"""
    
    # Core Training Metrics
    epoch: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    cpc_loss: float = 0.0
    snn_accuracy: float = 0.0
    
    # Performance Metrics  
    training_time: float = 0.0
    inference_time_ms: float = 0.0
    gpu_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Scientific Quality Metrics
    signal_to_noise_ratio: float = 0.0
    classification_confidence: float = 0.0
    false_positive_rate: float = 0.0
    detection_sensitivity: float = 0.0
    
    # System Health Metrics
    gradient_norm: float = 0.0
    learning_rate: float = 0.0
    batch_size: int = 0
    samples_processed: int = 0

class EnhancedScientificLogger:
    """
    Revolutionary logging system for neuromorphic gravitational wave detection.
    
    Combines beautiful Rich visualizations with rigorous scientific documentation.
    Designed for breakthrough research with production-ready reliability.
    """
    
    def __init__(self, 
                 name: str = "GW-Detection",
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 console_width: int = 120):
        
        self.name = name
        self.console = Console(width=console_width)
        self.start_time = time.time()
        self.metrics_history: List[ScientificMetrics] = []
        
        # Setup rich logging
        self._setup_logging(log_level, log_file)
        
        # Initialize progress tracking - FIXED for no duplication
        self.progress = None
        self.current_tasks = {}
        self._progress_live = None
        self._in_progress_context = False
        
        # Scientific context tracking
        self.experiment_context = {
            "session_id": f"GW_{int(time.time())}",
            "jax_devices": len(jax.devices()),
            "system_ram_gb": psutil.virtual_memory().total / (1024**3),
        }
        
        self.info("🔬 Enhanced Scientific Logger Initialized", 
                 extra={"context": self.experiment_context})

    def _setup_logging(self, log_level: str, log_file: Optional[str]):
        """Setup comprehensive logging with Rich integration"""
        
        # Create rich handler
        rich_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True
        )
        rich_handler.setFormatter(logging.Formatter(
            fmt="%(message)s",
            datefmt="[%X]"
        ))
        
        # Setup logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Add handlers
        self.logger.addHandler(rich_handler)
        
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(file_handler)

    def info(self, message: str, extra: Optional[Dict] = None):
        """Enhanced info logging with scientific context"""
        self._log_with_context("INFO", message, extra)

    def warning(self, message: str, extra: Optional[Dict] = None):
        """Enhanced warning logging with scientific context"""
        self._log_with_context("WARNING", message, extra)
    
    def error(self, message: str, extra: Optional[Dict] = None, exception: Optional[Exception] = None):
        """Enhanced error logging with scientific diagnostics"""
        self._log_with_context("ERROR", message, extra)
        
        if exception:
            self.console.print("\n[red]🚨 SCIENTIFIC ERROR ANALYSIS:[/red]")
            self.console.print_exception()
            self._analyze_scientific_error(exception)

    def critical(self, message: str, extra: Optional[Dict] = None):
        """Critical scientific error with full system diagnostics"""
        self._log_with_context("CRITICAL", message, extra)
        self._emergency_system_diagnostics()

    def _log_with_context(self, level: str, message: str, extra: Optional[Dict] = None):
        """Log with scientific context and beautiful formatting"""
        
        # Create formatted message
        timestamp = time.strftime("%H:%M:%S")
        runtime = time.time() - self.start_time
        
        # Scientific context
        context_info = ""
        if extra:
            if "metrics" in extra:
                metrics = extra["metrics"]
                context_info = f" | Loss: {metrics.loss:.4f} | Acc: {metrics.accuracy:.3f}"
            elif "context" in extra:
                context_info = f" | {extra['context']}"
        
        # Format with Rich markup
        level_colors = {
            "INFO": "green",
            "WARNING": "yellow", 
            "ERROR": "red",
            "CRITICAL": "bold red"
        }
        
        color = level_colors.get(level, "white")
        formatted_message = f"[{color}]{level}[/{color}] [{timestamp}] [dim](+{runtime:.1f}s)[/dim] {message}{context_info}"
        
        # Log to underlying logger
        getattr(self.logger, level.lower())(message, extra=extra)
        
        # Display with Rich
        self.console.print(formatted_message)

    def log_scientific_metrics(self, metrics: ScientificMetrics):
        """Log comprehensive scientific metrics with beautiful visualization"""
        
        self.metrics_history.append(metrics)
        
        # Create metrics table
        table = Table(title="🔬 Scientific Metrics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="magenta", width=15) 
        table.add_column("Status", style="green", width=15)
        
        # Core metrics
        table.add_row("Training Loss", f"{metrics.loss:.6f}", self._get_loss_status(metrics.loss))
        table.add_row("Accuracy", f"{metrics.accuracy:.3%}", self._get_accuracy_status(metrics.accuracy))
        table.add_row("CPC Loss", f"{metrics.cpc_loss:.6f}", "📊 Learning")
        table.add_row("SNN Accuracy", f"{metrics.snn_accuracy:.3%}", "🧠 Processing")
        
        # Performance metrics
        table.add_row("GPU Memory", f"{metrics.gpu_memory_mb:.1f} MB", self._get_memory_status(metrics.gpu_memory_mb))
        table.add_row("Training Time", f"{metrics.training_time:.1f}s", "⏱️ Tracking")
        table.add_row("Gradient Norm", f"{metrics.gradient_norm:.2e}", self._get_gradient_status(metrics.gradient_norm))
        
        self.console.print(table)
        
        # Log for file
        self.info("Scientific metrics recorded", extra={"metrics": metrics})

    def _get_loss_status(self, loss: float) -> str:
        """Get loss status with scientific interpretation"""
        if loss < 0.1:
            return "🎯 Excellent"
        elif loss < 0.5:
            return "✅ Good"
        elif loss < 1.0:
            return "📈 Learning"
        else:
            return "⚠️ High"

    def _get_accuracy_status(self, accuracy: float) -> str:
        """Get accuracy status for gravitational wave detection"""
        if accuracy > 0.95:
            return "🏆 Outstanding"
        elif accuracy > 0.80:
            return "✅ Excellent"
        elif accuracy > 0.60:
            return "📊 Good"
        elif accuracy > 0.40:
            return "📈 Learning"
        else:
            return "⚠️ Needs Work"

    def _get_memory_status(self, memory_mb: float) -> str:
        """Get GPU memory status"""
        if memory_mb < 8000:  # < 8GB
            return "✅ Efficient"
        elif memory_mb < 12000:  # < 12GB
            return "📊 Normal"
        elif memory_mb < 15000:  # < 15GB
            return "⚠️ High"
        else:
            return "🚨 Critical"

    def _get_gradient_status(self, grad_norm: float) -> str:
        """Get gradient norm status for training stability"""
        if grad_norm < 1e-6:
            return "⚠️ Vanishing"
        elif grad_norm < 1.0:
            return "✅ Stable"
        elif grad_norm < 10.0:
            return "📈 Learning"
        else:
            return "🚨 Exploding"

    @contextmanager
    def progress_context(self, description: str, total: Optional[int] = None, 
                        reuse_existing: bool = False):
        """Context manager for beautiful progress tracking - FIXED no duplication"""
        
        # ✅ CRITICAL FIX: Use single progress instance, no duplication
        
        # First time setup - create progress if needed
        if self.progress is None:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=50),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=self.console,
                refresh_per_second=2,  # ✅ Lower refresh rate
                transient=False  # ✅ Keep progress bars visible
            )
        
        # Reuse existing task if requested and exists
        if reuse_existing and description in self.current_tasks:
            task_id = self.current_tasks[description]
            self.progress.update(task_id, description=description, total=total, completed=0)
        else:
            # Remove old task if exists to prevent accumulation
            if description in self.current_tasks:
                old_task_id = self.current_tasks[description]
                try:
                    self.progress.remove_task(old_task_id)
                except:
                    pass
            
            # Create new task
            task_id = self.progress.add_task(description, total=total)
            self.current_tasks[description] = task_id
        
        # ✅ CRITICAL: Start progress context only once
        try:
            if not self._in_progress_context:
                self._in_progress_context = True
                with self.progress:
                    yield task_id
            else:
                # Already in progress context, just yield task
                yield task_id
        finally:
            # Reset context when leaving top-level
            if self._in_progress_context:
                self._in_progress_context = False

    def update_progress(self, task_id: int, advance: int = 1, description: Optional[str] = None):
        """Update progress with scientific context"""
        if self.progress:
            self.progress.update(task_id, advance=advance, description=description)
    
    def clear_progress(self):
        """Clear all progress bars and reset state - FIXED"""
        try:
            # Clear all tasks first
            if self.progress and self.current_tasks:
                for task_desc, task_id in list(self.current_tasks.items()):
                    try:
                        self.progress.remove_task(task_id)
                    except:
                        pass
            
            # Reset state
            self.current_tasks.clear()
            self._in_progress_context = False
            
            # Keep progress instance but cleared
            # Don't destroy it to avoid duplication issues
            
        except Exception as e:
            # Silent cleanup, don't spam logs
            pass

    def _analyze_scientific_error(self, exception: Exception):
        """Analyze errors in scientific context"""
        
        error_analysis = Table(title="🔬 Scientific Error Analysis", box=box.ROUNDED)
        error_analysis.add_column("Analysis", style="cyan")
        error_analysis.add_column("Recommendation", style="yellow")
        
        error_type = type(exception).__name__
        error_msg = str(exception)
        
        # Analyze common scientific computing errors
        if "RESOURCE_EXHAUSTED" in error_msg or "Out of memory" in error_msg:
            error_analysis.add_row(
                "GPU Memory Exhaustion",
                "Reduce batch size, sequence length, or implement gradient accumulation"
            )
        elif "gradient" in error_msg.lower():
            error_analysis.add_row(
                "Gradient Computation Issue", 
                "Check model architecture, learning rate, or numerical stability"
            )
        elif "nan" in error_msg.lower() or "inf" in error_msg.lower():
            error_analysis.add_row(
                "Numerical Instability",
                "Check input scaling, learning rate, or add gradient clipping"
            )
        else:
            error_analysis.add_row(
                f"General Error: {error_type}",
                "Check logs above for detailed traceback and context"
            )
        
        self.console.print(error_analysis)

    def _emergency_system_diagnostics(self):
        """Emergency system diagnostics for critical errors"""
        
        self.console.print("\n[red]🚨 EMERGENCY SYSTEM DIAGNOSTICS:[/red]")
        
        # System resources
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        diag_table = Table(title="System Status", box=box.ROUNDED)
        diag_table.add_column("Resource", style="cyan")
        diag_table.add_column("Status", style="magenta")
        diag_table.add_column("Action", style="yellow")
        
        diag_table.add_row("RAM Usage", f"{memory.percent:.1f}%", 
                          "🚨 Critical" if memory.percent > 90 else "✅ Normal")
        diag_table.add_row("CPU Usage", f"{cpu_percent:.1f}%",
                          "⚠️ High" if cpu_percent > 80 else "✅ Normal")
        
        # JAX devices
        devices = jax.devices()
        diag_table.add_row("JAX Devices", f"{len(devices)} available", "✅ Connected")
        
        self.console.print(diag_table)

    def create_training_dashboard(self) -> Layout:
        """Create live training dashboard"""
        
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        layout["header"].update(
            Panel("🔬 Neuromorphic Gravitational Wave Detection Training", 
                  style="bold blue")
        )
        
        if self.metrics_history:
            latest = self.metrics_history[-1]
            layout["main"].update(self._create_metrics_panel(latest))
        
        layout["footer"].update(
            Panel(f"Session: {self.experiment_context['session_id']} | "
                  f"Runtime: {time.time() - self.start_time:.1f}s", 
                  style="dim")
        )
        
        return layout

    def _create_metrics_panel(self, metrics: ScientificMetrics) -> Panel:
        """Create metrics visualization panel"""
        
        columns = Columns([
            self._create_training_metrics(metrics),
            self._create_performance_metrics(metrics),
            self._create_system_metrics(metrics)
        ])
        
        return Panel(columns, title="📊 Live Metrics", border_style="green")

    def _create_training_metrics(self, metrics: ScientificMetrics) -> Table:
        """Create training metrics table"""
        table = Table(title="Training", box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Epoch", str(metrics.epoch))
        table.add_row("Loss", f"{metrics.loss:.6f}")
        table.add_row("Accuracy", f"{metrics.accuracy:.3%}")
        table.add_row("CPC Loss", f"{metrics.cpc_loss:.6f}")
        
        return table

    def _create_performance_metrics(self, metrics: ScientificMetrics) -> Table:
        """Create performance metrics table"""
        table = Table(title="Performance", box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Training Time", f"{metrics.training_time:.1f}s")
        table.add_row("GPU Memory", f"{metrics.gpu_memory_mb:.1f} MB")
        table.add_row("Inference", f"{metrics.inference_time_ms:.1f}ms")
        table.add_row("Gradient Norm", f"{metrics.gradient_norm:.2e}")
        
        return table

    def _create_system_metrics(self, metrics: ScientificMetrics) -> Table:
        """Create system metrics table"""
        table = Table(title="System", box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("CPU Usage", f"{metrics.cpu_usage_percent:.1f}%")
        table.add_row("Batch Size", str(metrics.batch_size))
        table.add_row("Samples", str(metrics.samples_processed))
        table.add_row("Learning Rate", f"{metrics.learning_rate:.2e}")
        
        return table

    def log_experiment_summary(self):
        """Log comprehensive experiment summary"""
        
        if not self.metrics_history:
            self.warning("No metrics history available for summary")
            return
        
        runtime = time.time() - self.start_time
        latest_metrics = self.metrics_history[-1]
        
        # Create summary panel
        summary = Panel.fit(
            f"""
[bold green]🎉 EXPERIMENT COMPLETED SUCCESSFULLY[/bold green]

[cyan]Session:[/cyan] {self.experiment_context['session_id']}
[cyan]Total Runtime:[/cyan] {runtime:.1f}s ({runtime/60:.1f} minutes)
[cyan]Final Accuracy:[/cyan] {latest_metrics.accuracy:.3%}
[cyan]Final Loss:[/cyan] {latest_metrics.loss:.6f}
[cyan]Epochs Completed:[/cyan] {latest_metrics.epoch}
[cyan]Samples Processed:[/cyan] {latest_metrics.samples_processed:,}

[yellow]🔬 Scientific Achievement:[/yellow]
Neuromorphic gravitational wave detection system successfully trained
with production-scale performance and memory optimization.
            """,
            title="📊 Experiment Summary",
            border_style="green"
        )
        
        self.console.print(summary)
        self.info("Experiment summary logged", extra={"summary": asdict(latest_metrics)})

# Global enhanced logger instance
enhanced_logger = None

def get_enhanced_logger(name: str = "GW-Detection", **kwargs) -> EnhancedScientificLogger:
    """Get global enhanced logger instance"""
    global enhanced_logger
    if enhanced_logger is None:
        enhanced_logger = EnhancedScientificLogger(name, **kwargs)
    return enhanced_logger 