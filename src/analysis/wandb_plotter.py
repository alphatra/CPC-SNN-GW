import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

class WandBPlotter:
    def __init__(self, entity: str, project: str):
        self.api = wandb.Api()
        self.entity = entity
        self.project = project
        self.path = f"{entity}/{project}"
        
        # Set style
        sns.set_theme(style="darkgrid")
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['font.size'] = 12
        
    def get_runs(self, limit: int = 10) -> List["wandb.apis.public.Run"]:
        """Fetches recent runs."""
        runs = self.api.runs(self.path, order="-created_at")
        return [r for i, r in enumerate(runs) if i < limit]
        
    def plot_run_metrics(self, run_id: str, metrics: List[str] = None, smooth: float = 0.0):
        """Plots specific metrics for a single run."""
        run = self.api.run(f"{self.path}/{run_id}")
        history = run.history()
        
        if metrics is None:
            metrics = ["train_loss", "val_loss", "train_acc", "val_acc"]
            
        # Filter metrics that exist in history
        valid_metrics = [m for m in metrics if m in history.columns]
        
        if not valid_metrics:
            print(f"No valid metrics found for run {run_id}")
            return
            
        # Plot
        fig, axes = plt.subplots(len(valid_metrics), 1, figsize=(12, 4 * len(valid_metrics)), sharex=True)
        if len(valid_metrics) == 1:
            axes = [axes]
            
        for ax, metric in zip(axes, valid_metrics):
            data = history[metric]
            if smooth > 0:
                data = data.ewm(alpha=1-smooth).mean()
                
            sns.lineplot(x=history.index, y=data, ax=ax, label=metric, linewidth=2)
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} (Run: {run.name})")
            
        plt.xlabel("Step/Epoch")
        plt.tight_layout()
        plt.show()
        
    def compare_runs(self, run_ids: List[str], metric: str, smooth: float = 0.0):
        """Compares a specific metric across multiple runs."""
        plt.figure(figsize=(12, 6))
        
        for run_id in run_ids:
            try:
                run = self.api.run(f"{self.path}/{run_id}")
                history = run.history()
                
                if metric in history.columns:
                    data = history[metric]
                    if smooth > 0:
                        data = data.ewm(alpha=1-smooth).mean()
                    sns.lineplot(x=history.index, y=data, label=f"{run.name} ({run_id})")
            except Exception as e:
                print(f"Error fetching run {run_id}: {e}")
                
        plt.title(f"Comparison: {metric}")
        plt.ylabel(metric)
        plt.xlabel("Step")
        plt.legend()
        plt.show()
        
    def plot_advanced_metrics(self, run_id: str):
        """Plots SNN specific metrics."""
        metrics = [
            "snn_spike_density", 
            "rsnn_context_mean", 
            "rsnn_context_std", 
            "input_rms",
            "grad_norm"
        ]
        self.plot_run_metrics(run_id, metrics)

if __name__ == "__main__":
    # Example usage
    plotter = WandBPlotter("alpphatra", "cpc-snn-gw")
    runs = plotter.get_runs(5)
    print("Recent runs:")
    for r in runs:
        print(f"- {r.name} ({r.id}) [{r.state}]")
