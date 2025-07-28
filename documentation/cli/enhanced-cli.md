# Enhanced CLI

## Overview

The `enhanced_cli.py` script is an advanced command-line interface that builds upon the functionality of the main `cli.py`. It is designed for power users and researchers who require more control and detailed feedback during the training and evaluation process.


The enhanced CLI provides several key improvements over the main CLI:

1.  **Enhanced Logging**: It uses the `rich` and `tqdm` libraries to provide a more visually appealing and informative console output, including progress bars and formatted tables.
2.  **Gradient Accumulation**: It explicitly supports and manages gradient accumulation, allowing users to fine-tune this critical hyperparameter.
3.  **CPC Metrics**: It provides real-time monitoring of the CPC loss during training, which is essential for diagnosing issues with the self-supervised learning phase.
4.  **Advanced Configuration**: It offers more granular control over the training process through additional command-line arguments.

Like the main CLI, it automatically applies the 6-stage GPU warmup at startup.


## Commands and Options

The enhanced CLI supports the same core commands (`train`, `eval`, `infer`) as the main CLI but with additional options for advanced control.


### `train` Command (Enhanced)
This command provides detailed monitoring of the training process.

**Additional Key Options**:
*   `--use-gradient-accumulation`: A flag to enable gradient accumulation. Default: `True`.
*   `--accumulation-steps`: The number of steps to accumulate gradients over. Default: `4`.
*   `--log-cpc-loss`: A flag to enable logging of the CPC loss during training. Default: `True`.
*   `--verbose`: A flag for even more detailed output.

**Example**:
```bash
# Train with gradient accumulation and CPC loss logging
python enhanced_cli.py --mode train --use-gradient-accumulation --accumulation-steps 8 --log-cpc-loss
```

### `eval` Command (Enhanced)
This command provides a more detailed evaluation report, including a breakdown of the confusion matrix and a list of any suspicious patterns detected.

**Additional Key Options**:
*   `--detailed-report`: A flag to include a full breakdown of all metrics and diagnostic flags in the output.

## Implementation

The enhanced CLI shares much of its core logic with the main CLI but uses a more sophisticated logging and progress tracking system.

```python
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table
import time

# Initialize a rich console for formatted output
console = Console()

def main():
    parser = argparse.ArgumentParser(description='Enhanced CPC-SNN-GW CLI')
    # ... (same as main CLI, plus additional args) ...
    parser.add_argument('--use-gradient-accumulation', action='store_true', default=True)
    parser.add_argument('--accumulation-steps', type=int, default=4)
    parser.add_argument('--log-cpc-loss', action='store_true', default=True)
    parser.add_argument('--detailed-report', action='store_true', default=False)
    
    args = parser.parse_args()
    
    # ... (load config, GPU warmup, load data) ...
    
    if args.mode == 'train':
        # Create a rich progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task("[cyan]Training", total=args.epochs * len(train_signals))
            
            # Initialize the trainer with accumulation settings
            trainer = UnifiedTrainer(
                # ... (model components) ...
                # The trainer will use args.accumulation_steps
            )
            
            # Training loop with progress updates
            for epoch in range(args.epochs):
                epoch_loss = 0.0
                for i, (signal, label) in enumerate(zip(train_signals, train_labels)):
                    # ... (training step) ...
                    
                    # Update the progress bar
                    progress.update(task, advance=1, description=f"Epoch {epoch+1}/{args.epochs}")
                    
                    # Log CPC loss if requested
                    if args.log_cpc_loss and 'cpc_loss' in metrics:
                        console.print(f"CPC Loss: {metrics['cpc_loss']:.6f}", style="yellow")
            
            console.print("✅ Training complete!", style="bold green")
        
    elif args.mode == 'eval':
        # ... (evaluation logic) ...
        
        # Create a rich table for the results
        table = Table(title="Test Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="magenta")
        
        table.add_row("Test Accuracy", f"{test_results['test_accuracy']:.4f}")
        table.add_row("Sensitivity", f"{test_results['sensitivity']:.4f}")
        table.add_row("Specificity", f"{test_results['specificity']:.4f}")
        table.add_row("F1-Score", f"{test_results['f1_score']:.4f}")
        table.add_row("Model Collapse", str(test_results['model_collapse']))
        
        console.print(table)
        
        if args.detailed_report:
            console.print("\n🔍 Detailed Quality Report:")
            for pattern in test_results['suspicious_patterns']:
                console.print(f"  - [red]{pattern}[/red]")

if __name__ == '__main__':
    main()
```

## Usage

The enhanced CLI is ideal for monitoring the training process in detail and for debugging.

```bash
# Monitor training with a progress bar and CPC loss
python enhanced_cli.py --mode train --log-cpc-loss --accumulation-steps 4

# Get a detailed evaluation report
python enhanced_cli.py --mode eval --detailed-report
```

This tool provides a powerful interface for advanced users to gain deep insights into the model's behavior and performance.