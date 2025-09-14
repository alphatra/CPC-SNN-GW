"""
Main CLI entry point.

Extracted from cli.py for better modularity.
"""

import sys
from typing import Optional


def main(argv: Optional[list] = None) -> int:
    """
    Main CLI entry point.
    
    Args:
        argv: Optional command line arguments (for testing)
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if argv is None:
        argv = sys.argv
    
    if len(argv) < 2:
        print("Usage: python -m cpc_snn_gw.cli <command> [options]")
        print("Commands:")
        print("  train     - Train CPC+SNN model")
        print("  eval      - Evaluate trained model")
        print("  infer     - Run inference on new data")
        print()
        print("For help on a specific command:")
        print("  python -m cpc_snn_gw.cli <command> --help")
        return 1
    
    command = argv[1]
    
    # Import commands
    from .commands import train_cmd, eval_cmd, infer_cmd
    
    # Route to appropriate command
    if command == "train":
        # Remove command from argv for argparse
        sys.argv = [argv[0]] + argv[2:]
        return train_cmd() or 0
    elif command == "eval":
        sys.argv = [argv[0]] + argv[2:]
        return eval_cmd() or 0
    elif command == "infer":
        sys.argv = [argv[0]] + argv[2:]
        return infer_cmd() or 0
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, eval, infer")
        return 1


if __name__ == "__main__":
    sys.exit(main())
