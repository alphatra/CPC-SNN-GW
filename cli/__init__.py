"""
CLI Module: Command Line Interface Components

Modular implementation of CLI functionality split from 
cli.py for better maintainability.

Components:
- commands: Command implementations (train, eval, infer)
- parsers: Argument parsers and validation
- utils: CLI utilities and helpers
- main: Main entry point
"""

from .commands import train_cmd, eval_cmd, infer_cmd
from .parsers import get_base_parser, get_train_parser
from .utils import perform_gpu_warmup, load_cli_config, save_cli_config
from .main import main

__all__ = [
    # Commands
    "train_cmd",
    "eval_cmd", 
    "infer_cmd",
    
    # Parsers
    "get_base_parser",
    "get_train_parser",
    
    # Utilities
    "perform_gpu_warmup",
    "load_cli_config",
    "save_cli_config",
    
    # Main entry
    "main"
]
