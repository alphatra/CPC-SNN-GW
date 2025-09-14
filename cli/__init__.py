"""
Modular CLI package for CPC-SNN-GW.

This module provides a clean, modular command-line interface with backward compatibility.
"""

# Backward compatibility imports
from .commands.train import train_cmd
from .commands.evaluate import eval_cmd  
from .commands.inference import infer_cmd
from .main import main

__all__ = [
    'train_cmd',
    'eval_cmd', 
    'infer_cmd',
    'main'
]