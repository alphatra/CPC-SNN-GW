"""
Training runners and implementations.
"""

from .standard import run_standard_training
from .enhanced import run_enhanced_training

__all__ = [
    'run_standard_training',
    'run_enhanced_training'
]
