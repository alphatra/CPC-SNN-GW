"""
Base argument parser for CPC-SNN-GW CLI.

Extracted from cli.py for better modularity.
"""

import argparse
from pathlib import Path


def get_base_parser():
    """Create base argument parser with common options."""
    parser = argparse.ArgumentParser(add_help=False)
    
    # Global options
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("config.yaml"),
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (use -v, -vv, or -vvv)"
    )
    
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Log file path"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "gpu", "tpu"],
        default="auto",
        help="Device to use for computation"
    )
    
    return parser