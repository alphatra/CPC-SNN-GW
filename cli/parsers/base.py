"""
Base argument parsers for CLI commands.

This module contains base parser functionality extracted from
cli.py for better modularity.

Split from cli.py for better maintainability.
"""

import argparse
import logging
from pathlib import Path

try:
    from ..._version import __version__
except ImportError:
    try:
        from _version import __version__
    except ImportError:
        __version__ = "0.1.0-dev"

logger = logging.getLogger(__name__)


def get_base_parser() -> argparse.ArgumentParser:
    """Create base argument parser with common options."""
    parser = argparse.ArgumentParser(
        description="CPC+SNN Neuromorphic Gravitational Wave Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"ligo-cpc-snn {__version__}"
    )
    
    parser.add_argument(
        "--config", 
        type=Path,
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="count", 
        default=0,
        help="Increase verbosity level"
    )
    
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Log file path"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Select device backend: auto (default), cpu, or gpu"
    )
    
    return parser


def add_output_args(parser: argparse.ArgumentParser):
    """Add output-related arguments to parser."""
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./outputs"),
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path, 
        default=Path("./data"),
        help="Data directory"
    )


def add_training_args(parser: argparse.ArgumentParser):
    """Add training-specific arguments to parser."""
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--wandb",
        action="store_true", 
        help="Enable Weights & Biases logging"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Resume from checkpoint"
    )


def add_model_args(parser: argparse.ArgumentParser):
    """Add model-specific arguments to parser."""
    # SpikeBridge hyperparameters
    parser.add_argument("--spike-time-steps", type=int, default=24, 
                       help="SpikeBridge time steps T")
    parser.add_argument("--spike-threshold", type=float, default=0.1, 
                       help="Base threshold for encoders")
    parser.add_argument("--spike-learnable", action="store_true", 
                       help="Use learnable multi-threshold encoding")
    parser.add_argument("--no-spike-learnable", dest="spike_learnable", action="store_false", 
                       help="Disable learnable encoding")
    parser.set_defaults(spike_learnable=True)
    parser.add_argument("--spike-threshold-levels", type=int, default=4, 
                       help="Number of threshold levels")
    parser.add_argument("--spike-surrogate-type", type=str, default="adaptive_multi_scale", 
                       help="Surrogate type for spikes")
    parser.add_argument("--spike-surrogate-beta", type=float, default=4.0, 
                       help="Surrogate beta")
    parser.add_argument("--spike-pool-seq", action="store_true", 
                       help="Enable pooling over seq dimension before SNN")
    
    # CPC/Transformer params
    parser.add_argument("--cpc-heads", type=int, default=8, 
                       help="Temporal attention heads")
    parser.add_argument("--cpc-layers", type=int, default=4, 
                       help="Temporal transformer layers")
    
    # SNN params
    parser.add_argument("--snn-hidden", type=int, default=32, 
                       help="SNN hidden size")


def add_data_args(parser: argparse.ArgumentParser):
    """Add data-related arguments to parser."""
    parser.add_argument(
        "--window-size",
        type=int,
        default=512,
        help="Window size for real LIGO dataset windows"
    )
    
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap ratio for windowing (0.0-0.99)"
    )
    
    parser.add_argument(
        "--quick-mode",
        action="store_true",
        help="Use smaller windows for quick testing"
    )
    
    parser.add_argument(
        "--synthetic-quick",
        action="store_true",
        help="Force synthetic quick demo dataset"
    )
    
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=60,
        help="Number of samples for synthetic quick demo dataset"
    )


def add_advanced_args(parser: argparse.ArgumentParser):
    """Add advanced training arguments to parser."""
    # Early stopping and optimization
    parser.add_argument("--balanced-early-stop", action="store_true", 
                       help="Use balanced accuracy/F1 early stopping")
    parser.add_argument("--opt-threshold", action="store_true", 
                       help="Optimize decision threshold by F1/balanced acc")
    
    # PyCBC integration
    parser.add_argument("--use-pycbc", action="store_true",
                       help="Use PyCBC-enhanced synthetic dataset")
    parser.add_argument("--pycbc-psd", type=str, default="aLIGOZeroDetHighPower",
                       help="PyCBC PSD name")
    parser.add_argument("--pycbc-whiten", dest="pycbc_whiten", action="store_true",
                       help="Enable PyCBC time-domain whitening")
    parser.add_argument("--no-pycbc-whiten", dest="pycbc_whiten", action="store_false",
                       help="Disable PyCBC time-domain whitening")
    parser.set_defaults(pycbc_whiten=True)
    
    # PyCBC parameters
    parser.add_argument("--pycbc-snr-min", type=float, default=8.0,
                       help="Minimum target SNR for PyCBC injections")
    parser.add_argument("--pycbc-snr-max", type=float, default=20.0,
                       help="Maximum target SNR for PyCBC injections")
    parser.add_argument("--pycbc-mass-min", type=float, default=10.0,
                       help="Minimum component mass (solar masses)")
    parser.add_argument("--pycbc-mass-max", type=float, default=50.0,
                       help="Maximum component mass (solar masses)")
    
    # MLGWSC integration
    parser.add_argument("--use-mlgwsc", action="store_true",
                       help="Use MLGWSC-1 professional dataset")
    parser.add_argument("--mlgwsc-background-hdf", type=Path,
                       help="Path to MLGWSC background HDF")
    parser.add_argument("--mlgwsc-injections-npy", type=Path, default=None,
                       help="Optional path to MLGWSC injections .npy")
    parser.add_argument("--mlgwsc-slice-seconds", type=float, default=1.25,
                       help="Slice length in seconds for MLGWSC windows")
    parser.add_argument("--mlgwsc-samples", type=int, default=1024,
                       help="Maximum number of MLGWSC samples to load")
    
    # Preprocessing
    parser.add_argument("--whiten-psd", dest="whiten_psd", action="store_true",
                       help="Apply PSD whitening to input signals")
    parser.add_argument("--no-whiten-psd", dest="whiten_psd", action="store_false",
                       help="Disable PSD whitening")
    parser.set_defaults(whiten_psd=False)


def create_training_parser() -> argparse.ArgumentParser:
    """Create complete training argument parser."""
    parser = get_base_parser()
    parser.description = "Train CPC+SNN neuromorphic gravitational wave detector"
    
    # Add all argument groups
    add_output_args(parser)
    add_training_args(parser)
    add_model_args(parser)
    add_data_args(parser)
    add_advanced_args(parser)
    
    # Training mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "enhanced", "advanced", "complete_enhanced"],
        default="complete_enhanced",
        help="Training mode"
    )
    
    return parser


def create_eval_parser() -> argparse.ArgumentParser:
    """Create evaluation argument parser."""
    parser = get_base_parser()
    parser.description = "Evaluate CPC+SNN neuromorphic gravitational wave detector"
    
    add_output_args(parser)
    
    # Evaluation specific arguments
    parser.add_argument(
        "--model-path", "-m",
        type=Path,
        required=True,
        help="Path to trained model"
    )
    
    parser.add_argument(
        "--test-data",
        type=Path,
        help="Test data directory or file"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size"
    )
    
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save model predictions"
    )
    
    return parser


def create_infer_parser() -> argparse.ArgumentParser:
    """Create inference argument parser."""
    parser = get_base_parser()
    parser.description = "Run inference with CPC+SNN neuromorphic gravitational wave detector"
    
    add_output_args(parser)
    
    # Inference specific arguments
    parser.add_argument(
        "--model-path", "-m",
        type=Path,
        required=True,
        help="Path to trained model"
    )
    
    parser.add_argument(
        "--input-data",
        type=Path,
        required=True,
        help="Input data file or directory"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size"
    )
    
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Enable real-time inference mode"
    )
    
    return parser


# Export parser functions
__all__ = [
    "get_base_parser",
    "add_output_args",
    "add_training_args", 
    "add_model_args",
    "add_data_args",
    "add_advanced_args",
    "create_training_parser",
    "create_eval_parser",
    "create_infer_parser"
]
