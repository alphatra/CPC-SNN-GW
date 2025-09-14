"""
Training command implementation.

Extracted from cli.py for better modularity.
"""

import logging
from pathlib import Path

from ..parsers.base import get_base_parser

logger = logging.getLogger(__name__)


def train_cmd():
    """Main training command entry point."""
    parser = get_base_parser()
    parser.description = "Train CPC+SNN neuromorphic gravitational wave detector"
    
    # Training specific arguments
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./outputs"),
        help="Output directory for training artifacts"
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path, 
        default=Path("./data"),
        help="Data directory"
    )
    
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
    
    # SpikeBridge hyperparameters via CLI
    parser.add_argument("--spike-time-steps", type=int, default=24, 
                       help="SpikeBridge time steps T")
    parser.add_argument("--spike-threshold", type=float, default=0.1, 
                       help="Base threshold for encoders")
    parser.add_argument("--spike-learnable", action="store_true", 
                       help="Use learnable multi-threshold encoding")
    parser.add_argument("--no-spike-learnable", dest="spike_learnable", 
                       action="store_false", help="Disable learnable encoding")
    parser.set_defaults(spike_learnable=True)
    parser.add_argument("--spike-threshold-levels", type=int, default=4, 
                       help="Number of threshold levels")
    parser.add_argument("--spike-surrogate-type", type=str, default="adaptive_multi_scale", 
                       help="Surrogate type for spikes")
    parser.add_argument("--spike-surrogate-beta", type=float, default=4.0, 
                       help="Surrogate beta")
    
    # CPC/Transformer params
    parser.add_argument("--cpc-heads", type=int, default=8, 
                       help="Temporal attention heads")
    parser.add_argument("--cpc-layers", type=int, default=4, 
                       help="Temporal transformer layers")
    
    # SNN params
    parser.add_argument("--snn-hidden", type=int, default=32, 
                       help="SNN hidden size")
    
    # Early stop and thresholding
    parser.add_argument("--balanced-early-stop", action="store_true", 
                       help="Use balanced accuracy/F1 early stopping")
    parser.add_argument("--opt-threshold", action="store_true", 
                       help="Optimize decision threshold by F1/balanced acc")
    
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    
    args = parser.parse_args()
    
    # Import training runners
    from ..runners.standard import run_standard_training
    from ..runners.enhanced import run_enhanced_training
    
    # Setup logging
    from ...utils.setup_logging import setup_logging
    setup_logging(
        level=logging.INFO if args.verbose == 0 else logging.DEBUG,
        log_file=args.log_file
    )
    
    logger.info("🚀 Starting CPC+SNN training")
    
    # Load configuration
    from ...utils.config import load_config
    config = load_config(args.config)
    
    try:
        # Determine training mode
        training_mode = config.get('training', {}).get('mode', 'standard')
        
        if training_mode == 'enhanced':
            return run_enhanced_training(config, args)
        else:
            return run_standard_training(config, args)
            
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1