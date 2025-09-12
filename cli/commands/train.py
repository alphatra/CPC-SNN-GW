"""
Training Command (MODULAR)

This file delegates to modular training components for better maintainability.
The actual implementation has been split into:
- training/initializer.py: Environment setup
- training/data_loader.py: Data loading strategies
- training/standard.py: Standard training implementation
- training/enhanced.py: Enhanced training implementations

This file maintains CLI interface through delegation.
"""

import logging

from .training import (
    setup_training_environment,
    load_training_data,
    run_standard_training,
    run_enhanced_training,
    run_complete_enhanced_training
)
from ..parsers.base import create_training_parser

logger = logging.getLogger(__name__)


def train_cmd():
    """Main training command entry point."""
    parser = create_training_parser()
    args = parser.parse_args()
    
    # Setup logging
    from utils import setup_logging
    setup_logging(
        level=logging.INFO if args.verbose == 0 else logging.DEBUG,
        log_file=args.log_file
    )
    
    logger.info(f"🚀 Starting CPC+SNN training")
    logger.info(f"   Output directory: {args.output_dir}")
    logger.info(f"   Mode: {args.mode}")
    
    # Load configuration
    from utils.config import load_config, save_config
    config = load_config(args.config)
    
    # Update config with CLI args
    _update_config_from_args(config, args)
    
    # Create output directory and save config
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.yaml"
    save_config(config, config_path)
    
    try:
        logger.info(f"🎯 Starting {args.mode} training mode...")
        
        # Route to appropriate training implementation
        if args.mode == "standard":
            training_result = run_standard_training(config, args)
        elif args.mode == "enhanced":
            training_result = run_enhanced_training(config, args)
        elif args.mode == "complete_enhanced":
            training_result = run_complete_enhanced_training(config, args)
        else:
            raise ValueError(f"Unknown training mode: {args.mode}")
        
        # Check results
        if training_result and training_result.get('success', False):
            logger.info("✅ Training completed successfully!")
            logger.info(f"📊 Final metrics: {training_result.get('metrics', {})}")
            return 0
        else:
            logger.error("❌ Training failed!")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        return 1


def _update_config_from_args(config: Dict, args):
    """Update configuration with CLI arguments."""
    # Basic parameter overrides
    if args.output_dir:
        config.setdefault('logging', {})
        config['logging']['checkpoint_dir'] = str(args.output_dir)
    
    if args.epochs is not None:
        config.setdefault('training', {})
        config['training']['cpc_epochs'] = args.epochs
    
    if args.batch_size is not None:
        config.setdefault('training', {})
        config['training']['batch_size'] = args.batch_size
    
    if args.learning_rate is not None:
        config.setdefault('training', {})
        config['training']['cpc_lr'] = args.learning_rate
    
    # Device configuration
    if args.device and args.device != 'auto':
        config.setdefault('platform', {})
        config['platform']['device'] = args.device
    
    # Logging configuration
    if args.wandb:
        config.setdefault('logging', {})
        config['logging']['wandb_project'] = "cpc-snn-training"


# Export training command
__all__ = [
    "train_cmd"
]