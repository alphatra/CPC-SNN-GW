"""
Evaluation command implementation.

Extracted from cli.py for better modularity.
"""

import logging
from pathlib import Path

from ..parsers.base import get_base_parser

logger = logging.getLogger(__name__)


def eval_cmd():
    """Main evaluation command entry point."""
    parser = get_base_parser()
    parser.description = "Evaluate CPC+SNN neuromorphic gravitational wave detector"
    
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
        "--output-dir", "-o", 
        type=Path,
        default=Path("./evaluation"),
        help="Output directory for evaluation results"
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
    
    args = parser.parse_args()
    
    # Setup logging
    from ...utils.setup_logging import setup_logging
    setup_logging(
        level=logging.INFO if args.verbose == 0 else logging.DEBUG,
        log_file=args.log_file
    )
    
    logger.info("🔍 Starting CPC+SNN evaluation")
    logger.info(f"   Model: {args.model_path}")
    logger.info(f"   Output: {args.output_dir}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    from ...utils.config import load_config
    config = load_config(args.config)
    
    try:
        # Load trained model parameters
        logger.info("📂 Loading trained model parameters...")
        if not args.model_path.exists():
            logger.error(f"❌ Model path does not exist: {args.model_path}")
            return 1
            
        # Load or generate test data
        logger.info("📊 Loading test data...")
        
        # Run the evaluation pipeline
        logger.info("🔍 Running evaluation pipeline...")
        
        # Import evaluation modules
        from ...training.test_evaluation import evaluate_on_test_set
        from ...training.unified_trainer import create_unified_trainer, UnifiedTrainingConfig
        
        # Create trainer with same config
        trainer_config = UnifiedTrainingConfig(
            cpc_latent_dim=config['model']['cpc_latent_dim'],
            snn_hidden_size=config['model']['snn_layer_sizes'][0],
            num_classes=3,  # continuous_gw, binary_merger, noise_only
            random_seed=42  # Reproducible evaluation
        )
        
        trainer = create_unified_trainer(trainer_config)
        
        # Run evaluation
        # Load MLGWSC-1 test data
        from ...data.mlgwsc_data_loader import MLGWSCDataLoader
        from ...utils.config_loader import load_config
        
        config = load_config()
        data_loader = MLGWSCDataLoader(
            mode="validation",
            config=config
        )
        logger.info("✅ MLGWSC-1 data loader initialized")
        
        # Create labeled test dataset
        test_segments, test_labels = data_loader.create_labeled_dataset()
        logger.info(f"📊 Loaded {len(test_segments)} test segments")
        logger.info(f"   - Background: {test_labels.count(0)} segments")  
        logger.info(f"   - Signal: {test_labels.count(1)} segments")
        
        test_results = evaluate_on_test_set(
            trainer=trainer,
            test_signals=test_segments,
            test_labels=test_labels,
            verbose=True,
            batch_size=args.batch_size
        )
        
        logger.info("✅ Evaluation completed successfully")
        logger.info(f"   Accuracy: {test_results.get('accuracy', 0.0):.3f}")
        logger.info(f"   F1 Score: {test_results.get('f1_score', 0.0):.3f}")
        
        # Save results
        results_file = args.output_dir / "evaluation_results.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"📊 Results saved to: {results_file}")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
