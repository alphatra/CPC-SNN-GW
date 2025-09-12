"""
Training command implementation.

This module contains training command functionality extracted from
cli.py for better modularity.

Split from cli.py for better maintainability.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any

import jax
import jax.numpy as jnp

from ..parsers.base import create_training_parser
from ..utils.gpu_warmup import setup_jax_environment, perform_gpu_warmup

logger = logging.getLogger(__name__)


def run_standard_training(config: Dict, args) -> Dict[str, Any]:
    """Run real CPC+SNN training using CPCSNNTrainer."""
    try:
        # Import training components
        from training.base_trainer import CPCSNNTrainer, TrainingConfig
        from utils.device_auto_detection import setup_auto_device_optimization
        
        # Device auto-detection
        try:
            device_config, optimal_training_config = setup_auto_device_optimization()
            logger.info(f"🎮 Platform detected: {device_config.platform.upper()}")
            logger.info(f"⚡ Expected speedup: {device_config.expected_speedup:.1f}x")
        except ImportError:
            logger.warning("Auto-detection not available, using default settings")
            optimal_training_config = {}
        
        # Create output directory
        training_dir = args.output_dir / f"standard_training_{config['training']['batch_size']}bs"
        training_dir.mkdir(parents=True, exist_ok=True)
        
        # Create training configuration
        trainer_config = TrainingConfig(
            model_name="cpc_snn_gw",
            learning_rate=config['training']['cpc_lr'],
            batch_size=config['training']['batch_size'],
            num_epochs=config['training']['cpc_epochs'],
            output_dir=str(training_dir),
            use_wandb=getattr(args, 'wandb', False)
        )
        
        logger.info("🔧 Real CPC+SNN training pipeline:")
        logger.info(f"   - Batch Size: {trainer_config.batch_size}")
        logger.info(f"   - Learning Rate: {trainer_config.learning_rate}")
        logger.info(f"   - Epochs: {trainer_config.num_epochs}")
        
        # Create and initialize trainer
        trainer = CPCSNNTrainer(trainer_config)
        model = trainer.create_model()
        
        # Load data
        signals, labels, test_signals, test_labels = _load_training_data(args, config)
        
        # Initialize training state
        sample_input = signals[:1]
        trainer.train_state = trainer.create_train_state(model, sample_input)
        
        # Run training loop
        training_results = _run_training_loop(trainer, trainer_config, signals, labels, training_dir)
        
        # Final evaluation
        final_results = _run_final_evaluation(trainer, test_signals, test_labels, signals, training_results)
        
        return final_results
        
    except Exception as e:
        logger.error(f"Standard training failed: {e}")
        return {'success': False, 'error': str(e)}


def run_enhanced_training(config: Dict, args) -> Dict[str, Any]:
    """Run enhanced training with mixed continuous+binary dataset."""
    try:
        from training.enhanced_gw_training import EnhancedGWTrainer, EnhancedGWConfig
        
        # Create enhanced config
        enhanced_config = EnhancedGWConfig(
            num_continuous_signals=200,
            num_binary_signals=200,
            signal_duration=4.0,
            batch_size=config['training']['batch_size'],
            learning_rate=config['training']['cpc_lr'],
            num_epochs=50,
            output_dir=str(args.output_dir / "enhanced_training")
        )
        
        # Create and run trainer
        trainer = EnhancedGWTrainer(enhanced_config)
        result = trainer.run_enhanced_training()
        
        return {
            'success': True,
            'metrics': result.get('final_metrics', {}),
            'model_path': result.get('model_path', enhanced_config.output_dir)
        }
        
    except Exception as e:
        logger.error(f"Enhanced training failed: {e}")
        return {'success': False, 'error': str(e)}


def run_complete_enhanced_training(config: Dict, args) -> Dict[str, Any]:
    """Run complete enhanced training with ALL 5 revolutionary improvements."""
    try:
        from training.complete_enhanced_training import CompleteEnhancedTrainer, CompleteEnhancedConfig
        from models.snn_utils import SurrogateGradientType
        
        # Create complete enhanced config
        complete_config = CompleteEnhancedConfig(
            # Core parameters
            num_epochs=args.epochs,
            batch_size=min(args.batch_size, 16),
            learning_rate=5e-4,
            sequence_length=256,
            
            # Enable all 5 enhancements
            use_temporal_transformer=True,
            use_learnable_thresholds=True,
            use_enhanced_lif=True,
            use_momentum_negatives=True,
            use_adaptive_surrogate=True,
            
            # Output
            output_dir=str(args.output_dir / "complete_enhanced_training")
        )
        
        logger.info("🚀 COMPLETE ENHANCED TRAINING - ALL 5 IMPROVEMENTS ACTIVE!")
        logger.info("   1. 🧠 Adaptive Multi-Scale Surrogate Gradients")
        logger.info("   2. 🔄 Temporal Transformer with Multi-Scale Convolution")
        logger.info("   3. 🎯 Learnable Multi-Threshold Spike Encoding")
        logger.info("   4. 💾 Enhanced LIF with Memory and Refractory Period")
        logger.info("   5. 🚀 Momentum-based InfoNCE with Hard Negative Mining")
        
        # Create trainer
        trainer = CompleteEnhancedTrainer(complete_config)
        
        # Load enhanced data
        train_data = _load_enhanced_data(args, complete_config)
        
        # Run training
        result = trainer.run_complete_enhanced_training(
            train_data=train_data,
            num_epochs=complete_config.num_epochs
        )
        
        if result and result.get('success', False):
            logger.info("✅ Complete enhanced training finished successfully!")
            logger.info("🚀 ALL 5 ENHANCEMENTS SUCCESSFULLY INTEGRATED!")
            
            return {
                'success': True,
                'metrics': result.get('metrics', {}),
                'model_path': complete_config.output_dir,
                'final_accuracy': result.get('final_accuracy'),
                'final_loss': result.get('final_loss')
            }
        else:
            raise RuntimeError("Complete enhanced training pipeline failed")
            
    except Exception as e:
        logger.error(f"Complete enhanced training failed: {e}")
        return {'success': False, 'error': str(e)}


def _load_training_data(args, config):
    """Load training data based on configuration."""
    # MLGWSC-1 route
    if getattr(args, 'use_mlgwsc', False):
        return _load_mlgwsc_data(args)
    # Synthetic quick route
    elif getattr(args, 'synthetic_quick', False):
        return _load_synthetic_data(args)
    # Real LIGO route
    else:
        return _load_real_ligo_data(args)


def _load_mlgwsc_data(args):
    """Load MLGWSC-1 professional dataset."""
    logger.info("📦 Using MLGWSC-1 professional dataset")
    
    from implement_mlgwsc_loader import MLGWSCDataLoader
    from utils.data_split import create_stratified_split
    
    # Setup paths
    default_bg = "/teamspace/studios/this_studio/data/dataset-4/v2/val_background_s24w6d1_1.hdf"
    background_hdf = str(getattr(args, 'mlgwsc_background_hdf', None) or default_bg)
    
    if not Path(background_hdf).exists():
        raise FileNotFoundError(f"MLGWSC background HDF not found: {background_hdf}")
    
    # Create loader
    slice_len = int(float(getattr(args, 'mlgwsc_slice_seconds', 1.25)) * 2048)
    loader = MLGWSCDataLoader(
        background_hdf_path=background_hdf,
        injections_npy_path=getattr(args, 'mlgwsc_injections_npy', None),
        slice_len=slice_len,
        batch_size=int(args.batch_size)
    )
    
    # Collect samples
    max_samples = int(getattr(args, 'mlgwsc_samples', 1024))
    collected_x, collected_y = [], []
    total_collected = 0
    
    for batch_x, batch_y in loader.create_training_batches(batch_size=int(args.batch_size)):
        remain = max_samples - total_collected
        if remain <= 0:
            break
        take = min(remain, int(batch_x.shape[0]))
        collected_x.append(batch_x[:take])
        collected_y.append(batch_y[:take])
        total_collected += take
    
    if total_collected == 0:
        raise RuntimeError("MLGWSC loader yielded no samples")
    
    # Combine and split
    all_signals = jnp.concatenate(collected_x, axis=0)
    all_labels = jnp.concatenate(collected_y, axis=0)
    
    (signals, labels), (test_signals, test_labels) = create_stratified_split(
        all_signals, all_labels, train_ratio=0.8, random_seed=42
    )
    
    logger.info(f"MLGWSC samples: train={len(signals)}, test={len(test_signals)}")
    return signals, labels, test_signals, test_labels


def _load_synthetic_data(args):
    """Load synthetic data for quick testing."""
    logger.info("⚡ Using synthetic quick demo dataset")
    
    from data.gw_dataset_builder import create_evaluation_dataset
    from utils.data_split import create_stratified_split
    
    num_samples = int(getattr(args, 'synthetic_samples', 60))
    seq_len = 256
    
    train_data = create_evaluation_dataset(
        num_samples=num_samples,
        sequence_length=seq_len,
        sample_rate=4096,
        random_seed=42
    )
    
    # Convert to arrays
    all_signals = jnp.stack([sample[0] for sample in train_data])
    all_labels = jnp.array([sample[1] for sample in train_data])
    
    # Split data
    (signals, labels), (test_signals, test_labels) = create_stratified_split(
        all_signals, all_labels, train_ratio=0.8, random_seed=42
    )
    
    logger.info(f"Synthetic samples: train={len(signals)}, test={len(test_signals)}")
    return signals, labels, test_signals, test_labels


def _load_real_ligo_data(args):
    """Load real LIGO data."""
    logger.info("Creating REAL LIGO dataset with GW150914 data...")
    
    try:
        from data.real_ligo_integration import create_real_ligo_dataset
        from utils.data_split import create_stratified_split
        
        if getattr(args, 'quick_mode', False):
            # Quick mode
            (train_signals, train_labels), (test_signals, test_labels) = create_real_ligo_dataset(
                num_samples=200,
                window_size=int(getattr(args, 'window_size', 256)),
                quick_mode=True,
                return_split=True,
                train_ratio=0.8,
                overlap=float(getattr(args, 'overlap', 0.7))
            )
            signals, labels = train_signals, train_labels
            logger.info(f"Quick REAL LIGO samples: train={len(signals)}, test={len(test_signals)}")
        else:
            # Enhanced path
            from data.real_ligo_integration import create_enhanced_ligo_dataset
            
            enhanced_signals, enhanced_labels = create_enhanced_ligo_dataset(
                num_samples=2000,
                window_size=int(getattr(args, 'window_size', 256)),
                enhanced_overlap=0.9,
                data_augmentation=True,
                noise_scaling=True
            )
            
            (train_signals, train_labels), (test_signals, test_labels) = create_stratified_split(
                enhanced_signals, enhanced_labels, train_ratio=0.8, random_seed=42
            )
            signals, labels = train_signals, train_labels
            logger.info(f"Enhanced REAL LIGO samples: train={len(signals)}, test={len(test_signals)}")
        
        return signals, labels, test_signals, test_labels
        
    except ImportError:
        logger.warning("Real LIGO integration not available - falling back to synthetic")
        return _load_synthetic_data(args)


def _load_enhanced_data(args, config):
    """Load data for enhanced training."""
    try:
        from data.real_ligo_integration import create_enhanced_ligo_dataset
        
        logger.info("🚀 Loading ENHANCED LIGO dataset with augmentation...")
        enhanced_signals, enhanced_labels = create_enhanced_ligo_dataset(
            num_samples=2000,
            window_size=config.sequence_length,
            enhanced_overlap=0.9,
            data_augmentation=True,
            noise_scaling=True
        )
        return (enhanced_signals, enhanced_labels)
        
    except Exception as e:
        logger.warning(f"Real LIGO data unavailable: {e}")
        logger.info("🔄 Generating synthetic gravitational wave data...")
        
        # Fallback to synthetic
        key = jax.random.PRNGKey(42)
        signals = jax.random.normal(key, (1000, config.sequence_length))
        labels = jax.random.randint(jax.random.split(key)[0], (1000,), 0, 2)
        
        return (signals, labels)


def _run_training_loop(trainer, config, signals, labels, training_dir):
    """Run the main training loop."""
    logger.info("⏳ Starting real training loop...")
    
    epoch_results = []
    
    for epoch in range(config.num_epochs):
        logger.info(f"🔥 Epoch {epoch+1}/{config.num_epochs}")
        
        # Create batches
        num_samples = len(signals)
        num_batches = min(
            (num_samples + config.batch_size - 1) // config.batch_size,
            100  # Cap for quick feedback
        )
        
        epoch_losses = []
        epoch_accuracies = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * config.batch_size
            end_idx = min(start_idx + config.batch_size, num_samples)
            
            batch_signals = signals[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            batch = (batch_signals, batch_labels)
            
            # Training step
            trainer.train_state, metrics = trainer.train_step(trainer.train_state, batch)
            
            epoch_losses.append(metrics.loss)
            epoch_accuracies.append(metrics.accuracy)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"      Step {batch_idx+1}/{num_batches} "
                           f"loss={metrics.loss:.4f} acc={metrics.accuracy:.3f}")
        
        # Epoch summary
        import numpy as np
        avg_loss = float(np.mean(np.array(epoch_losses)))
        avg_accuracy = float(np.mean(np.array(epoch_accuracies)))
        
        logger.info(f"      Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        
        epoch_results.append({
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': avg_accuracy
        })
    
    # Final training results
    final_epoch = epoch_results[-1] if epoch_results else {'loss': 0.0, 'accuracy': 0.0}
    training_results = {
        'final_loss': final_epoch['loss'],
        'accuracy': final_epoch['accuracy'],
        'epochs_completed': config.num_epochs
    }
    
    return training_results


def _run_final_evaluation(trainer, test_signals, test_labels, train_signals, training_results):
    """Run final evaluation on test set."""
    logger.info("🔧 Running final evaluation...")
    
    try:
        from training.test_evaluation import evaluate_on_test_set, create_test_evaluation_summary
        
        test_results = evaluate_on_test_set(
            trainer.train_state,
            test_signals,
            test_labels,
            train_signals=train_signals,
            verbose=True
        )
        
        # Create summary
        test_summary = create_test_evaluation_summary(
            train_accuracy=training_results['accuracy'],
            test_results=test_results,
            data_source="Real LIGO GW150914",
            num_epochs=training_results['epochs_completed']
        )
        
        logger.info(test_summary)
        
        return {
            'success': True,
            'metrics': {
                'final_train_loss': training_results['final_loss'],
                'final_train_accuracy': training_results['accuracy'],
                'final_test_accuracy': test_results['test_accuracy'],
                'total_epochs': training_results['epochs_completed'],
                'has_proper_test_set': test_results['has_proper_test_set'],
                'model_collapse': test_results.get('model_collapse', False)
            }
        }
        
    except Exception as e:
        logger.error(f"Final evaluation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'partial_metrics': training_results
        }


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
    
    # Setup JAX environment
    setup_jax_environment(args.device)
    
    # GPU warmup
    if args.device != 'cpu':
        perform_gpu_warmup(args.device)
    
    # Apply optimizations
    from cli.utils.gpu_warmup import apply_performance_optimizations, setup_training_environment
    apply_performance_optimizations()
    setup_training_environment()
    
    # Load configuration
    from utils.config import load_config
    config = load_config(args.config)
    
    # Override config with CLI args
    _update_config_from_args(config, args)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    from utils.config import save_config
    config_path = args.output_dir / "config.yaml"
    save_config(config, config_path)
    
    # Run training based on mode
    try:
        logger.info(f"🎯 Starting {args.mode} training mode...")
        
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
    
    if args.device and args.device != 'auto':
        config.setdefault('platform', {})
        config['platform']['device'] = args.device
    
    if args.wandb:
        config.setdefault('logging', {})
        config['logging']['wandb_project'] = "cpc-snn-training"


# Export training command
__all__ = [
    "train_cmd",
    "run_standard_training",
    "run_enhanced_training", 
    "run_complete_enhanced_training"
]
