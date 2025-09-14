"""
Data loading logic for training commands.

This module contains data loading functionality extracted from
train.py for better modularity.

Split from cli/commands/train.py for better maintainability.
"""

import logging
from pathlib import Path
from typing import Tuple
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def _load_mlgwsc_data(args) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load MLGWSC-1 professional dataset."""
    logger.info("📦 Using MLGWSC-1 professional dataset via MLGWSCDataLoader")
    
    try:
        from implement_mlgwsc_loader import MLGWSCDataLoader
        from utils.data_split import create_stratified_split
    except ImportError as e:
        logger.error(f"❌ Cannot import MLGWSC loader: {e}")
        raise
    
    # Setup paths with defaults
    default_bg = "/teamspace/studios/this_studio/data/dataset-4/v2/val_background_s24w6d1_1.hdf"
    background_hdf = str(getattr(args, 'mlgwsc_background_hdf', None) or default_bg)
    injections_npy = str(getattr(args, 'mlgwsc_injections_npy', "")) or None
    
    if not Path(background_hdf).exists():
        raise FileNotFoundError(f"MLGWSC background HDF not found: {background_hdf}")
    
    # Create loader configuration
    slice_len = int(float(getattr(args, 'mlgwsc_slice_seconds', 1.25)) * 2048)
    
    loader = MLGWSCDataLoader(
        background_hdf_path=background_hdf,
        injections_npy_path=injections_npy,
        slice_len=slice_len,
        batch_size=int(args.batch_size)
    )
    
    # Collect samples efficiently
    max_samples = int(getattr(args, 'mlgwsc_samples', 1024))
    collected_x = []
    collected_y = []
    total_collected = 0
    
    for batch_x, batch_y in loader.create_training_batches(batch_size=int(args.batch_size)):
        remain = max_samples - total_collected
        if remain <= 0:
            break
        take = min(remain, int(batch_x.shape[0]))
        collected_x.append(batch_x[:take])
        collected_y.append(batch_y[:take])
        total_collected += take
        
        if total_collected >= max_samples:
            break
    
    if total_collected == 0:
        raise RuntimeError("MLGWSC loader yielded no samples. Check dataset paths.")
    
    # Combine and split
    all_signals = jnp.concatenate(collected_x, axis=0)
    all_labels = jnp.concatenate(collected_y, axis=0)
    
    (signals, labels), (test_signals, test_labels) = create_stratified_split(
        all_signals, all_labels, train_ratio=0.8, random_seed=42
    )
    
    logger.info(f"MLGWSC samples: train={len(signals)}, test={len(test_signals)}, T={signals.shape[-1]}")
    
    # Optional PSD whitening
    if getattr(args, 'whiten_psd', False):
        signals, test_signals = _apply_psd_whitening(signals, test_signals)
    
    return signals, labels, test_signals, test_labels


def _load_synthetic_data(args) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load synthetic data for quick testing."""
    logger.info("⚡ Synthetic quick-mode enabled: using synthetic demo dataset")
    
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
    
    # Convert to JAX arrays
    all_signals = jnp.stack([sample[0] for sample in train_data])
    all_labels = jnp.array([sample[1] for sample in train_data])
    
    # Split data
    (signals, labels), (test_signals, test_labels) = create_stratified_split(
        all_signals, all_labels, train_ratio=0.8, random_seed=42
    )
    
    logger.info(f"Synthetic samples: train={len(signals)}, test={len(test_signals)}")
    return signals, labels, test_signals, test_labels


def _load_real_ligo_data(args) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load real LIGO data with GW150914."""
    logger.info("Creating REAL LIGO dataset with GW150914 data...")
    
    try:
        # Use synthetic generator instead of deprecated real_ligo_integration
        from data.gw_synthetic_generator import ContinuousGWGenerator
        from data.gw_signal_params import SignalConfiguration, GeneratorSettings
        from utils.data_split import create_stratified_split
        
        if getattr(args, 'quick_mode', False) and not getattr(args, 'synthetic_quick', False):
            # Quick path - lightweight real LIGO windows
            logger.info("⚡ Quick mode: using lightweight real LIGO windows")
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
            # Enhanced path with augmentation
            logger.info("🚀 Enhanced mode: using augmented LIGO dataset")
            
            # Optional PyCBC integration
            pycbc_ds = None
            if getattr(args, 'use_pycbc', False):
                pycbc_ds = _load_pycbc_dataset(args)
            
            enhanced_signals, enhanced_labels = create_enhanced_ligo_dataset(
                num_samples=2000,
                window_size=int(getattr(args, 'window_size', 256)),
                enhanced_overlap=0.9,
                data_augmentation=True,
                noise_scaling=True
            )
            
            # Mix PyCBC if available
            if pycbc_ds is not None:
                enhanced_signals, enhanced_labels = _mix_with_pycbc(enhanced_signals, enhanced_labels, pycbc_ds)
            
            # Split enhanced dataset
            (train_signals, train_labels), (test_signals, test_labels) = create_stratified_split(
                enhanced_signals, enhanced_labels, train_ratio=0.8, random_seed=42
            )
            signals, labels = train_signals, train_labels
            logger.info(f"Enhanced REAL LIGO samples: train={len(signals)}, test={len(test_signals)}")
        
        return signals, labels, test_signals, test_labels
        
    except ImportError:
        logger.warning("Real LIGO integration not available - falling back to synthetic")
        return _load_synthetic_data(args)


def _load_pycbc_dataset(args):
    """Load PyCBC enhanced dataset if requested."""
    try:
        from data.pycbc_integration import create_pycbc_enhanced_dataset
        
        pycbc_ds = create_pycbc_enhanced_dataset(
            num_samples=2000,
            window_size=int(getattr(args, 'window_size', 256)),
            sample_rate=4096,
            snr_range=(float(getattr(args, 'pycbc_snr_min', 8.0)), float(getattr(args, 'pycbc_snr_max', 20.0))),
            mass_range=(float(getattr(args, 'pycbc_mass_min', 10.0)), float(getattr(args, 'pycbc_mass_max', 50.0))),
            positive_ratio=0.35,
            random_seed=42,
            psd_name=str(getattr(args, 'pycbc_psd', 'aLIGOZeroDetHighPower')),
            whiten=bool(getattr(args, 'pycbc_whiten', True)),
            multi_channel=bool(getattr(args, 'pycbc_multi_channel', False))
        )
        
        if pycbc_ds is not None:
            logger.info("✅ PyCBC enhanced synthetic dataset available for mixing")
            return pycbc_ds
    
    except Exception as e:
        logger.warning(f"PyCBC dataset unavailable: {e}")
    
    return None


def _mix_with_pycbc(enhanced_signals, enhanced_labels, pycbc_ds):
    """Mix enhanced LIGO data with PyCBC dataset."""
    import jax
    
    pycbc_signals, pycbc_labels = pycbc_ds
    
    # Concatenate datasets
    mixed_signals = jnp.concatenate([enhanced_signals, pycbc_signals], axis=0)
    mixed_labels = jnp.concatenate([enhanced_labels, pycbc_labels], axis=0)
    
    # Shuffle combined dataset
    key = jax.random.PRNGKey(7)
    perm = jax.random.permutation(key, len(mixed_signals))
    mixed_signals = mixed_signals[perm]
    mixed_labels = mixed_labels[perm]
    
    logger.info(f"Mixed PyCBC dataset: {len(mixed_signals)} total samples")
    return mixed_signals, mixed_labels


def _apply_psd_whitening(train_signals, test_signals):
    """Apply PSD whitening to signals."""
    try:
        logger.info("🔧 Applying PSD whitening (AdvancedDataPreprocessor)...")
        from data.gw_preprocessor import AdvancedDataPreprocessor
        
        preprocessor = AdvancedDataPreprocessor(sample_rate=2048, apply_whitening=True)
        
        # Process training signals
        train_results = preprocessor.process_batch([train_signals[i] for i in range(len(train_signals))])
        processed_train = jnp.stack([r.strain_data for r in train_results])
        
        # Process test signals
        test_results = preprocessor.process_batch([test_signals[i] for i in range(len(test_signals))])
        processed_test = jnp.stack([r.strain_data for r in test_results])
        
        logger.info("✅ PSD whitening applied to train/test sets")
        return processed_train, processed_test
        
    except Exception as e:
        logger.warning(f"⚠️ Whitening skipped due to error: {e}")
        return train_signals, test_signals


def load_training_data(args) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Main data loading function that routes to appropriate loader.
    
    Args:
        args: CLI arguments
        
    Returns:
        Tuple of (train_signals, train_labels, test_signals, test_labels)
    """
    logger.info("📊 Loading training data...")
    
    # Route to appropriate data loader
    if getattr(args, 'use_mlgwsc', False):
        return _load_mlgwsc_data(args)
    elif getattr(args, 'synthetic_quick', False):
        return _load_synthetic_data(args)
    else:
        return _load_real_ligo_data(args)


# Export data loading functions
__all__ = [
    "load_training_data",
    "_load_mlgwsc_data",
    "_load_synthetic_data", 
    "_load_real_ligo_data"
]
