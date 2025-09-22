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
        # Absolute imports from repo root package
        from data.mlgwsc_data_loader import MLGWSCDataLoader
        from utils.data_split import create_stratified_split
        # Use the same config loader as CLI (respects -c path)
        from utils.config import load_config
    except ImportError as e:
        logger.error(f"❌ Cannot import MLGWSC loader: {e}")
        raise
    
    # Load config-driven data directory (no hardcoded paths)
    try:
        cfg = load_config(getattr(args, 'config', None))
    except Exception as e:
        logger.warning(f"⚠️ Falling back to default config loader due to: {e}")
        cfg = load_config()
    # Allow env override for data dir
    import os
    env_data_dir = os.environ.get('CPC_SNN_DATA_DIR')
    data_dir = str(env_data_dir or cfg['system']['data_dir'])
    if not Path(data_dir).exists():
        raise FileNotFoundError(f"Configured data_dir not found: {data_dir}")
    
    # Create loader (auto-discovers files in data_dir)
    loader = MLGWSCDataLoader(
        data_dir=data_dir,
        mode="training",
        config=cfg
    )
    
    # Collect samples efficiently
    max_samples = int(getattr(args, 'mlgwsc_samples', -1))
    collected_x = []
    collected_y = []
    total_collected = 0
    
    # Create labeled segments from available files
    data_segments, labels = loader.create_labeled_dataset()
    all_signals = jnp.stack([seg for seg in data_segments])
    all_labels = jnp.array(labels)
    
    if max_samples > 0 and total_collected == 0:
        # Respect max_samples if provided
        take = min(max_samples, int(all_signals.shape[0]))
        all_signals = all_signals[:take]
        all_labels = all_labels[:take]
        total_collected = take
    else:
        total_collected = int(all_signals.shape[0])
    
    if total_collected == 0:
        raise RuntimeError("MLGWSC loader yielded no samples. Check dataset paths.")
    
    (signals, labels), (test_signals, test_labels) = create_stratified_split(
        all_signals, all_labels, train_ratio=0.8, random_seed=42
    )
    
    # Log true temporal and feature dims assuming [N, T, F]
    try:
        t_dim = int(signals.shape[1]) if signals.ndim >= 2 else int(signals.shape[0])
        f_dim = int(signals.shape[-1]) if signals.ndim >= 2 else 1
        logger.info(f"MLGWSC samples: train={len(signals)}, test={len(test_signals)}, T={t_dim}, F={f_dim}")
    except Exception:
        logger.info(f"MLGWSC samples: train={len(signals)}, test={len(test_signals)}")
    
    # Optional PSD whitening
    if getattr(args, 'whiten_psd', False):
        # Use sample_rate from config if available; fallback 4096
        try:
            sr = int(cfg.get('data', {}).get('sample_rate', 4096))
        except Exception:
            sr = 4096
        signals, test_signals = _apply_psd_whitening(signals, test_signals, sample_rate=sr)
    
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


def _apply_psd_whitening(train_signals, test_signals, sample_rate: int = 4096):
    """Apply PSD whitening to signals."""
    try:
        logger.info("🔧 Applying PSD whitening (AdvancedDataPreprocessor)...")
        from data.gw_preprocessor import AdvancedDataPreprocessor
        
        preprocessor = AdvancedDataPreprocessor(sample_rate=sample_rate, apply_whitening=True)
        
        # Reduce multi-channel to mono (mean over features) and ensure 1D per sample
        train_list = [jnp.mean(train_signals[i], axis=-1) if train_signals[i].ndim == 2 else train_signals[i]
                      for i in range(len(train_signals))]
        test_list = [jnp.mean(test_signals[i], axis=-1) if test_signals[i].ndim == 2 else test_signals[i]
                     for i in range(len(test_signals))]
        # Process training signals
        train_results = preprocessor.process_batch(train_list)
        processed_train = jnp.stack([r.strain_data for r in train_results])
        
        # Process test signals
        test_results = preprocessor.process_batch(test_list)
        processed_test = jnp.stack([r.strain_data for r in test_results])
        
        logger.info("✅ PSD whitening applied to train/test sets")
        # Expand channel dim to [N, T, 1]
        return processed_train[..., None], processed_test[..., None]
        
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
