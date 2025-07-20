"""
GW Dataset Builder: Dataset Creation and Export Functionality
Extracted from continuous_gw_generator.py for modular architecture.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import jax
import jax.numpy as jnp
from .gw_signal_params import GeneratorSettings
from .gw_synthetic_generator import ContinuousGWGenerator

logger = logging.getLogger(__name__)

# Optional dependencies for export
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    logger.warning("h5py not available - HDF5 export disabled")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available - TFRecord export disabled")


class GWDatasetBuilder:
    """
    Builder class for creating and exporting GW datasets.
    
    Features:
    - Mixed signal types (continuous GW, noise, binary mergers)
    - Configurable train/validation/test splits
    - Multiple export formats (HDF5, TFRecord, NumPy)
    - Dataset statistics and validation
    """
    
    def __init__(self, 
                 generator: ContinuousGWGenerator,
                 settings: Optional[GeneratorSettings] = None):
        """
        Initialize dataset builder.
        
        Args:
            generator: Continuous GW signal generator
            settings: Generator settings for dataset creation
        """
        self.generator = generator
        self.settings = settings or GeneratorSettings()
        
        logger.info(f"Initialized GW Dataset Builder")
        logger.info(f"  Number of signals: {self.settings.num_signals}")
        logger.info(f"  Signal duration: {self.settings.signal_duration}s")
        logger.info(f"  Include noise-only: {self.settings.include_noise_only}")


    def generate_training_dataset(self, 
                                num_signals: int = 100,
                                signal_duration: float = 4.0,
                                include_noise_only: bool = True,
                                key: Optional[jax.random.PRNGKey] = None) -> Dict:
        """
        Generate comprehensive training dataset with multiple signal types.
        
        Args:
            num_signals: Number of signals per type
            signal_duration: Duration of each signal (seconds)
            include_noise_only: Whether to include pure noise samples
            key: JAX random key
            
        Returns:
            Dictionary containing dataset arrays and metadata
        """
        if key is None:
            key = jax.random.PRNGKey(42)
            
        signals = []
        labels = []
        metadata = []
        
        # Split random key for different signal types
        keys = jax.random.split(key, 10)
        key_idx = 0
        
        # Generate continuous GW signals
        logger.info(f"Generating {num_signals} continuous GW signals...")
        params_list = self.generator.generate_signal_parameters(
            num_signals=num_signals, 
            key=keys[key_idx]
        )
        key_idx += 1
        
        for i, params in enumerate(params_list):
            signal = self.generator.create_synthetic_timeseries(
                params, 
                duration=signal_duration,
                key=keys[key_idx % len(keys)]
            )
            key_idx += 1
            
            signals.append(signal)
            labels.append(1)  # Continuous GW signal
            metadata.append({
                'signal_type': 'continuous_gw',
                'frequency': params.frequency,
                'amplitude': params.amplitude_h0,
                'alpha': params.alpha,
                'delta': params.delta
            })
        
        # Generate noise-only signals if requested
        if include_noise_only:
            logger.info(f"Generating {num_signals} noise-only signals...")
            for i in range(num_signals):
                noise = self.generator.generate_noise_timeseries(
                    duration=signal_duration,
                    key=keys[key_idx % len(keys)]
                )
                key_idx += 1
                
                signals.append(noise)
                labels.append(0)  # Noise only
                metadata.append({
                    'signal_type': 'noise',
                    'noise_level': self.settings.noise_level
                })
        
        # Convert to JAX arrays
        signals_array = jnp.stack(signals)
        labels_array = jnp.array(labels)
        
        logger.info(f"Generated dataset: {signals_array.shape} signals, "
                   f"{len(jnp.unique(labels_array))} classes")
        
        return {
            'signals': signals_array,
            'labels': labels_array,
            'metadata': metadata,
            'num_samples': len(signals),
            'signal_duration': signal_duration,
            'sampling_rate': self.generator.sampling_rate,
            'classes': {0: 'noise', 1: 'continuous_gw'}
        }


    def split_dataset(self, 
                     dataset: Dict,
                     train_split: float = 0.7,
                     validation_split: float = 0.2,
                     test_split: float = 0.1,
                     shuffle: bool = True,
                     key: Optional[jax.random.PRNGKey] = None) -> Dict:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            dataset: Dataset dictionary from generate_training_dataset
            train_split: Fraction for training set
            validation_split: Fraction for validation set
            test_split: Fraction for test set
            shuffle: Whether to shuffle before splitting
            key: JAX random key for shuffling
            
        Returns:
            Dictionary with train/val/test splits
        """
        if abs(train_split + validation_split + test_split - 1.0) > 1e-6:
            raise ValueError("Split fractions must sum to 1.0")
        
        signals = dataset['signals']
        labels = dataset['labels']
        metadata = dataset['metadata']
        num_samples = len(signals)
        
        # Shuffle if requested
        if shuffle:
            if key is None:
                key = jax.random.PRNGKey(123)
            
            indices = jax.random.permutation(key, num_samples)
            signals = signals[indices]
            labels = labels[indices]
            metadata = [metadata[i] for i in indices]
        
        # Calculate split indices
        train_end = int(train_split * num_samples)
        val_end = train_end + int(validation_split * num_samples)
        
        # Split data
        splits = {
            'train': {
                'signals': signals[:train_end],
                'labels': labels[:train_end],
                'metadata': metadata[:train_end]
            },
            'validation': {
                'signals': signals[train_end:val_end],
                'labels': labels[train_end:val_end],
                'metadata': metadata[train_end:val_end]
            },
            'test': {
                'signals': signals[val_end:],
                'labels': labels[val_end:],
                'metadata': metadata[val_end:]
            }
        }
        
        # Add global metadata
        for split_name in splits:
            splits[split_name].update({
                'signal_duration': dataset['signal_duration'],
                'sampling_rate': dataset['sampling_rate'],
                'classes': dataset['classes']
            })
        
        logger.info(f"Dataset split: train={len(splits['train']['signals'])}, "
                   f"val={len(splits['validation']['signals'])}, "
                   f"test={len(splits['test']['signals'])}")
        
        return splits


    def compute_dataset_statistics(self, dataset: Dict) -> Dict:
        """
        Compute comprehensive statistics for the dataset.
        
        Args:
            dataset: Dataset dictionary
            
        Returns:
            Dictionary with computed statistics
        """
        signals = dataset['signals']
        labels = dataset['labels']
        
        # Basic statistics
        stats = {
            'num_samples': len(signals),
            'signal_shape': signals.shape,
            'signal_duration': dataset.get('signal_duration', 'unknown'),
            'sampling_rate': dataset.get('sampling_rate', 'unknown'),
            'num_classes': len(jnp.unique(labels))
        }
        
        # Class distribution
        unique_labels, counts = jnp.unique(labels, return_counts=True)
        stats['class_distribution'] = {}
        for label, count in zip(unique_labels, counts):
            class_name = dataset.get('classes', {}).get(int(label), f'class_{label}')
            stats['class_distribution'][class_name] = int(count)
        
        # Signal statistics
        stats['signal_stats'] = {
            'mean': float(jnp.mean(signals)),
            'std': float(jnp.std(signals)),
            'min': float(jnp.min(signals)),
            'max': float(jnp.max(signals)),
            'rms': float(jnp.sqrt(jnp.mean(signals**2)))
        }
        
        # Per-class statistics
        stats['per_class_stats'] = {}
        for label in unique_labels:
            mask = labels == label
            class_signals = signals[mask]
            class_name = dataset.get('classes', {}).get(int(label), f'class_{label}')
            
            stats['per_class_stats'][class_name] = {
                'count': int(jnp.sum(mask)),
                'mean_amplitude': float(jnp.mean(jnp.abs(class_signals))),
                'rms': float(jnp.sqrt(jnp.mean(class_signals**2))),
                'snr_estimate': float(jnp.mean(class_signals**2) / jnp.var(class_signals))
            }
        
        return stats


    def export_to_hdf5(self, 
                      dataset: Dict,
                      output_path: Union[str, Path],
                      compression: str = 'gzip',
                      shuffle: bool = True) -> bool:
        """
        Export dataset to HDF5 format.
        
        Args:
            dataset: Dataset dictionary
            output_path: Output file path
            compression: HDF5 compression method
            shuffle: Whether to shuffle data before export
            
        Returns:
            True if successful, False otherwise
        """
        if not HDF5_AVAILABLE:
            logger.error("HDF5 export requires h5py package")
            return False
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(output_path, 'w') as f:
                # Write main data
                f.create_dataset('signals', data=dataset['signals'], 
                               compression=compression)
                f.create_dataset('labels', data=dataset['labels'],
                               compression=compression)
                
                # Write metadata
                for key, value in dataset.items():
                    if key not in ['signals', 'labels', 'metadata']:
                        if isinstance(value, (int, float, str)):
                            f.attrs[key] = value
                        elif isinstance(value, dict):
                            # Store dict as JSON string
                            import json
                            f.attrs[f'{key}_json'] = json.dumps(value)
                
                # Write detailed metadata if available
                if 'metadata' in dataset:
                    meta_group = f.create_group('metadata')
                    for i, meta in enumerate(dataset['metadata']):
                        sample_group = meta_group.create_group(f'sample_{i}')
                        for key, value in meta.items():
                            sample_group.attrs[key] = value
            
            logger.info(f"Dataset exported to HDF5: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"HDF5 export failed: {e}")
            return False


    def export_to_numpy(self,
                       dataset: Dict,
                       output_dir: Union[str, Path]) -> bool:
        """
        Export dataset to NumPy format.
        
        Args:
            dataset: Dataset dictionary
            output_dir: Output directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save main arrays
            jnp.save(output_dir / 'signals.npy', dataset['signals'])
            jnp.save(output_dir / 'labels.npy', dataset['labels'])
            
            # Save metadata as JSON
            import json
            metadata_to_save = {}
            for key, value in dataset.items():
                if key not in ['signals', 'labels']:
                    if isinstance(value, (int, float, str, list, dict)):
                        metadata_to_save[key] = value
            
            with open(output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata_to_save, f, indent=2)
            
            logger.info(f"Dataset exported to NumPy: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"NumPy export failed: {e}")
            return False


def create_mixed_gw_dataset(continuous_generator: ContinuousGWGenerator,
                          binary_data: Optional[Dict] = None,
                          mix_ratio: float = 0.5,
                          num_total_signals: int = 300,
                          signal_duration: float = 4.0,
                          key: Optional[jax.random.PRNGKey] = None) -> Dict:
    """
    Create mixed dataset with continuous GW and binary merger signals.
    
    Args:
        continuous_generator: Generator for continuous signals
        binary_data: Optional binary merger data
        mix_ratio: Ratio of continuous to binary signals
        num_total_signals: Total number of signals
        signal_duration: Duration per signal
        key: JAX random key
        
    Returns:
        Mixed dataset dictionary
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    builder = GWDatasetBuilder(continuous_generator)
    
    # Calculate signal counts
    num_continuous = int(num_total_signals * mix_ratio)
    num_noise = num_total_signals - num_continuous
    
    # Generate dataset with specified proportions
    dataset = builder.generate_training_dataset(
        num_signals=num_continuous,
        signal_duration=signal_duration,
        include_noise_only=True,
        key=key
    )
    
    # If binary data provided, incorporate it
    if binary_data is not None:
        # TODO: Implement binary merger integration
        logger.warning("Binary merger integration not yet implemented")
    
    return dataset


def test_dataset_builder():
    """Test the dataset builder functionality."""
    try:
        from .gw_signal_params import SignalConfiguration
        
        # Create generator and builder
        config = SignalConfiguration(duration=100.0)
        generator = ContinuousGWGenerator(config=config)
        settings = GeneratorSettings(num_signals=10, signal_duration=2.0)
        builder = GWDatasetBuilder(generator, settings)
        
        # Test dataset generation
        dataset = builder.generate_training_dataset(
            num_signals=5,
            signal_duration=1.0,
            key=jax.random.PRNGKey(42)
        )
        
        assert 'signals' in dataset
        assert 'labels' in dataset
        assert dataset['signals'].shape[0] == 10  # 5 continuous + 5 noise
        
        # Test dataset splitting
        splits = builder.split_dataset(dataset, key=jax.random.PRNGKey(123))
        
        assert 'train' in splits
        assert 'validation' in splits  
        assert 'test' in splits
        
        # Test statistics
        stats = builder.compute_dataset_statistics(dataset)
        assert 'num_samples' in stats
        assert 'class_distribution' in stats
        
        logger.info("✅ Dataset builder tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset builder test failed: {e}")
        return False 