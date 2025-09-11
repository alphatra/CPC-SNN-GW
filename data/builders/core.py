"""
Core dataset builder implementation.

This module contains the main GWDatasetBuilder class extracted from
gw_dataset_builder.py for better modularity.

Split from gw_dataset_builder.py for better maintainability.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import jax
import jax.numpy as jnp

from ..gw_signal_params import GeneratorSettings
from ..gw_synthetic_generator import ContinuousGWGenerator

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
            settings: Generator settings (optional)
        """
        self.generator = generator
        self.settings = settings or GeneratorSettings()
        
        # Dataset metadata
        self.metadata = {
            'created_at': None,
            'generator_config': None,
            'dataset_stats': None,
            'export_info': {}
        }
        
        logger.info("GWDatasetBuilder initialized")
    
    def build_mixed_dataset(self, 
                           total_samples: int,
                           continuous_ratio: float = 0.3,
                           noise_ratio: float = 0.5,
                           binary_ratio: float = 0.2,
                           signal_duration: float = 4.0) -> Dict[str, jnp.ndarray]:
        """
        Build mixed dataset with different signal types.
        
        Args:
            total_samples: Total number of samples to generate
            continuous_ratio: Fraction of continuous GW signals
            noise_ratio: Fraction of noise-only samples
            binary_ratio: Fraction of binary merger signals
            signal_duration: Duration of each signal in seconds
            
        Returns:
            Dictionary with data and labels
        """
        logger.info(f"Building mixed dataset: {total_samples} samples")
        
        # Validate ratios
        if abs(continuous_ratio + noise_ratio + binary_ratio - 1.0) > 1e-6:
            raise ValueError("Signal ratios must sum to 1.0")
        
        # Calculate sample counts
        num_continuous = int(total_samples * continuous_ratio)
        num_noise = int(total_samples * noise_ratio) 
        num_binary = total_samples - num_continuous - num_noise  # Remainder
        
        all_signals = []
        all_labels = []
        
        # ✅ CONTINUOUS GW SIGNALS
        if num_continuous > 0:
            logger.info(f"Generating {num_continuous} continuous GW signals...")
            continuous_data = self.generator.generate_training_dataset(
                num_signals=num_continuous,
                signal_duration=signal_duration,
                include_noise_only=False
            )
            
            # Extract continuous signals (label == 1)
            continuous_signals = continuous_data['data'][continuous_data['labels'] == 1]
            
            # Take exactly num_continuous signals
            if len(continuous_signals) >= num_continuous:
                selected_continuous = continuous_signals[:num_continuous]
            else:
                # Generate more if needed
                additional_needed = num_continuous - len(continuous_signals)
                additional_data = self.generator.generate_training_dataset(
                    num_signals=additional_needed,
                    signal_duration=signal_duration,
                    include_noise_only=False
                )
                additional_signals = additional_data['data'][additional_data['labels'] == 1]
                selected_continuous = jnp.concatenate([continuous_signals, additional_signals[:additional_needed]])
            
            all_signals.append(selected_continuous)
            all_labels.extend([1] * num_continuous)  # Label 1: continuous GW
        
        # ✅ NOISE SIGNALS
        if num_noise > 0:
            logger.info(f"Generating {num_noise} noise signals...")
            noise_data = self.generator.generate_training_dataset(
                num_signals=num_noise,
                signal_duration=signal_duration,
                include_noise_only=True
            )
            
            # Extract noise signals (label == 0)
            noise_signals = noise_data['data'][noise_data['labels'] == 0][:num_noise]
            
            all_signals.append(noise_signals)
            all_labels.extend([0] * num_noise)  # Label 0: noise
        
        # ✅ BINARY MERGER SIGNALS (simplified)
        if num_binary > 0:
            logger.info(f"Generating {num_binary} binary merger signals...")
            # For now, generate using generator with different parameters
            # In production, would use PyCBC or similar
            binary_data = self.generator.generate_training_dataset(
                num_signals=num_binary,
                signal_duration=signal_duration,
                include_noise_only=False
            )
            
            # Use generated signals as binary approximation
            binary_signals = binary_data['data'][binary_data['labels'] == 1][:num_binary]
            
            all_signals.append(binary_signals)
            all_labels.extend([2] * num_binary)  # Label 2: binary merger
        
        # ✅ COMBINE AND SHUFFLE
        if all_signals:
            combined_data = jnp.concatenate(all_signals, axis=0)
            combined_labels = jnp.array(all_labels)
            
            # Shuffle
            key = jax.random.PRNGKey(42)
            indices = jax.random.permutation(key, len(combined_data))
            
            shuffled_data = combined_data[indices]
            shuffled_labels = combined_labels[indices]
            
            # ✅ METADATA
            self.metadata.update({
                'total_samples': total_samples,
                'continuous_samples': num_continuous,
                'noise_samples': num_noise,
                'binary_samples': num_binary,
                'signal_duration': signal_duration,
                'data_shape': shuffled_data.shape,
                'label_distribution': {
                    'noise': int(jnp.sum(shuffled_labels == 0)),
                    'continuous': int(jnp.sum(shuffled_labels == 1)), 
                    'binary': int(jnp.sum(shuffled_labels == 2))
                }
            })
            
            logger.info(f"Mixed dataset built: {shuffled_data.shape}, "
                       f"labels: {self.metadata['label_distribution']}")
            
            return {
                'data': shuffled_data,
                'labels': shuffled_labels,
                'metadata': self.metadata
            }
        else:
            raise ValueError("No signals generated - check ratios and total_samples")
    
    def split_dataset(self, dataset: Dict[str, jnp.ndarray],
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15) -> Dict[str, Dict[str, jnp.ndarray]]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            dataset: Dataset dictionary with 'data' and 'labels'
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        data = dataset['data']
        labels = dataset['labels']
        total_samples = len(data)
        
        # Calculate split sizes
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        test_size = total_samples - train_size - val_size
        
        # ✅ STRATIFIED SPLIT: Maintain class balance
        unique_labels = jnp.unique(labels)
        train_data, train_labels = [], []
        val_data, val_labels = [], []
        test_data, test_labels = [], []
        
        for label in unique_labels:
            # Get indices for this class
            class_indices = jnp.where(labels == label)[0]
            num_class_samples = len(class_indices)
            
            # Calculate splits for this class
            class_train_size = int(num_class_samples * train_ratio)
            class_val_size = int(num_class_samples * val_ratio)
            class_test_size = num_class_samples - class_train_size - class_val_size
            
            # Shuffle class indices
            key = jax.random.PRNGKey(42 + int(label))
            shuffled_indices = jax.random.permutation(key, class_indices)
            
            # Split indices
            train_indices = shuffled_indices[:class_train_size]
            val_indices = shuffled_indices[class_train_size:class_train_size + class_val_size]
            test_indices = shuffled_indices[class_train_size + class_val_size:]
            
            # Add to splits
            train_data.append(data[train_indices])
            train_labels.append(labels[train_indices])
            val_data.append(data[val_indices])
            val_labels.append(labels[val_indices])
            test_data.append(data[test_indices])
            test_labels.append(labels[test_indices])
        
        # Combine and shuffle each split
        def combine_and_shuffle(data_list, labels_list, split_key):
            if data_list:
                combined_data = jnp.concatenate(data_list, axis=0)
                combined_labels = jnp.concatenate(labels_list, axis=0)
                
                # Final shuffle
                indices = jax.random.permutation(split_key, len(combined_data))
                return combined_data[indices], combined_labels[indices]
            else:
                return jnp.array([]), jnp.array([])
        
        # Create splits with shuffling
        train_key, val_key, test_key = jax.random.split(jax.random.PRNGKey(42), 3)
        
        final_train_data, final_train_labels = combine_and_shuffle(train_data, train_labels, train_key)
        final_val_data, final_val_labels = combine_and_shuffle(val_data, val_labels, val_key)
        final_test_data, final_test_labels = combine_and_shuffle(test_data, test_labels, test_key)
        
        splits = {
            'train': {
                'data': final_train_data,
                'labels': final_train_labels
            },
            'val': {
                'data': final_val_data,
                'labels': final_val_labels
            },
            'test': {
                'data': final_test_data,
                'labels': final_test_labels
            }
        }
        
        # Log split statistics
        for split_name, split_data in splits.items():
            if len(split_data['data']) > 0:
                unique_labels, counts = jnp.unique(split_data['labels'], return_counts=True)
                label_dist = {int(label): int(count) for label, count in zip(unique_labels, counts)}
                logger.info(f"{split_name} split: {len(split_data['data'])} samples, "
                           f"distribution: {label_dist}")
        
        return splits
    
    def export_dataset(self, dataset: Dict[str, jnp.ndarray],
                      output_path: Union[str, Path],
                      format: str = "hdf5") -> Dict[str, Any]:
        """
        Export dataset to specified format.
        
        Args:
            dataset: Dataset dictionary to export
            output_path: Output file path
            format: Export format ("hdf5", "tfrecord", "numpy")
            
        Returns:
            Export information dictionary
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "hdf5":
            return self._export_hdf5(dataset, output_path)
        elif format == "tfrecord":
            return self._export_tfrecord(dataset, output_path)
        elif format == "numpy":
            return self._export_numpy(dataset, output_path)
        else:
            raise ValueError(f"Unknown export format: {format}")
    
    def _export_hdf5(self, dataset: Dict[str, jnp.ndarray], output_path: Path) -> Dict[str, Any]:
        """Export dataset to HDF5 format."""
        if not HDF5_AVAILABLE:
            raise ImportError("h5py not available for HDF5 export")
        
        with h5py.File(output_path, 'w') as f:
            # Main data
            f.create_dataset('data', data=np.array(dataset['data']))
            f.create_dataset('labels', data=np.array(dataset['labels']))
            
            # Metadata
            if 'metadata' in dataset:
                metadata_group = f.create_group('metadata')
                for key, value in dataset['metadata'].items():
                    if isinstance(value, dict):
                        subgroup = metadata_group.create_group(key)
                        for subkey, subvalue in value.items():
                            subgroup.attrs[subkey] = subvalue
                    else:
                        metadata_group.attrs[key] = value
        
        file_size = output_path.stat().st_size
        logger.info(f"HDF5 export completed: {output_path} ({file_size/1024/1024:.1f} MB)")
        
        return {
            'format': 'hdf5',
            'path': str(output_path),
            'size_mb': file_size / 1024 / 1024,
            'samples': len(dataset['data'])
        }
    
    def _export_numpy(self, dataset: Dict[str, jnp.ndarray], output_path: Path) -> Dict[str, Any]:
        """Export dataset to NumPy .npz format."""
        # Convert JAX arrays to NumPy for saving
        save_dict = {
            'data': np.array(dataset['data']),
            'labels': np.array(dataset['labels'])
        }
        
        # Add metadata if available
        if 'metadata' in dataset:
            save_dict['metadata'] = dataset['metadata']
        
        np.savez_compressed(output_path.with_suffix('.npz'), **save_dict)
        
        file_size = output_path.with_suffix('.npz').stat().st_size
        logger.info(f"NumPy export completed: {output_path.with_suffix('.npz')} "
                   f"({file_size/1024/1024:.1f} MB)")
        
        return {
            'format': 'numpy',
            'path': str(output_path.with_suffix('.npz')),
            'size_mb': file_size / 1024 / 1024,
            'samples': len(dataset['data'])
        }
    
    def _export_tfrecord(self, dataset: Dict[str, jnp.ndarray], output_path: Path) -> Dict[str, Any]:
        """Export dataset to TensorFlow TFRecord format."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available for TFRecord export")
        
        with tf.io.TFRecordWriter(str(output_path.with_suffix('.tfrecord'))) as writer:
            data = np.array(dataset['data'])
            labels = np.array(dataset['labels'])
            
            for i in range(len(data)):
                # Create TensorFlow example
                example = tf.train.Example(features=tf.train.Features(feature={
                    'data': tf.train.Feature(
                        float_list=tf.train.FloatList(value=data[i].flatten())
                    ),
                    'label': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[labels[i]])
                    ),
                    'shape': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=data[i].shape)
                    )
                }))
                
                writer.write(example.SerializeToString())
        
        file_size = output_path.with_suffix('.tfrecord').stat().st_size
        logger.info(f"TFRecord export completed: {output_path.with_suffix('.tfrecord')} "
                   f"({file_size/1024/1024:.1f} MB)")
        
        return {
            'format': 'tfrecord',
            'path': str(output_path.with_suffix('.tfrecord')),
            'size_mb': file_size / 1024 / 1024,
            'samples': len(dataset['data'])
        }
    
    def get_dataset_statistics(self, dataset: Dict[str, jnp.ndarray]) -> Dict[str, Any]:
        """Calculate comprehensive dataset statistics."""
        data = dataset['data']
        labels = dataset['labels']
        
        # Basic statistics
        stats = {
            'total_samples': len(data),
            'data_shape': data.shape,
            'data_dtype': str(data.dtype),
            'label_dtype': str(labels.dtype)
        }
        
        # Label distribution
        unique_labels, counts = jnp.unique(labels, return_counts=True)
        label_distribution = {}
        label_names = {0: 'noise', 1: 'continuous_gw', 2: 'binary_merger'}
        
        for label, count in zip(unique_labels, counts):
            label_name = label_names.get(int(label), f'class_{int(label)}')
            label_distribution[label_name] = {
                'count': int(count),
                'fraction': float(count / len(labels))
            }
        
        stats['label_distribution'] = label_distribution
        
        # Data quality metrics
        stats['data_quality'] = {
            'mean_amplitude': float(jnp.mean(jnp.abs(data))),
            'std_amplitude': float(jnp.std(data)),
            'dynamic_range': float(jnp.max(data) - jnp.min(data)),
            'has_nan': bool(jnp.any(jnp.isnan(data))),
            'has_inf': bool(jnp.any(jnp.isinf(data)))
        }
        
        return stats


# Export builder class
__all__ = [
    "GWDatasetBuilder"
]

