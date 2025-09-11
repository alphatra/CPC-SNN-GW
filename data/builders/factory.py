"""
Factory functions for dataset creation.

This module contains dataset creation factory functions extracted from
gw_dataset_builder.py for better modularity.

Split from gw_dataset_builder.py for better maintainability.
"""

import logging
from typing import Dict, Optional, Union
import jax
import jax.numpy as jnp

from .core import GWDatasetBuilder
from ..gw_synthetic_generator import ContinuousGWGenerator
from ..gw_signal_params import GeneratorSettings

logger = logging.getLogger(__name__)


def create_mixed_gw_dataset(continuous_generator: ContinuousGWGenerator,
                          binary_data: Optional[Dict] = None,
                          mix_ratio: float = 0.5,
                          total_samples: int = 1000,
                          signal_duration: float = 4.0) -> Dict[str, jnp.ndarray]:
    """
    Create mixed gravitational wave dataset with multiple signal types.
    
    Args:
        continuous_generator: Generator for continuous GW signals
        binary_data: Optional pre-generated binary merger data
        mix_ratio: Ratio of GW signals to noise (0.0 = all noise, 1.0 = all GW)
        total_samples: Total number of samples to generate
        signal_duration: Duration of each signal in seconds
        
    Returns:
        Dictionary containing mixed dataset with data and labels
    """
    logger.info(f"Creating mixed GW dataset: {total_samples} samples, mix_ratio={mix_ratio}")
    
    # Create dataset builder
    builder = GWDatasetBuilder(continuous_generator)
    
    # Calculate signal type ratios
    if binary_data is not None:
        # 3-class dataset: noise, continuous, binary
        noise_ratio = (1.0 - mix_ratio) * 0.8  # 80% of non-GW is noise
        continuous_ratio = mix_ratio * 0.7     # 70% of GW is continuous  
        binary_ratio = mix_ratio * 0.3 + (1.0 - mix_ratio) * 0.2  # Rest is binary
    else:
        # 2-class dataset: noise, continuous  
        noise_ratio = 1.0 - mix_ratio
        continuous_ratio = mix_ratio
        binary_ratio = 0.0
    
    # Build mixed dataset
    dataset = builder.build_mixed_dataset(
        total_samples=total_samples,
        continuous_ratio=continuous_ratio,
        noise_ratio=noise_ratio,
        binary_ratio=binary_ratio,
        signal_duration=signal_duration
    )
    
    logger.info(f"Mixed dataset created successfully: {dataset['data'].shape}")
    
    return dataset


def create_evaluation_dataset(num_samples: int = 1000,
                            sequence_length: int = 16384,
                            sample_rate: int = 4096,
                            snr_range: Tuple[float, float] = (8.0, 25.0),
                            include_glitches: bool = True) -> Dict[str, jnp.ndarray]:
    """
    Create evaluation dataset with controlled parameters for testing.
    
    Args:
        num_samples: Number of evaluation samples
        sequence_length: Length of each sequence (in samples)
        sample_rate: Sampling rate in Hz
        snr_range: Range of signal-to-noise ratios
        include_glitches: Whether to include glitch simulations
        
    Returns:
        Evaluation dataset with known ground truth
    """
    logger.info(f"Creating evaluation dataset: {num_samples} samples")
    
    # Create generator with evaluation settings
    settings = GeneratorSettings(
        base_frequency=50.0,
        freq_range=(20.0, 500.0),
        duration=sequence_length / sample_rate,
        sample_rate=sample_rate
    )
    
    generator = ContinuousGWGenerator(config=settings)
    builder = GWDatasetBuilder(generator, settings)
    
    # Generate balanced evaluation set
    eval_dataset = builder.build_mixed_dataset(
        total_samples=num_samples,
        continuous_ratio=0.3,  # 30% GW signals
        noise_ratio=0.7,       # 70% noise
        binary_ratio=0.0,      # No binary for simplicity
        signal_duration=sequence_length / sample_rate
    )
    
    # ✅ QUALITY CONTROL: Add SNR control and glitch injection
    if include_glitches:
        eval_dataset = _add_controlled_glitches(eval_dataset)
    
    eval_dataset = _control_snr_range(eval_dataset, snr_range)
    
    logger.info(f"Evaluation dataset created: {eval_dataset['data'].shape}")
    
    return eval_dataset


def create_training_dataset(continuous_generator: ContinuousGWGenerator,
                          num_train: int = 8000,
                          num_val: int = 1000,
                          num_test: int = 1000,
                          signal_duration: float = 4.0) -> Dict[str, Dict[str, jnp.ndarray]]:
    """
    Create complete training dataset with train/val/test splits.
    
    Args:
        continuous_generator: Generator for GW signals
        num_train: Number of training samples
        num_val: Number of validation samples  
        num_test: Number of test samples
        signal_duration: Duration of each signal
        
    Returns:
        Dictionary with train/val/test splits
    """
    total_samples = num_train + num_val + num_test
    logger.info(f"Creating training dataset: {total_samples} total samples")
    
    # Create large mixed dataset
    full_dataset = create_mixed_gw_dataset(
        continuous_generator=continuous_generator,
        total_samples=total_samples,
        mix_ratio=0.5,  # Balanced dataset
        signal_duration=signal_duration
    )
    
    # Calculate split ratios
    train_ratio = num_train / total_samples
    val_ratio = num_val / total_samples
    test_ratio = num_test / total_samples
    
    # Create builder and split
    builder = GWDatasetBuilder(continuous_generator)
    splits = builder.split_dataset(
        full_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    logger.info(f"Training dataset created with splits: "
               f"train={len(splits['train']['data'])}, "
               f"val={len(splits['val']['data'])}, "
               f"test={len(splits['test']['data'])}")
    
    return splits


def _add_controlled_glitches(dataset: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Add controlled glitches to evaluation dataset."""
    # Simple glitch injection for testing
    data = dataset['data']
    labels = dataset['labels']
    
    # Add glitches to 10% of samples
    num_glitch_samples = int(len(data) * 0.1)
    glitch_indices = jax.random.choice(
        jax.random.PRNGKey(42),
        len(data),
        shape=(num_glitch_samples,),
        replace=False
    )
    
    # Simple glitch: add impulse noise
    glitched_data = data.copy()
    for idx in glitch_indices:
        # Add random impulse
        glitch_pos = jax.random.randint(jax.random.PRNGKey(idx), (), 0, len(data[idx]) - 100)
        glitch_amplitude = jax.random.uniform(jax.random.PRNGKey(idx + 1000), (), minval=5.0, maxval=20.0)
        
        # Create impulse glitch
        glitch = jax.random.normal(jax.random.PRNGKey(idx + 2000), (100,)) * glitch_amplitude
        glitched_data = glitched_data.at[idx, glitch_pos:glitch_pos+100].add(glitch)
    
    return {
        'data': glitched_data,
        'labels': labels,
        'metadata': {**dataset.get('metadata', {}), 'glitches_added': num_glitch_samples}
    }


def _control_snr_range(dataset: Dict[str, jnp.ndarray], 
                      snr_range: Tuple[float, float]) -> Dict[str, jnp.ndarray]:
    """Control SNR range of signals in dataset."""
    data = dataset['data']
    labels = dataset['labels']
    
    # Apply SNR control only to GW signals (label == 1)
    gw_mask = labels == 1
    
    if jnp.sum(gw_mask) > 0:
        gw_data = data[gw_mask]
        
        # Scale signals to desired SNR range
        target_snrs = jax.random.uniform(
            jax.random.PRNGKey(42),
            (jnp.sum(gw_mask),),
            minval=snr_range[0],
            maxval=snr_range[1]
        )
        
        # Normalize and scale each signal
        scaled_gw_data = []
        for i, target_snr in enumerate(target_snrs):
            signal = gw_data[i]
            
            # Estimate current SNR (simplified)
            signal_power = jnp.var(signal)
            noise_estimate = jnp.var(signal) * 0.1  # Assume 10% noise
            current_snr = signal_power / noise_estimate
            
            # Scale to target SNR
            scale_factor = jnp.sqrt(target_snr / (current_snr + 1e-10))
            scaled_signal = signal * scale_factor
            
            scaled_gw_data.append(scaled_signal)
        
        scaled_gw_data = jnp.stack(scaled_gw_data)
        
        # Replace GW signals in dataset
        modified_data = data.at[gw_mask].set(scaled_gw_data)
        
        return {
            'data': modified_data,
            'labels': labels,
            'metadata': {**dataset.get('metadata', {}), 'snr_controlled': True, 'snr_range': snr_range}
        }
    
    return dataset


# Export factory functions
__all__ = [
    "create_mixed_gw_dataset",
    "create_evaluation_dataset",
    "create_training_dataset"
]

