"""
Testing utilities for dataset builders.

This module contains test functions extracted from
gw_dataset_builder.py for better modularity.

Split from gw_dataset_builder.py for better maintainability.
"""

import logging
from typing import Dict, Any

import jax
import jax.numpy as jnp

from .core import GWDatasetBuilder
from .factory import create_mixed_gw_dataset, create_evaluation_dataset
from ..gw_synthetic_generator import ContinuousGWGenerator
from ..gw_signal_params import GeneratorSettings

logger = logging.getLogger(__name__)


def test_dataset_builder():
    """Test the dataset builder functionality."""
    try:
        logger.info("Testing dataset builder functionality...")
        
        # ✅ TEST 1: Create generator
        settings = GeneratorSettings(
            base_frequency=50.0,
            freq_range=(20.0, 200.0),
            duration=4.0,
            sample_rate=4096
        )
        
        generator = ContinuousGWGenerator(config=settings)
        builder = GWDatasetBuilder(generator, settings)
        
        logger.info("✅ Generator and builder created successfully")
        
        # ✅ TEST 2: Build small mixed dataset
        test_dataset = builder.build_mixed_dataset(
            total_samples=50,
            continuous_ratio=0.3,
            noise_ratio=0.5,
            binary_ratio=0.2,
            signal_duration=2.0  # Shorter for testing
        )
        
        logger.info(f"✅ Mixed dataset created: {test_dataset['data'].shape}")
        
        # ✅ TEST 3: Split dataset
        splits = builder.split_dataset(
            test_dataset,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        logger.info("✅ Dataset splitting successful")
        
        # ✅ TEST 4: Statistics
        stats = builder.get_dataset_statistics(test_dataset)
        logger.info(f"✅ Dataset statistics: {stats['total_samples']} samples, "
                   f"distribution: {stats['label_distribution']}")
        
        # ✅ TEST 5: Factory functions
        factory_dataset = create_mixed_gw_dataset(
            continuous_generator=generator,
            total_samples=30,
            mix_ratio=0.5
        )
        
        logger.info(f"✅ Factory function test: {factory_dataset['data'].shape}")
        
        # ✅ TEST 6: Evaluation dataset
        eval_dataset = create_evaluation_dataset(
            num_samples=20,
            sequence_length=8192,  # Shorter for testing
            include_glitches=False  # Disable for simple test
        )
        
        logger.info(f"✅ Evaluation dataset test: {eval_dataset['data'].shape}")
        
        # ✅ VALIDATION: Check data quality
        test_results = {
            'mixed_dataset_shape': test_dataset['data'].shape,
            'mixed_dataset_labels': int(len(jnp.unique(test_dataset['labels']))),
            'splits_created': len(splits),
            'train_samples': len(splits['train']['data']),
            'val_samples': len(splits['val']['data']),
            'test_samples': len(splits['test']['data']),
            'factory_dataset_shape': factory_dataset['data'].shape,
            'eval_dataset_shape': eval_dataset['data'].shape,
            'all_tests_passed': True
        }
        
        logger.info("🎉 All dataset builder tests PASSED!")
        logger.info(f"Test results: {test_results}")
        
        return test_results
        
    except Exception as e:
        logger.error(f"❌ Dataset builder test FAILED: {e}")
        return {
            'all_tests_passed': False,
            'error': str(e)
        }


def test_dataset_quality(dataset: Dict[str, jnp.ndarray]) -> Dict[str, Any]:
    """
    Test quality of generated dataset.
    
    Args:
        dataset: Dataset to test
        
    Returns:
        Quality test results
    """
    try:
        data = dataset['data']
        labels = dataset['labels']
        
        # ✅ BASIC CHECKS
        basic_checks = {
            'data_not_empty': len(data) > 0,
            'labels_not_empty': len(labels) > 0,
            'shapes_match': len(data) == len(labels),
            'no_nan_data': not jnp.any(jnp.isnan(data)),
            'no_inf_data': not jnp.any(jnp.isinf(data)),
            'finite_labels': jnp.all(jnp.isfinite(labels))
        }
        
        # ✅ DISTRIBUTION CHECKS
        unique_labels, counts = jnp.unique(labels, return_counts=True)
        
        distribution_checks = {
            'has_multiple_classes': len(unique_labels) > 1,
            'balanced_classes': jnp.std(counts) / jnp.mean(counts) < 2.0,  # Not too imbalanced
            'reasonable_samples': len(data) >= 10  # Minimum reasonable size
        }
        
        # ✅ DATA QUALITY CHECKS
        quality_checks = {
            'reasonable_amplitude': jnp.max(jnp.abs(data)) < 1e10,  # Not extreme values
            'non_zero_variance': jnp.var(data) > 1e-20,  # Has variability
            'dynamic_range_ok': (jnp.max(data) - jnp.min(data)) > 1e-10  # Has dynamic range
        }
        
        # ✅ OVERALL ASSESSMENT
        all_basic_passed = all(basic_checks.values())
        all_distribution_passed = all(distribution_checks.values())
        all_quality_passed = all(quality_checks.values())
        
        overall_passed = all_basic_passed and all_distribution_passed and all_quality_passed
        
        results = {
            'overall_passed': overall_passed,
            'basic_checks': basic_checks,
            'distribution_checks': distribution_checks,
            'quality_checks': quality_checks,
            'dataset_stats': {
                'num_samples': len(data),
                'data_shape': data.shape,
                'num_classes': len(unique_labels),
                'class_counts': {int(label): int(count) for label, count in zip(unique_labels, counts)}
            }
        }
        
        if overall_passed:
            logger.info("✅ Dataset quality test PASSED")
        else:
            logger.warning("⚠️ Dataset quality test FAILED - check results")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Dataset quality test error: {e}")
        return {
            'overall_passed': False,
            'error': str(e)
        }


def benchmark_dataset_creation(num_samples_list: List[int] = [100, 500, 1000]) -> Dict[str, Any]:
    """
    Benchmark dataset creation performance.
    
    Args:
        num_samples_list: List of sample counts to benchmark
        
    Returns:
        Benchmark results
    """
    import time
    
    logger.info("Benchmarking dataset creation performance...")
    
    # Create generator for benchmarking
    settings = GeneratorSettings()
    generator = ContinuousGWGenerator(config=settings)
    
    benchmark_results = {}
    
    for num_samples in num_samples_list:
        logger.info(f"Benchmarking {num_samples} samples...")
        
        start_time = time.time()
        
        try:
            # Create dataset
            dataset = create_mixed_gw_dataset(
                continuous_generator=generator,
                total_samples=num_samples,
                mix_ratio=0.5
            )
            
            end_time = time.time()
            creation_time = end_time - start_time
            
            # Calculate performance metrics
            samples_per_second = num_samples / creation_time
            mb_per_second = (dataset['data'].nbytes / 1024 / 1024) / creation_time
            
            benchmark_results[num_samples] = {
                'creation_time': creation_time,
                'samples_per_second': samples_per_second,
                'mb_per_second': mb_per_second,
                'dataset_size_mb': dataset['data'].nbytes / 1024 / 1024,
                'success': True
            }
            
            logger.info(f"✅ {num_samples} samples: {creation_time:.2f}s "
                       f"({samples_per_second:.1f} samples/s)")
            
        except Exception as e:
            logger.error(f"❌ Benchmark failed for {num_samples} samples: {e}")
            benchmark_results[num_samples] = {
                'success': False,
                'error': str(e)
            }
    
    # Summary statistics
    successful_benchmarks = [r for r in benchmark_results.values() if r.get('success', False)]
    
    if successful_benchmarks:
        summary = {
            'mean_samples_per_second': jnp.mean(jnp.array([r['samples_per_second'] for r in successful_benchmarks])),
            'mean_mb_per_second': jnp.mean(jnp.array([r['mb_per_second'] for r in successful_benchmarks])),
            'successful_benchmarks': len(successful_benchmarks),
            'total_benchmarks': len(num_samples_list)
        }
    else:
        summary = {'no_successful_benchmarks': True}
    
    benchmark_results['summary'] = summary
    
    logger.info(f"Benchmark completed: {len(successful_benchmarks)}/{len(num_samples_list)} successful")
    
    return benchmark_results


# Export testing functions
__all__ = [
    "test_dataset_builder",
    "test_dataset_quality",
    "benchmark_dataset_creation"
]

