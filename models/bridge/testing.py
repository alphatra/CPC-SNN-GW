"""
Testing utilities for spike bridge components.

This module contains test functions extracted from spike_bridge.py
for gradient flow validation and bridge component testing.

Split from spike_bridge.py for better maintainability.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple
import logging

from .gradients import GradientFlowMonitor

logger = logging.getLogger(__name__)


def test_gradient_flow(spike_bridge,
                      input_shape: Tuple[int, ...],
                      key: jax.random.PRNGKey) -> Dict[str, Any]:
    """
    Test end-to-end gradient flow through spike bridge.
    Executive Summary requirement: gradient flow validation.
    
    Args:
        spike_bridge: Spike bridge instance
        input_shape: Input tensor shape  
        key: Random key for test data
        
    Returns:
        Test results and diagnostics
    """
    logger.info("Testing gradient flow through spike bridge")
    
    try:
        # Initialize spike bridge
        test_input = jax.random.normal(key, input_shape)
        variables = spike_bridge.init(key, test_input, training=True)
        
        # Define loss function for testing
        def test_loss_fn(params, input_data):
            spikes = spike_bridge.apply(params, input_data, training=True)
            # Simple loss: encourage moderate spike rate
            target_rate = 0.1
            actual_rate = jnp.mean(spikes)
            return (actual_rate - target_rate)**2
        
        # Compute gradients
        loss_value, gradients = jax.value_and_grad(test_loss_fn)(variables, test_input)
        
        # Check gradient flow
        monitor = GradientFlowMonitor()
        gradient_stats = monitor.check_gradient_flow(variables, gradients)
        
        # Test results
        results = {
            'test_passed': gradient_stats['healthy_flow'],
            'loss_value': float(loss_value),
            'gradient_norm': gradient_stats['gradient_norm'],
            'gradient_to_param_ratio': gradient_stats['gradient_to_param_ratio'],
            'vanishing_gradients': gradient_stats['vanishing_gradients'],
            'exploding_gradients': gradient_stats['exploding_gradients'],
            'spike_rate': float(jnp.mean(spike_bridge.apply(variables, test_input, training=True)))
        }
        
        if results['test_passed']:
            logger.info(f"✅ Gradient flow test PASSED - ratio: {results['gradient_to_param_ratio']:.2e}")
        else:
            logger.error(f"❌ Gradient flow test FAILED - check gradient statistics")
            
        return results
        
    except Exception as e:
        logger.error(f"Gradient flow test failed with exception: {e}")
        return {
            'test_passed': False,
            'error': str(e)
        }


def test_encoding_quality(encoder, signal: jnp.ndarray, time_steps: int = 16) -> Dict[str, Any]:
    """
    Test encoding quality for any spike encoder.
    
    Args:
        encoder: Spike encoder instance
        signal: Test signal
        time_steps: Number of time steps
        
    Returns:
        Encoding quality metrics
    """
    try:
        # Generate spikes
        spikes = encoder.encode(signal, time_steps) if hasattr(encoder, 'encode') else encoder(signal, time_steps)
        
        # Calculate quality metrics
        spike_rate = float(jnp.mean(spikes))
        spike_variance = float(jnp.var(spikes))
        temporal_consistency = float(jnp.std(jnp.mean(spikes, axis=(0, 2))))  # Variance across time
        
        # Check for reasonable spike rate
        rate_ok = 0.01 <= spike_rate <= 0.5  # Reasonable spike rate range
        
        # Check for temporal variability
        temporal_ok = spike_variance > 1e-6  # Not all zeros
        
        results = {
            'spike_rate': spike_rate,
            'spike_variance': spike_variance,
            'temporal_consistency': temporal_consistency,
            'rate_ok': rate_ok,
            'temporal_ok': temporal_ok,
            'encoding_quality_passed': rate_ok and temporal_ok
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Encoding quality test failed: {e}")
        return {
            'encoding_quality_passed': False,
            'error': str(e)
        }


def test_spike_bridge_comprehensive(spike_bridge_factory, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """
    Comprehensive test suite for spike bridge components.
    
    Args:
        spike_bridge_factory: Factory function for creating spike bridge
        input_shape: Input tensor shape
        
    Returns:
        Comprehensive test results
    """
    logger.info("Running comprehensive spike bridge test suite")
    
    key = jax.random.PRNGKey(42)
    test_results = {}
    
    # Test different encoding strategies
    encoding_strategies = ["temporal_contrast", "phase_preserving", "learnable_multi_threshold"]
    
    for encoding in encoding_strategies:
        try:
            # Create spike bridge with specific encoding
            spike_bridge = spike_bridge_factory(
                spike_encoding=encoding,
                time_steps=16,
                threshold=0.1
            )
            
            # Test gradient flow
            gradient_results = test_gradient_flow(spike_bridge, input_shape, key)
            
            # Test encoding quality
            test_signal = jax.random.normal(jax.random.split(key)[0], input_shape)
            quality_results = test_encoding_quality(spike_bridge, test_signal)
            
            test_results[encoding] = {
                'gradient_flow': gradient_results,
                'encoding_quality': quality_results,
                'overall_passed': gradient_results['test_passed'] and quality_results['encoding_quality_passed']
            }
            
        except Exception as e:
            logger.error(f"Test failed for {encoding}: {e}")
            test_results[encoding] = {
                'overall_passed': False,
                'error': str(e)
            }
    
    # Summary
    passed_encodings = [enc for enc, results in test_results.items() if results.get('overall_passed', False)]
    
    test_results['summary'] = {
        'total_encodings_tested': len(encoding_strategies),
        'passed_encodings': len(passed_encodings),
        'failed_encodings': len(encoding_strategies) - len(passed_encodings),
        'all_tests_passed': len(passed_encodings) == len(encoding_strategies)
    }
    
    logger.info(f"Comprehensive test results: {len(passed_encodings)}/{len(encoding_strategies)} encodings passed")
    
    return test_results


# Export test functions
__all__ = [
    "test_gradient_flow",
    "test_encoding_quality", 
    "test_spike_bridge_comprehensive"
]

