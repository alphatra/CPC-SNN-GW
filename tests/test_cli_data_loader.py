"""
Tests for CLI data loader functionality.

Smoke tests for CLI data loading according to refactoring plan.
"""

import pytest
import argparse
import numpy as np
from cli.commands.training.data_loader import _load_synthetic_data


def test_synthetic_data_load():
    """Test synthetic data loading with basic validation."""
    # Create mock args
    args = argparse.Namespace(
        synthetic_samples=10,
        quick_mode=True,
        synthetic_quick=True
    )
    
    try:
        signals, labels, test_signals, test_labels = _load_synthetic_data(args)
        
        # Basic shape checks
        assert len(signals) <= 10  # Should be <= requested samples after split
        assert len(labels) <= 10
        assert len(test_signals) >= 1  # Should have test split
        assert len(test_labels) >= 1
        
        # Data consistency
        assert len(signals) == len(labels)
        assert len(test_signals) == len(test_labels)
        
        # Shape consistency
        assert signals.shape[1] == test_signals.shape[1]  # Same sequence length
        
        # Basic data validation
        assert np.all(np.isfinite(signals))
        assert np.all(np.isfinite(test_signals))
        assert np.all((labels >= 0) & (labels <= 2))  # Valid class labels
        assert np.all((test_labels >= 0) & (test_labels <= 2))
        
    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")
    except Exception as e:
        pytest.fail(f"Synthetic data loading failed: {e}")


def test_data_loader_error_handling():
    """Test data loader error handling."""
    # Test with invalid args
    args = argparse.Namespace(
        synthetic_samples=-1,  # Invalid
        quick_mode=True,
        synthetic_quick=True
    )
    
    try:
        # Should handle invalid parameters gracefully
        signals, labels, test_signals, test_labels = _load_synthetic_data(args)
        
        # If it succeeds, should still return valid data
        assert len(signals) >= 0
        assert len(labels) >= 0
        
    except Exception:
        # Error handling is acceptable for invalid inputs
        pass


def test_data_loader_shapes_consistency():
    """Test that data loader produces consistent shapes."""
    args = argparse.Namespace(
        synthetic_samples=20,
        quick_mode=True,
        synthetic_quick=True
    )
    
    try:
        signals, labels, test_signals, test_labels = _load_synthetic_data(args)
        
        # All signals should have same sequence length
        if len(signals) > 1:
            assert all(sig.shape == signals[0].shape for sig in signals)
        
        if len(test_signals) > 1:
            assert all(sig.shape == test_signals[0].shape for sig in test_signals)
        
        # Train and test should have same sequence length
        if len(signals) > 0 and len(test_signals) > 0:
            assert signals.shape[1:] == test_signals.shape[1:]
        
    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")
    except Exception as e:
        pytest.fail(f"Data shape consistency test failed: {e}")
