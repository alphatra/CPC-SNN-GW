"""
Label Validation and Error Handling

Advanced validation system for gravitational wave detection labels with:
- Typed error classes with detailed reporting
- JAX-optimized validation functions
- Comprehensive dataset validation
- Detailed error reporting with line numbers and context

Features:
- Fast JAX-compiled validation functions
- Comprehensive error reporting
- Dataset-level validation
- Legacy label conversion
- Detailed validation results
"""

from typing import Dict, List, Union, Optional, Tuple, Any
import jax
import jax.numpy as jnp
import numpy as np
import logging
from dataclasses import dataclass

from .label_enums import (
    GWSignalType, 
    CANONICAL_LABELS,
    VALID_LABEL_VALUES,
    MAX_LABEL_VALUE,
    MIN_LABEL_VALUE,
    NUM_CLASSES,
    is_valid_label,
    normalize_label_value
)

logger = logging.getLogger(__name__)


class LabelError(Exception):
    """
    Enhanced exception for label-related errors with detailed context.
    
    Provides line numbers, error context, and suggested corrections
    for debugging and error reporting.
    """
    
    def __init__(self, 
                 message: str,
                 line_number: Optional[int] = None,
                 row_number: Optional[int] = None,
                 invalid_value: Any = None,
                 context: Optional[Dict[str, Any]] = None,
                 suggestions: Optional[List[str]] = None):
        """
        Initialize enhanced label error.
        
        Args:
            message: Error description
            line_number: Line number where error occurred (for file processing)
            row_number: Row number in dataset (for array processing)
            invalid_value: The invalid value that caused the error
            context: Additional context information
            suggestions: Suggested corrections
        """
        self.line_number = line_number
        self.row_number = row_number
        self.invalid_value = invalid_value
        self.context = context or {}
        self.suggestions = suggestions or []
        
        # Build detailed error message
        enhanced_message = message
        
        if line_number is not None:
            enhanced_message += f" (line {line_number})"
        if row_number is not None:
            enhanced_message += f" (row {row_number})"
        if invalid_value is not None:
            enhanced_message += f" - invalid value: {invalid_value}"
            
        if suggestions:
            enhanced_message += f"\nSuggestions: {', '.join(suggestions)}"
            
        super().__init__(enhanced_message)


@dataclass
class LabelValidationResult:
    """
    Comprehensive validation result with detailed metrics and error reporting.
    
    Contains all information needed for debugging and quality assessment.
    """
    is_valid: bool
    total_labels: int
    valid_labels: int
    invalid_labels: int
    invalid_indices: List[int]
    invalid_values: List[Any]
    class_distribution: Dict[int, int]
    error_messages: List[str]
    suggestions: List[str]
    
    @property
    def validity_rate(self) -> float:
        """Calculate percentage of valid labels."""
        return self.valid_labels / self.total_labels if self.total_labels > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        """Calculate percentage of invalid labels."""
        return self.invalid_labels / self.total_labels if self.total_labels > 0 else 0.0


# JIT-compiled helper functions for fast validation
@jax.jit
def _fast_validity_check(labels: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Fast JAX-compiled validity checking for label arrays.
    
    Args:
        labels: Array of integer labels
        
    Returns:
        Dictionary with validity statistics
    """
    # Check for NaN/infinite values
    is_finite = jnp.isfinite(labels)
    
    # Check for valid range
    in_range = (labels >= MIN_LABEL_VALUE) & (labels <= MAX_LABEL_VALUE)
    
    # Check for integer values (detect floating point labels)
    is_integer = jnp.equal(labels, jnp.round(labels))
    
    # Combine all validity checks
    is_valid = is_finite & in_range & is_integer
    
    return {
        'is_valid': is_valid,
        'is_finite': is_finite,
        'in_range': in_range,
        'is_integer': is_integer,
        'num_valid': jnp.sum(is_valid),
        'num_invalid': jnp.sum(~is_valid),
        'total': len(labels)
    }


@jax.jit  
def _fast_class_distribution(labels: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    """
    Fast JAX-compiled class distribution calculation.
    
    Args:
        labels: Array of integer labels
        num_classes: Number of expected classes
        
    Returns:
        Array with count for each class
    """
    # Use JAX's efficient bincount with fixed length
    return jnp.bincount(labels, length=num_classes, minlength=num_classes)


def validate_labels(labels: jnp.ndarray,
                   expected_classes: Optional[List[int]] = None,
                   allow_missing_classes: bool = True,
                   strict_mode: bool = False) -> LabelValidationResult:
    """
    Comprehensive label validation with detailed error reporting.
    
    Args:
        labels: Array of labels to validate
        expected_classes: List of expected class values
        allow_missing_classes: Whether missing classes are allowed
        strict_mode: Enable strict validation (no warnings, only errors)
        
    Returns:
        Detailed validation result
        
    Raises:
        LabelError: If strict_mode=True and validation fails
    """
    if len(labels) == 0:
        result = LabelValidationResult(
            is_valid=True, total_labels=0, valid_labels=0, invalid_labels=0,
            invalid_indices=[], invalid_values=[], class_distribution={},
            error_messages=[], suggestions=[]
        )
        return result
    
    # Convert to JAX array if needed
    if isinstance(labels, (list, np.ndarray)):
        labels = jnp.array(labels)
    
    # Fast validity checking using JAX
    validity_stats = _fast_validity_check(labels)
    
    # Extract validity information
    is_valid_mask = validity_stats['is_valid']
    num_valid = int(validity_stats['num_valid'])
    num_invalid = int(validity_stats['num_invalid'])
    total = int(validity_stats['total'])
    
    # Find invalid indices and values
    invalid_indices = [int(i) for i in jnp.where(~is_valid_mask)[0]]
    invalid_values = [labels[i] for i in invalid_indices]
    
    # Class distribution for valid labels only
    valid_labels = labels[is_valid_mask]
    if len(valid_labels) > 0:
        class_counts = _fast_class_distribution(valid_labels, NUM_CLASSES)
        class_distribution = {int(i): int(count) for i, count in enumerate(class_counts)}
    else:
        class_distribution = {}
    
    # Error analysis
    error_messages = []
    suggestions = []
    
    if num_invalid > 0:
        # Analyze types of errors
        finite_mask = validity_stats['is_finite']
        range_mask = validity_stats['in_range'] 
        integer_mask = validity_stats['is_integer']
        
        # NaN/infinite values
        num_non_finite = total - int(jnp.sum(finite_mask))
        if num_non_finite > 0:
            error_messages.append(f"{num_non_finite} labels are NaN or infinite")
            suggestions.append("Check data preprocessing pipeline for NaN generation")
        
        # Out of range values
        num_out_range = total - int(jnp.sum(range_mask))
        if num_out_range > 0:
            error_messages.append(f"{num_out_range} labels are outside valid range [{MIN_LABEL_VALUE}, {MAX_LABEL_VALUE}]")
            suggestions.append(f"Ensure labels are in range [0, {NUM_CLASSES-1}]")
        
        # Non-integer values
        num_non_integer = total - int(jnp.sum(integer_mask))
        if num_non_integer > 0:
            error_messages.append(f"{num_non_integer} labels are not integers")
            suggestions.append("Convert floating point labels to integers or check data types")
    
    # Check expected classes
    if expected_classes is not None:
        present_classes = set(class_distribution.keys())
        expected_set = set(expected_classes)
        missing_classes = expected_set - present_classes
        unexpected_classes = present_classes - expected_set
        
        if missing_classes and not allow_missing_classes:
            error_messages.append(f"Missing expected classes: {sorted(missing_classes)}")
            suggestions.append("Check if dataset is properly balanced or if sampling is correct")
        
        if unexpected_classes:
            error_messages.append(f"Unexpected classes found: {sorted(unexpected_classes)}")
            suggestions.append("Verify label mappings and dataset preparation")
    
    # Check class balance
    if class_distribution:
        counts = list(class_distribution.values())
        max_count = max(counts)
        min_count = min(counts)
        
        if max_count > 0 and min_count / max_count < 0.1:  # 10:1 ratio threshold
            error_messages.append(f"Severe class imbalance detected (ratio {max_count/min_count:.1f}:1)")
            suggestions.append("Consider class balancing techniques or weighted loss functions")
    
    # Create result
    is_overall_valid = num_invalid == 0
    
    result = LabelValidationResult(
        is_valid=is_overall_valid,
        total_labels=total,
        valid_labels=num_valid,
        invalid_labels=num_invalid,
        invalid_indices=invalid_indices,
        invalid_values=invalid_values,
        class_distribution=class_distribution,
        error_messages=error_messages,
        suggestions=suggestions
    )
    
    # Strict mode handling
    if strict_mode and not is_overall_valid:
        raise LabelError(
            f"Label validation failed: {num_invalid}/{total} invalid labels",
            context={
                'invalid_indices': invalid_indices[:10],  # Show first 10
                'invalid_values': invalid_values[:10],
                'error_messages': error_messages
            },
            suggestions=suggestions
        )
    
    return result


def validate_dataset_labels(dataset: Dict[str, Any],
                          label_key: str = 'labels',
                          split_validation: bool = True) -> Dict[str, LabelValidationResult]:
    """
    Validate labels across multiple dataset splits.
    
    Args:
        dataset: Dataset dictionary with splits
        label_key: Key for labels in each split
        split_validation: Whether to validate each split separately
        
    Returns:
        Dictionary mapping split names to validation results
    """
    results = {}
    
    # Common splits to check
    common_splits = ['train', 'valid', 'test', 'validation']
    
    # Find available splits
    available_splits = []
    for split in common_splits:
        if split in dataset:
            available_splits.append(split)
    
    # If no common splits found, check all keys
    if not available_splits:
        available_splits = [k for k in dataset.keys() if isinstance(dataset[k], dict)]
    
    # Validate each split
    for split_name in available_splits:
        split_data = dataset[split_name]
        
        if isinstance(split_data, dict) and label_key in split_data:
            labels = split_data[label_key]
            
            try:
                result = validate_labels(labels, strict_mode=False)
                results[split_name] = result
                
                # Log validation results
                if result.is_valid:
                    logger.info(f"Split '{split_name}': {result.total_labels} labels valid")
                else:
                    logger.warning(f"Split '{split_name}': {result.invalid_labels}/{result.total_labels} invalid labels")
                    
            except Exception as e:
                logger.error(f"Failed to validate split '{split_name}': {e}")
                results[split_name] = LabelValidationResult(
                    is_valid=False, total_labels=0, valid_labels=0, invalid_labels=0,
                    invalid_indices=[], invalid_values=[], class_distribution={},
                    error_messages=[str(e)], suggestions=[]
                )
    
    return results


def convert_legacy_labels(labels: jnp.ndarray,
                         legacy_mapping: Optional[Dict] = None) -> Tuple[jnp.ndarray, List[str]]:
    """
    Convert legacy labels to current format with detailed logging.
    
    Args:
        labels: Array of legacy labels
        legacy_mapping: Custom mapping dictionary
        
    Returns:
        Tuple of (converted_labels, conversion_log)
    """
    conversion_log = []
    converted_labels = []
    
    # Use default legacy mapping if none provided
    if legacy_mapping is None:
        from .label_enums import LEGACY_LABEL_MAPPINGS
        legacy_mapping = LEGACY_LABEL_MAPPINGS
    
    for i, label in enumerate(labels):
        try:
            if label in legacy_mapping:
                new_label = int(legacy_mapping[label])
                converted_labels.append(new_label)
                conversion_log.append(f"Index {i}: {label} -> {new_label}")
            else:
                # Try direct conversion
                normalized = normalize_label_value(label)
                converted_labels.append(normalized)
                if normalized != label:
                    conversion_log.append(f"Index {i}: {label} -> {normalized} (normalized)")
        except ValueError as e:
            conversion_log.append(f"Index {i}: Failed to convert {label} - {e}")
            converted_labels.append(0)  # Default to noise class
    
    return jnp.array(converted_labels), conversion_log


def normalize_labels(labels: Union[List, jnp.ndarray],
                   expected_type: str = 'int',
                   handle_unknown: str = 'error') -> Tuple[jnp.ndarray, List[str]]:
    """
    Normalize labels to standard format with comprehensive error handling.
    
    Args:
        labels: Input labels in various formats
        expected_type: Expected output type ('int' or 'categorical')  
        handle_unknown: How to handle unknown labels ('error', 'ignore', 'map_to_noise')
        
    Returns:
        Tuple of (normalized_labels, normalization_log)
    """
    normalization_log = []
    normalized = []
    
    for i, label in enumerate(labels):
        try:
            if expected_type == 'int':
                norm_label = normalize_label_value(label)
                normalized.append(norm_label)
                
                if str(label) != str(norm_label):
                    normalization_log.append(f"Index {i}: {label} -> {norm_label}")
            else:
                # Categorical format
                if isinstance(label, (int, np.integer)):
                    if label in CANONICAL_LABELS:
                        canonical = CANONICAL_LABELS[GWSignalType(label)]
                        normalized.append(canonical)
                        normalization_log.append(f"Index {i}: {label} -> {canonical}")
                    else:
                        raise ValueError(f"Invalid integer label: {label}")
                else:
                    # Already string, validate
                    norm_int = normalize_label_value(label)
                    canonical = CANONICAL_LABELS[GWSignalType(norm_int)]
                    normalized.append(canonical)
                    
                    if label != canonical:
                        normalization_log.append(f"Index {i}: {label} -> {canonical}")
                        
        except ValueError as e:
            if handle_unknown == 'error':
                raise LabelError(
                    f"Cannot normalize label at index {i}",
                    row_number=i,
                    invalid_value=label,
                    context={'normalization_error': str(e)},
                    suggestions=['Check label format', 'Use handle_unknown="map_to_noise"']
                )
            elif handle_unknown == 'map_to_noise':
                default_val = 0 if expected_type == 'int' else 'noise'
                normalized.append(default_val)
                normalization_log.append(f"Index {i}: {label} -> {default_val} (mapped unknown to noise)")
            else:  # ignore
                normalization_log.append(f"Index {i}: Ignoring unknown label {label}")
                continue
    
    return jnp.array(normalized), normalization_log


# Export main functions
__all__ = [
    'LabelError',
    'LabelValidationResult', 
    'validate_labels',
    'validate_dataset_labels',
    'convert_legacy_labels',
    'normalize_labels',
    '_fast_validity_check',
    '_fast_class_distribution'
] 