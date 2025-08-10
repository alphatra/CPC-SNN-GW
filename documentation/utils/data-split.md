# Data Split

## Concept

The data split module is responsible for dividing the complete dataset of gravitational wave signals into distinct subsets for training and testing. This is a fundamental step in machine learning to evaluate a model's ability to generalize to unseen data.


The primary challenge addressed by this module is the prevention of **"fake accuracy."** This occurs when a test set is poorly constructed, for example, by containing only noise samples (all labels are 0). A model that simply predicts "noise" for every input will achieve 100% accuracy on such a test set, which is a meaningless and misleading result.


To solve this, the module implements a **stratified split**. This ensures that both the training and test sets have a proportional representation of each class (signal and noise). For instance, if the full dataset is 30% signal and 70% noise, the training and test sets will also be approximately 30% signal and 70% noise. This creates a balanced and scientifically valid evaluation environment.


## Implementation

The stratified split is implemented in `utils/data_split.py`. The core function is `create_stratified_split`.

```python
import jax.numpy as jnp
import jax
from typing import Tuple

def create_stratified_split(
    signals: jnp.ndarray, 
    labels: jnp.ndarray,
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """Split the data into training and test sets while preserving the proportion of each class.
    
    Args:
        signals: A 2D array of input signals of shape (num_samples, sequence_length).
        labels: A 1D array of binary labels (0 for noise, 1 for signal).
        train_ratio: The proportion of data to use for training (e.g., 0.8 for 80%).
        random_seed: A seed for the random number generator to ensure reproducibility.
    
    Returns:
        A tuple of tuples `((train_signals, train_labels), (test_signals, test_labels))`.
    """
    # Separate the indices of the two classes
    class_0_indices = jnp.where(labels == 0)[0]
    class_1_indices = jnp.where(labels == 1)[0]
    
    n_class_0 = len(class_0_indices)
    n_class_1 = len(class_1_indices)
    
    # Handle the case where one class is missing (e.g., all noise)
    if n_class_0 == 0 or n_class_1 == 0:
        # Fallback to a simple random split
        n_train = max(1, int(train_ratio * len(signals)))
        key = jax.random.PRNGKey(random_seed)
        indices = jax.random.permutation(key, len(signals))
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
    else:
        # Calculate the number of samples for training for each class
        n_train_0 = max(1, int(train_ratio * n_class_0))
        n_train_1 = max(1, int(train_ratio * n_class_1))
        
        # Shuffle the indices of each class separately
        key = jax.random.PRNGKey(random_seed)
        shuffled_0 = jax.random.permutation(key, class_0_indices)
        shuffled_1 = jax.random.permutation(jax.random.split(key)[0], class_1_indices)
        
        # Split each class
        train_indices_0 = shuffled_0[:n_train_0]
        test_indices_0 = shuffled_0[n_train_0:]
        train_indices_1 = shuffled_1[:n_train_1] 
        test_indices_1 = shuffled_1[n_train_1:]
        
        # Combine the indices and shuffle them together
        train_indices = jnp.concatenate([train_indices_0, train_indices_1])
        test_indices = jnp.concatenate([test_indices_0, test_indices_1])
        
        # Shuffle the combined indices to mix the classes
        train_indices = jax.random.permutation(jax.random.split(key)[1], train_indices)
        test_indices = jax.random.permutation(jax.random.split(key)[2], test_indices)
    
    # Extract the final splits
    train_signals = signals[train_indices]
    train_labels = labels[train_indices]
    test_signals = signals[test_indices] 
    test_labels = labels[test_indices]
    
    return (train_signals, train_labels), (test_signals, test_labels)

def validate_split_quality(test_labels: jnp.ndarray) -> bool:
    """Validate that the test set contains a proper mix of classes.
    
    Args:
        test_labels: A 1D array of labels from the test set.
    
    Returns:
        True if the split is valid.
    
    Raises:
        ValueError: If the test set contains only one class.
    """
    if len(test_labels) == 0:
        return True  # An empty test set is not our concern here
    
    # Check if all labels are the same (all 0s or all 1s)
    if jnp.all(test_labels == 0) or jnp.all(test_labels == 1):
        raise ValueError("Single-class test set detected - would give fake accuracy!")
    return True
```

## Usage

The `create_stratified_split` function is a critical component used in the `create_real_ligo_dataset` function to ensure the integrity of the final dataset.

```python
# Example: Create a stratified train/test split
from utils.data_split import create_stratified_split, validate_split_quality

# Assume we have a complete dataset
all_signals = jnp.array([...]) # Shape: (2000, 256)
all_labels = jnp.array([...]) # Shape: (2000,)

# Perform the stratified split
(train_signals, train_labels), (test_signals, test_labels) = create_stratified_split(
    all_signals, 
    all_labels, 
    train_ratio=0.8, 
    random_seed=42
)

print(f"Training set: {train_signals.shape[0]} samples ({jnp.mean(train_labels):.1%} signal)")
print(f"Test set: {test_signals.shape[0]} samples ({jnp.mean(test_labels):.1%} signal)")

# Validate the test set quality
try:
    validate_split_quality(test_labels)
    print("✅ Test set quality validated: contains both classes.")
except ValueError as e:
    print(f"❌ Validation failed: {e}")
```

By enforcing a stratified split and validating the test set, this module plays a vital role in ensuring the scientific rigor and credibility of the entire project.