# API Reference

## Data Module

### `data.real_ligo_integration`

This module provides the primary interface for loading and processing real LIGO gravitational wave data.

#### `download_gw150914_data() -> Optional[np.ndarray]`
Loads the authentic GW150914 strain data from the LIGO Open Science Center (LOSC) HDF5 files. It combines data from the H1 and L1 detectors and extracts a 2048-sample segment centered on the event. If the real data is unavailable, it returns a physics-accurate simulated fallback.

*   **Returns**: A 1D numpy array of float32 strain data, or `None` if both real and simulated data fail to load.

#### `create_proper_windows(strain_data: np.ndarray, window_size: int = 256, overlap: float = 0.5) -> Tuple[jnp.ndarray, jnp.ndarray]`
Creates overlapping, labeled windows from a continuous strain data segment. The labeling is based on the temporal proximity to the known GW150914 event.
*   **Parameters**:
    *   `strain_data`: The 1D array of raw strain data.
    *   `window_size`: The number of samples in each window.
    *   `overlap`: The fraction of overlap between consecutive windows (e.g., 0.5 for 50% overlap).
*   **Returns**: A tuple of `(signals, labels)`, where `signals` is a 2D JAX array of shape `(num_windows, window_size)` and `labels` is a 1D JAX array of binary labels.

#### `create_real_ligo_dataset(num_samples: int = 1000, window_size: int = 256, quick_mode: bool = False, return_split: bool = True, train_ratio: float = 0.8) -> Union[Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]], Tuple[jnp.ndarray, jnp.ndarray]]`
A high-level function that orchestrates the entire data pipeline. It downloads the data, creates windows, applies data augmentation (if not in `quick_mode`), and optionally performs a stratified train/test split.
*   **Parameters**:
    *   `num_samples`: The target number of samples in the final dataset.
    *   `window_size`: The size of each data window.
    *   `quick_mode`: If `True`, skips data augmentation for faster dataset creation.
    *   `return_split`: If `True`, returns a train/test split; otherwise, returns the full dataset.
    *   `train_ratio`: The proportion of data to use for training if `return_split` is `True`.
*   **Returns**: If `return_split` is `True`, returns a tuple of tuples `((train_signals, train_labels), (test_signals, test_labels))`. Otherwise, returns a tuple `(signals, labels)`.

---

## Training Module

### `training.cpc_loss_fixes`

This module contains the corrected implementation of the CPC loss function.

#### `calculate_fixed_cpc_loss(cpc_features: Optional[jnp.ndarray], temperature: float = 0.07) -> jnp.ndarray`
Calculates the Temporal InfoNCE loss for the CPC encoder. This function is robust and works for any batch size, including the critical `batch_size=1` case.
*   **Parameters**:
    *   `cpc_features`: A 3D JAX array of shape `(batch_size, time_steps, feature_dim)` containing the latent features from the CPC encoder.
    *   `temperature`: A scalar value that controls the sharpness of the similarity distribution.
*   **Returns**: A scalar JAX array representing the CPC loss value.

#### `create_enhanced_loss_fn(cpc_weight: float = 1.0, classification_weight: float = 1.0, temperature: float = 0.07) -> Callable`
Creates a composite loss function that combines the fixed CPC loss and the SNN classification loss (e.g., cross-entropy) with configurable weights.
*   **Parameters**:
    *   `cpc_weight`: The weight assigned to the CPC loss component.
    *   `classification_weight`: The weight assigned to the SNN classification loss component.
    *   `temperature`: The temperature parameter for the CPC loss.
*   **Returns**: A callable loss function that takes model parameters, inputs, and targets, and returns the total loss.


### `training.test_evaluation`

This module provides functions for comprehensive model evaluation.

#### `evaluate_on_test_set(trainer_state, test_signals: jnp.ndarray, test_labels: jnp.ndarray, train_signals: jnp.ndarray = None, verbose: bool = True) -> Dict[str, Any]`
Evaluates a trained model on a test set and performs a detailed analysis.
*   **Parameters**:
    *   `trainer_state`: The state of the trained model (parameters, optimizer state, etc.).
    *   `test_signals`: A 2D JAX array of test input signals.
    *   `test_labels`: A 1D JAX array of true binary labels for the test set.
    *   `train_signals`: Optional. The training signals, used to check for data leakage.
    *   `verbose`: If `True`, prints a detailed summary of the results.
*   **Returns**: A dictionary containing `test_accuracy`, `model_collapse`, `sensitivity`, `specificity`, `precision`, `f1_score`, and a list of `suspicious_patterns`.

#### `create_test_evaluation_summary(train_accuracy: float, test_results: Dict[str, Any], data_source: str, num_epochs: int) -> str`
Generates a professional, human-readable summary of the test evaluation results.
*   **Parameters**:
    *   `train_accuracy`: The final training accuracy.
    *   `test_results`: The dictionary returned by `evaluate_on_test_set`.
    *   `data_source`: A string describing the data source (e.g., "Real ReadLIGO GW150914").
    *   `num_epochs`: The number of epochs the model was trained for.
*   **Returns**: A formatted string containing the summary report.

---

## Utils Module

### `utils.data_split`

This module provides functions for splitting data into training and test sets.

#### `create_stratified_split(signals: jnp.ndarray, labels: jnp.ndarray, train_ratio: float = 0.8, random_seed: int = 42) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]`
Splits the data into training and test sets while preserving the proportion of each class.
*   **Parameters**:
    *   `signals`: A 2D JAX array of input signals.
    *   `labels`: A 1D JAX array of binary labels.
    *   `train_ratio`: The proportion of data to use for training.
    *   `random_seed`: A seed for the random number generator to ensure reproducibility.
*   **Returns**: A tuple of tuples `((train_signals, train_labels), (test_signals, test_labels))`.

#### `validate_split_quality(test_labels: jnp.ndarray) -> bool`
Validates that the test set contains a proper mix of classes and raises an error if it does not.
*   **Parameters**:
    *   `test_labels`: A 1D JAX array of labels from the test set.
*   **Returns**: `True` if the split is valid.
*   **Raises**: `ValueError` if the test set contains only one class.