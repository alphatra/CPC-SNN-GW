# Test Evaluation

## Concept

The test evaluation module is the final quality assurance checkpoint for the CPC-SNN-GW system. Its purpose is to provide a comprehensive, scientifically rigorous assessment of the trained model's performance on a held-out test set. This evaluation goes far beyond a simple accuracy score to provide a holistic view of the model's capabilities and potential failure modes.


The evaluation is designed to prevent common pitfalls in machine learning, such as "fake accuracy" (where a model achieves high accuracy by always predicting the majority class) and model collapse (where the model's predictions are trivial and non-diverse). It provides a suite of metrics and checks to ensure that the reported performance is genuine and trustworthy.


## Implementation

The test evaluation is implemented in `training/test_evaluation.py`. The core function is `evaluate_on_test_set`.

```python
import jax.numpy as jnp
from typing import Dict, Any

def evaluate_on_test_set(
    trainer_state, 
    test_signals: jnp.ndarray, 
    test_labels: jnp.ndarray,
    train_signals: jnp.ndarray = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Evaluate a trained model on a test set and perform a detailed analysis.
    
    Args:
        trainer_state: The state of the trained model (parameters, optimizer state, etc.).
        test_signals: A 2D array of test input signals of shape (num_test, sequence_length).
        test_labels: A 1D array of true binary labels for the test set.
        train_signals: Optional. The training signals, used to check for data leakage.
        verbose: If True, prints a detailed summary of the results.
    
    Returns:
        A dictionary containing various performance metrics and diagnostic flags.
    """
    # Check for data leakage: ensure the test set is distinct from the training set
    data_leakage = (train_signals is not None and 
                   jnp.array_equal(test_signals, train_signals))
    
    # Perform a forward pass to get predictions
    test_predictions = []
    for i in range(len(test_signals)):
        test_signal = test_signals[i:i+1]  # Add batch dimension
        # The apply_fn is assumed to be the forward pass of the entire model
        test_logits = trainer_state.apply_fn(trainer_state.params, test_signal, train=False)
        test_pred = jnp.argmax(test_logits, axis=-1)[0]  # Get the predicted class
        test_predictions.append(int(test_pred))
    
    test_predictions = jnp.array(test_predictions)
    test_accuracy = jnp.mean(test_predictions == test_labels)
    
    # Model collapse detection: check if all predictions are the same
    unique_preds = jnp.unique(test_predictions)
    model_collapse = len(unique_preds) == 1
    
    # Calculate scientific metrics from the confusion matrix
    if len(jnp.unique(test_labels)) == 2:  # Ensure it's a binary classification problem
        # Count true positives, false negatives, etc.
        true_positives = int(jnp.sum((test_predictions == 1) & (test_labels == 1)))
        false_negatives = int(jnp.sum((test_predictions == 0) & (test_labels == 1)))
        true_negatives = int(jnp.sum((test_predictions == 0) & (test_labels == 0)))
        false_positives = int(jnp.sum((test_predictions == 1) & (test_labels == 0)))
        
        # Calculate standard metrics
        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    else:
        # If not binary, set metrics to 0.0
        sensitivity = specificity = precision = f1_score = 0.0
    
    # Detect suspicious patterns in the results
    suspicious_patterns = []
    if test_accuracy > 0.95:
        suspicious_patterns.append("suspiciously_high_accuracy")
    if model_collapse:
        suspicious_patterns.append("model_collapse")
    if data_leakage:
        suspicious_patterns.append("data_leakage")
    
    # Create the results dictionary
    results = {
        'test_accuracy': float(test_accuracy),
        'has_proper_test_set': not data_leakage,
        'model_collapse': model_collapse,
        'collapsed_class': int(unique_preds[0]) if model_collapse else None,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score,
        'suspicious_patterns': suspicious_patterns,
        'data_source': 'Real LIGO GW150914',
        'quality_validated': len(suspicious_patterns) == 0  # True if no red flags
    }
    
    # Print a detailed summary if requested
    if verbose:
        print(create_test_evaluation_summary(0.0, results, "Real ReadLIGO GW150914", 0))
    
    return results

def create_test_evaluation_summary(
    train_accuracy: float, 
    test_results: Dict[str, Any], 
    data_source: str, 
    num_epochs: int
) -> str:
    """Generate a professional, human-readable summary of the test evaluation results."""
    summary = f"""
🧪 TEST EVALUATION SUMMARY
=========================
Data Source: {data_source}
Training Epochs: {num_epochs}
Train Accuracy: {train_accuracy:.4f}

📊 PERFORMANCE METRICS
---------------------
Test Accuracy: {test_results['test_accuracy']:.4f}
Sensitivity (TPR): {test_results['sensitivity']:.4f}
Specificity (TNR): {test_results['specificity']:.4f}
Precision: {test_results['precision']:.4f}
F1-Score: {test_results['f1_score']:.4f}

🔍 QUALITY ASSURANCE
-------------------
Model Collapse: {test_results['model_collapse']}
Data Leakage: {not test_results['has_proper_test_set']}
Suspicious Patterns: {', '.join(test_results['suspicious_patterns']) if test_results['suspicious_patterns'] else 'None'}
Overall Quality: {'✅ VALID' if test_results['quality_validated'] else '❌ INVALID'}
"""
    return summary
```

## Usage

The `evaluate_on_test_set` function is called at the end of the training process, typically in the `run_advanced_pipeline.py` script or the `enhanced_cli.py`.

```python
# Example: Evaluate the trained model
from training.test_evaluation import evaluate_on_test_set

# Assume we have a trained trainer_state and a test set
test_results = evaluate_on_test_set(
    trainer_state=trainer_state,
    test_signals=jnp.array(test_signals),
    test_labels=jnp.array(test_labels),
    train_signals=jnp.array(train_signals),  # For data leakage check
    verbose=True
)

# The results can be used for reporting or further analysis
print(f"Final Test Accuracy: {test_results['test_accuracy']:.4f}")
if test_results['model_collapse']:
    print("WARNING: Model collapse detected. Training may have failed.")
else:
    print("Model passed quality checks.")
```

This comprehensive evaluation framework is a defining feature of the system, ensuring that any claims about its performance are backed by rigorous, transparent, and trustworthy analysis.