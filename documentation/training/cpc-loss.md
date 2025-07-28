# CPC Loss Function

## Concept

The Contrastive Predictive Coding (CPC) loss function is the cornerstone of the self-supervised learning phase in the CPC-SNN-GW system. Its purpose is to train the CPC encoder to learn a meaningful, high-level representation of the input gravitational wave strain data without requiring any manual labels.


The core idea is **future prediction**. The model is trained to predict the features of future time steps in a sequence based on the context (features) of past time steps. This forces the encoder to capture the underlying temporal dynamics and structure of the signal.


The specific implementation used in this system is the **Temporal InfoNCE (Noise Contrastive Estimation)** loss. This is a powerful objective that works by contrasting a true future target (a positive sample) against a large number of negative samples (randomly selected from other time steps or sequences). The model is trained to assign a high similarity score to the positive pair and low scores to the negative pairs.


A critical innovation in this implementation is its robustness to the `batch_size=1` constraint. Earlier versions of the system suffered from a `CPC loss = 0.000000` bug, which rendered the contrastive learning non-functional. The current `calculate_fixed_cpc_loss` function resolves this by being batch-agnostic and using a proper temporal shift for positive pairs.


## Implementation

The CPC loss function is implemented in `training/cpc_loss_fixes.py`. The key function is `calculate_fixed_cpc_loss`.

```python
import jax.numpy as jnp
import jax

def calculate_fixed_cpc_loss(
    cpc_features: jnp.ndarray, 
    temperature: float = 0.07
) -> jnp.ndarray:
    """Calculate the Temporal InfoNCE loss for the CPC encoder.
    
    This function is robust and works for any batch size, including batch_size=1.
    
    Args:
        cpc_features: A 3D array of shape (batch_size, time_steps, feature_dim)
                      containing the latent features from the CPC encoder.
        temperature: A scalar value that controls the sharpness of the similarity 
                     distribution. A lower temperature makes the distribution sharper.
    
    Returns:
        A scalar array representing the CPC loss value.
    """
    # Handle edge cases
    if cpc_features is None or cpc_features.shape[1] <= 1:
        return jnp.array(0.0)
    
    batch_size, time_steps, feature_dim = cpc_features.shape
    
    # CRITICAL FIX: Create positive pairs by shifting context and target in time
    # context_features[t] should predict target_features[t+1]
    context_features = cpc_features[:, :-1, :]  # [batch, time-1, features]
    target_features = cpc_features[:, 1:, :]    # [batch, time-1, features]
    
    # Flatten the sequences to treat each time step as a separate sample
    # This creates a large number of positive pairs for the contrastive loss
    context_flat = context_features.reshape(-1, feature_dim) # [batch*(time-1), features]
    target_flat = target_features.reshape(-1, feature_dim)   # [batch*(time-1), features]
    
    # Only proceed if we have at least 2 samples to contrast
    if context_flat.shape[0] > 1:
        # L2 normalize the features for numerical stability
        context_norm = context_flat / (jnp.linalg.norm(context_flat, axis=-1, keepdims=True) + 1e-8)
        target_norm = target_flat / (jnp.linalg.norm(target_flat, axis=-1, keepdims=True) + 1e-8)
        
        # Compute the similarity matrix (dot product of normalized features)
        # Each row i represents the similarity of context_norm[i] to all target_norm[j]
        similarity_matrix = jnp.dot(context_norm, target_norm.T) # [N, N] where N = batch*(time-1)
        
        # The positive pairs are on the diagonal (i==j)
        num_samples = similarity_matrix.shape[0]
        labels = jnp.arange(num_samples)  # [0, 1, 2, ..., N-1]
        
        # Scale the similarities by the temperature
        scaled_similarities = similarity_matrix / temperature
        
        # Compute the InfoNCE loss
        # log_sum_exp is the log of the sum of exp(similarities) for each row
        log_sum_exp = jnp.log(jnp.sum(jnp.exp(scaled_similarities), axis=1) + 1e-8)
        # The loss is the mean of (log_sum_exp - positive_similarity)
        cpc_loss = -jnp.mean(scaled_similarities[jnp.arange(num_samples), labels] - log_sum_exp)
        
        return cpc_loss
    else:
        # Fallback: If we have only one sample, use a variance-based loss
        # This encourages the model to produce diverse features
        return -jnp.log(jnp.var(context_flat) + 1e-8)
```

## Mathematical Foundation

The Temporal InfoNCE loss can be expressed mathematically as:

```
L_CPC = -E[ log( exp(s(c_t, x_{t+k}) / τ) / Σ_{i=1}^N exp(s(c_t, x̃_i) / τ) ) ]
```

Where:
*   `c_t` is the context vector at time `t`.
*   `x_{t+k}` is the true future target at time `t+k` (the positive sample).
*   `x̃_i` are the negative samples (N in total).
*   `s(a, b)` is the similarity function (dot product of L2-normalized vectors in this case).
*   `τ` is the temperature parameter.


The loss function maximizes the log-probability of the positive sample relative to the negative samples. A lower loss indicates that the model is better at distinguishing the true future from the distractors.


## Usage

The CPC loss function is used during the pre-training phase of the `UnifiedTrainer`. It is typically combined with a classification loss in a composite loss function for the joint training phase.


```python
# Example: Calculate the CPC loss for a batch of latent features
latent_features = encoder.apply(encoder_params, train_signals)  # Shape: (16, 256, 128)

cpc_loss_value = calculate_fixed_cpc_loss(latent_features, temperature=0.07)
print(f"CPC Loss: {cpc_loss_value:.6f}")

# The loss should be a positive value (e.g., > 0.5). A value of 0.0 indicates a problem.
assert cpc_loss_value > 1e-6, "CPC loss is zero, indicating a training failure."
```

The successful operation of this loss function, as evidenced by a non-zero and decreasing loss value during training, is a critical indicator that the self-supervised learning phase is working correctly and that the CPC encoder is learning useful representations.