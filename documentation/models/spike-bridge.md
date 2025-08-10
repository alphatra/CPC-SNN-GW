# Spike Bridge

## Architecture

The Spike Bridge is a critical component that acts as a transducer between the continuous, high-dimensional latent features produced by the CPC encoder and the discrete, event-driven Spiking Neural Network (SNN) classifier. Its primary function is to convert the continuous representations into a spike-based code that the SNN can process.


The bridge employs a **Temporal-Contrast encoding** scheme, which is a significant improvement over simpler methods like Poisson encoding. This scheme generates spikes based on the rate of change (contrast) of the input signal over time. A spike is emitted when the difference between consecutive time steps exceeds a certain threshold. This method is particularly effective for gravitational wave signals because it preserves the phase and frequency information of the input, which are crucial for detection.


The architecture of the Spike Bridge is relatively simple but highly effective. It takes the sequence of latent features from the CPC encoder and applies the temporal-contrast encoding to each feature dimension independently. The output is a 3D tensor of shape `(batch_size, time_steps, latent_dim)` where the values are binary (0 or 1), representing the absence or presence of a spike at that time step for that feature.


## Implementation

The `SpikeBridge` class is implemented in `models/spike_bridge.py`. It is designed to be differentiable, which is essential for end-to-end training of the entire pipeline. This is achieved through the use of a surrogate gradient function during backpropagation.

```python
import jax.numpy as jnp
import jax
import flax.linen as nn

# Define a surrogate gradient function for the spike generation
@jax.custom_jvp
def spike_function(x):
    """A step function that outputs 1 if x > 0, else 0."""
    return (x > 0).astype(jnp.float32)

@spike_function.defjvp
def spike_function_jvp(primals, tangents):
    """Define the surrogate gradient for the spike function.
    
    During the forward pass, this is the same as the primal function.
    During the backward pass, the gradient is passed through as if the 
    function were a sigmoid with a high slope (beta).
    """
    (x,), (x_dot,) = primals, tangents
    # Forward pass
    primal_out = spike_function(x)
    # Backward pass: use a sigmoid-like gradient (surrogate gradient)
    beta = 4.0  # Slope of the surrogate gradient
    tangent_out = x_dot * beta * jnp.exp(-beta * jnp.abs(x))
    return primal_out, tangent_out

class SpikeBridge(nn.Module):
    """A module that converts continuous features into spike trains."""
    threshold: float = 0.1
    
    @nn.compact
    def __call__(self, x):
        """Forward pass of the Spike Bridge.
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, feature_dim) 
                 containing continuous latent features.
            
        Returns:
            A tensor of shape (batch_size, time_steps, feature_dim) containing 
            binary spike trains.
        """
        # Calculate the temporal contrast (difference between consecutive time steps)
        # Pad the first time step with zeros to maintain the sequence length
        x_padded = jnp.pad(x, ((0, 0), (1, 0), (0, 0)), mode='constant')
        contrast = x - x_padded[:, :-1, :]  # (batch_size, time_steps, feature_dim)
        
        # Generate spikes using the temporal-contrast encoding
        spikes = spike_function(contrast - self.threshold)
        
        return spikes
```

## Usage

The Spike Bridge is used as an intermediate layer between the CPC encoder and the SNN classifier. Its parameters are typically not trained, as the encoding scheme is fixed.

```python
from models.spike_bridge import SpikeBridge

# Initialize the Spike Bridge
bridge = SpikeBridge(threshold=0.1)

# Create a dummy input for initialization (use output shape from CPC encoder)
dummy_latent_features = jnp.ones((1, 256, 128))  # (batch_size, time_steps, latent_dim)

# Initialize the parameters (though they are not used in this simple module)
rng = jax.random.PRNGKey(1)
params = bridge.init(rng, dummy_latent_features)

# Apply the bridge to latent features from the CPC encoder
latent_features = encoder.apply(encoder_params, real_strain_data)  # From previous example
spike_trains = bridge.apply(params, latent_features)
print(f"Spike trains shape: {spike_trains.shape}")  # Should be (1, 256, 128)
print(f"Spike rate: {jnp.mean(spike_trains):.4f}")  # Average fraction of spikes
```

The resulting `spike_trains` are then fed into the SNN classifier for the final detection task.