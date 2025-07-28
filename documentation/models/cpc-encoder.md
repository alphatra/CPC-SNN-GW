# CPC Encoder

## Architecture

The CPC (Contrastive Predictive Coding) Encoder is the first stage of the neuromorphic pipeline, responsible for learning high-level, self-supervised representations from raw gravitational wave strain data. Its primary function is to encode a sequence of input time steps into a sequence of latent features, which are then used by the subsequent stages of the model.

The encoder's architecture is designed to capture the temporal structure of the signal. It consists of a series of convolutional layers followed by a recurrent layer (e.g., LSTM or GRU). The convolutional layers extract local features from the input, while the recurrent layer models the long-range temporal dependencies in the data.

The output of the encoder is a 3D tensor of shape `(batch_size, time_steps, latent_dim)`, where `latent_dim` is a configurable hyperparameter (default: 128). This sequence of latent features serves as the context for the contrastive learning objective.


## Implementation

The `CPCEncoder` class is implemented in `models/cpc_encoder.py`. It is a Flax `nn.Module` that defines the forward pass of the encoder.

```python
import jax.numpy as jnp
import flax.linen as nn

class CPCEncoder(nn.Module):
    """A neural network encoder for Contrastive Predictive Coding."""
    latent_dim: int = 128
    
    @nn.compact
    def __call__(self, x):
        """Forward pass of the CPC encoder.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, 1).
            
        Returns:
            A tensor of shape (batch_size, sequence_length, latent_dim) containing 
            the latent features.
        """
        # Convolutional layers for local feature extraction
        x = nn.Conv(features=64, kernel_size=(7,), strides=2)(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(5,), strides=2)(x)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(3,), strides=2)(x)
        x = nn.relu(x)
        
        # Recurrent layer for temporal modeling
        x, _ = nn.GRUCell()(x)
        
        # Final projection to latent space
        x = nn.Dense(features=self.latent_dim)(x)
        return x
```

## Usage

To use the CPC encoder, you first need to initialize it and its parameters. This is typically done as part of the `UnifiedTrainer`.

```python
import jax
import jax.numpy as jnp
from models.cpc_encoder import CPCEncoder

# Initialize the encoder
encoder = CPCEncoder(latent_dim=128)

# Create a dummy input for initialization
dummy_input = jnp.ones((1, 256, 1))  # (batch_size, sequence_length, channels)

# Initialize the parameters
rng = jax.random.PRNGKey(0)
params = encoder.init(rng, dummy_input)

# Apply the encoder to real data
real_strain_data = jnp.array(train_signals[0:1])  # Example: first training sample
latent_features = encoder.apply(params, real_strain_data)
print(f"Latent features shape: {latent_features.shape}")  # Should be (1, 256, 128)
```

The `latent_features` output can then be passed to the `calculate_fixed_cpc_loss` function for training or to the `SpikeBridge` for neuromorphic conversion.