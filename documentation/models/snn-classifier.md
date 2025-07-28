# SNN Classifier

## Architecture

The SNN (Spiking Neural Network) Classifier is the final stage of the neuromorphic pipeline, responsible for performing the binary classification task of distinguishing between gravitational wave signals (positive class) and noise (negative class). It is implemented using the LIF (Leaky Integrate-and-Fire) neuron model, a standard and biologically plausible model of neuronal dynamics.


The classifier has a feedforward architecture with 4 fully connected layers. The number of neurons in each layer is configurable, with a default configuration of `[512, 256, 128, 64]`. The final layer has 2 neurons, corresponding to the two output classes (signal and noise). The network is designed to be deep enough to learn complex, hierarchical representations of the spike-based input.


A key innovation is the use of an **enhanced LIF model** that includes two additional biological mechanisms:

1.  **Refractory Period**: After a neuron fires a spike, it enters a refractory period during which its membrane potential is clamped to a low value, preventing it from firing again immediately. This adds temporal dynamics and realism to the model.
2.  **Adaptation**: The neuron's firing threshold increases with sustained activity. This helps prevent runaway excitation and improves the network's stability during prolonged input, making it more robust to noisy data.


The network operates in a time-stepped simulation. At each time step, the input spike trains are presented to the first layer, and the state of all neurons is updated. The final classification decision is made by aggregating the total number of spikes (or the final membrane potential) from the output neurons over the entire simulation period (e.g., 4096 time steps).


## Implementation

The `SNNClassifier` class is implemented in `models/snn_classifier.py`. It is built using the Spyx library, which provides JAX-compatible primitives for spiking neural networks.

```python
import jax.numpy as jnp
import jax
import spyx.nn as snn
import flax.linen as nn

# Define the enhanced LIF neuron with refractory period and adaptation
class EnhancedLIF(snn.LIF):
    """An enhanced LIF neuron with refractory period and adaptation."""
    def __init__(self, num_features, tau_mem=5e-5, tau_ref=2e-3, tau_adapt=2e-2, beta=4.0):
        super().__init__(num_features, tau_mem=tau_mem, beta=beta)
        self.tau_ref = tau_ref  # Refractory time constant
        self.tau_adapt = tau_adapt  # Adaptation time constant
        
    def __call__(self, x, state):
        """Forward pass with enhanced dynamics."""
        # Unpack the state
        v, ref, adapt = state
        
        # Apply refractory period: if ref > 0, set input to 0
        x = jnp.where(ref > 0, 0.0, x)
        
        # Standard LIF dynamics for membrane potential
        dv = (-(v - self.v_leak) + x) / self.tau_mem
        v_new = v + dv * self.dt
        
        # Spike generation
        z = (v_new >= self.v_th + adapt).astype(jnp.float32)  # Threshold is increased by adaptation
        
        # Reset membrane potential
        v_new = jnp.where(z > 0, self.v_reset, v_new)
        
        # Update refractory period
        dref = -ref / self.tau_ref
        ref_new = ref + dref * self.dt
        ref_new = jnp.where(z > 0, self.tau_ref, ref_new)  # Reset to tau_ref on spike
        
        # Update adaptation variable
        dadapt = -adapt / self.tau_adapt
        adapt_new = adapt + dadapt * self.dt
        adapt_new = jnp.where(z > 0, adapt_new + 0.1, adapt_new)  # Increase on spike
        
        return z, (v_new, ref_new, adapt_new)

class SNNClassifier(nn.Module):
    """A spiking neural network for binary classification."""
    num_classes: int = 2
    hidden_sizes: list = (512, 256, 128)
    
    @nn.compact
    def __call__(self, x):
        """Forward pass of the SNN classifier.
        
        Args:
            x: Input spike trains of shape (batch_size, time_steps, input_dim).
            
        Returns:
            A scalar value representing the classification decision.
        """
        # Define the layers
        layers = []
        input_size = x.shape[-1]
        for size in self.hidden_sizes:
            layers.append(snn.Dense(size))
            layers.append(EnhancedLIF(size))
            x = x  # The input dimension for the next layer
        
        # Output layer
        layers.append(snn.Dense(self.num_classes))
        output_layer = EnhancedLIF(self.num_classes)
        
        # Simulate the network over time
        # Initialize the state for all layers
        states = [layer.initialize_carry(x.shape[0]) for layer in layers[:-1]]
        output_state = output_layer.initialize_carry(x.shape[0])
        
        # Accumulate output spikes over time
        output_spikes = jnp.zeros((x.shape[0], self.num_classes))
        
        for t in range(x.shape[1]):
            # Get input for current time step
            spike_input = x[:, t, :]
            
            # Forward pass through hidden layers
            for i, layer in enumerate(layers[:-1]):
                if isinstance(layer, snn.Dense):
                    spike_input = layer(spike_input)
                else:  # It's an LIF layer
                    spike_input, states[i] = layer(spike_input, states[i])
            
            # Forward pass through output layer
            output_spikes_t, output_state = output_layer(spike_input, output_state)
            output_spikes += output_spikes_t
        
        # Classification decision based on total output spikes
        return jnp.argmax(output_spikes, axis=-1)
```

## Usage

The SNN Classifier is the final component in the pipeline. It takes the spike trains from the Spike Bridge and produces the final classification.

```python
from models.snn_classifier import SNNClassifier

# Initialize the SNN classifier
classifier = SNNClassifier(num_classes=2, hidden_sizes=[512, 256, 128])

# Create a dummy input for initialization (use output shape from Spike Bridge)
dummy_spike_trains = jnp.ones((1, 4096, 128))  # (batch_size, time_steps, feature_dim)

# Initialize the parameters
rng = jax.random.PRNGKey(2)
params = classifier.init(rng, dummy_spike_trains)

# Apply the classifier to spike trains from the Spike Bridge
spike_trains = bridge.apply(bridge_params, latent_features)  # From previous example
prediction = classifier.apply(params, spike_trains)
print(f"Final prediction: {'Signal' if prediction[0] == 1 else 'Noise'}")
```

This completes the end-to-end neuromorphic pipeline from raw strain data to a final detection decision.