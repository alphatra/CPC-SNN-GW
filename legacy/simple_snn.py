"""
Simple JAX-based SNN Implementation

Lightweight LIF (Leaky Integrate-and-Fire) SNN classifier
for quick testing of CPC + Spike Bridge + SNN integration.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Tuple


class LIFLayer(nn.Module):
    """Simple LIF (Leaky Integrate-and-Fire) layer in JAX/Flax."""
    
    hidden_size: int
    tau_mem: float = 20e-3  # Membrane time constant
    tau_syn: float = 5e-3   # Synaptic time constant  
    threshold: float = 1.0  # Spike threshold
    dt: float = 1e-3        # Time step
    
    @nn.compact
    def __call__(self, spikes: jnp.ndarray) -> jnp.ndarray:
        """
        LIF layer forward pass.
        
        Args:
            spikes: Input spikes [batch, time, input_dim]
            
        Returns:
            output_spikes: Output spikes [batch, time, hidden_size]
        """
        batch_size, time_steps, input_dim = spikes.shape
        
        # Linear transformation (synaptic weights)
        W = self.param('kernel', nn.initializers.glorot_uniform(), (input_dim, self.hidden_size))
        b = self.param('bias', nn.initializers.zeros, (self.hidden_size,))
        
        # Decay factors
        alpha_mem = jnp.exp(-self.dt / self.tau_mem)  # Membrane decay
        alpha_syn = jnp.exp(-self.dt / self.tau_syn)  # Synaptic decay
        
        # Initial states
        v_mem = jnp.zeros((batch_size, self.hidden_size))  # Membrane potential
        i_syn = jnp.zeros((batch_size, self.hidden_size))  # Synaptic current
        
        def lif_step(states, x_t):
            """Single LIF time step."""
            v_mem, i_syn = states
            
            # Synaptic input
            i_input = jnp.dot(x_t, W) + b
            
            # Update synaptic current (exponential decay + input)
            i_syn = alpha_syn * i_syn + i_input
            
            # Update membrane potential (leak + synaptic current)
            v_mem = alpha_mem * v_mem + i_syn
            
            # Spike generation (threshold crossing)
            spikes_out = (v_mem >= self.threshold).astype(jnp.float32)
            
            # Reset membrane potential after spike
            v_mem = v_mem * (1 - spikes_out)
            
            return (v_mem, i_syn), spikes_out
        
        # Scan over time dimension
        _, output_spikes = jax.lax.scan(
            lif_step, 
            init=(v_mem, i_syn),
            xs=jnp.transpose(spikes, (1, 0, 2))  # [time, batch, input_dim]
        )
        
        # Transpose back: [batch, time, hidden_size]
        output_spikes = jnp.transpose(output_spikes, (1, 0, 2))
        
        return output_spikes


class SimpleSNN(nn.Module):
    """Simple 2-layer SNN classifier."""
    
    hidden_size: int = 128
    num_classes: int = 2
    tau_mem: float = 20e-3
    tau_syn: float = 5e-3
    threshold: float = 1.0
    
    @nn.compact  
    def __call__(self, spikes: jnp.ndarray) -> jnp.ndarray:
        """
        SNN forward pass.
        
        Args:
            spikes: Input spike trains [batch, time, input_dim]
            
        Returns:
            logits: Classification logits [batch, num_classes]
        """
        # First LIF layer
        h1 = LIFLayer(
            hidden_size=self.hidden_size,
            tau_mem=self.tau_mem,
            tau_syn=self.tau_syn,
            threshold=self.threshold
        )(spikes)
        
        # Second LIF layer
        h2 = LIFLayer(
            hidden_size=self.hidden_size // 2,
            tau_mem=self.tau_mem,
            tau_syn=self.tau_syn,
            threshold=self.threshold
        )(h1)
        
        # Global average pooling over time
        h_pooled = jnp.mean(h2, axis=1)  # [batch, hidden_size//2]
        
        # Linear readout
        logits = nn.Dense(self.num_classes)(h_pooled)
        
        return logits


class SimpleSNNTrainer:
    """Training utilities dla Simple SNN."""
    
    def __init__(self, learning_rate: float = 1e-3):
        self.optimizer = optax.adam(learning_rate)
        
    def classification_loss(self, 
                          params: dict,
                          spikes: jnp.ndarray, 
                          labels: jnp.ndarray,
                          model: SimpleSNN) -> jnp.ndarray:
        """Classification loss function."""
        logits = model.apply(params, spikes)
        
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, labels
        ).mean()
        
    def training_step(self,
                     params: dict,
                     opt_state: optax.OptState,
                     spikes: jnp.ndarray,
                     labels: jnp.ndarray,
                     model: SimpleSNN) -> Tuple[dict, optax.OptState, float]:
        """Single training step."""
        
        loss, grads = jax.value_and_grad(self.classification_loss)(
            params, spikes, labels, model
        )
        
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss
        
    def accuracy(self, 
                params: dict,
                spikes: jnp.ndarray, 
                labels: jnp.ndarray,
                model: SimpleSNN) -> float:
        """Compute classification accuracy."""
        logits = model.apply(params, spikes)
        predictions = jnp.argmax(logits, axis=-1)
        
        return jnp.mean(predictions == labels)


# Convenience functions
def create_simple_snn(hidden_size: int = 64, num_classes: int = 2) -> SimpleSNN:
    """Create simple SNN model."""
    return SimpleSNN(
        hidden_size=hidden_size,
        num_classes=num_classes,
        tau_mem=20e-3,
        tau_syn=5e-3,
        threshold=1.0
    )


def test_simple_lif_layer():
    """Quick test of LIF layer functionality."""
    # Create test data
    batch_size, time_steps, input_dim = 2, 10, 8
    test_spikes = jax.random.bernoulli(
        jax.random.PRNGKey(0), 0.1, (batch_size, time_steps, input_dim)
    ).astype(jnp.float32)
    
    # Create and test LIF layer
    lif = LIFLayer(hidden_size=16)
    key = jax.random.PRNGKey(42)
    params = lif.init(key, test_spikes)
    
    output_spikes = lif.apply(params, test_spikes)
    
    print(f"Input shape: {test_spikes.shape}")
    print(f"Output shape: {output_spikes.shape}")
    print(f"Input spike rate: {jnp.mean(test_spikes):.3f}")
    print(f"Output spike rate: {jnp.mean(output_spikes):.3f}")
    
    return output_spikes.shape == (batch_size, time_steps, 16)


def test_simple_snn():
    """Quick test of complete SNN."""
    # Create test data
    batch_size, time_steps, input_dim = 2, 20, 32
    test_spikes = jax.random.bernoulli(
        jax.random.PRNGKey(0), 0.05, (batch_size, time_steps, input_dim)
    ).astype(jnp.float32)
    test_labels = jnp.array([0, 1])
    
    # Create and test SNN
    snn = create_simple_snn(hidden_size=32, num_classes=2)
    trainer = SimpleSNNTrainer(learning_rate=1e-3)
    
    key = jax.random.PRNGKey(42)
    params = snn.init(key, test_spikes)
    opt_state = trainer.optimizer.init(params)
    
    # Forward pass
    logits = snn.apply(params, test_spikes)
    
    # Training step
    params, opt_state, loss = trainer.training_step(
        params, opt_state, test_spikes, test_labels, snn
    )
    
    # Accuracy
    accuracy = trainer.accuracy(params, test_spikes, test_labels, snn)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Training loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.3f}")
    
    return True


if __name__ == "__main__":
    print("Testing Simple LIF Layer...")
    success1 = test_simple_lif_layer()
    print(f"LIF Layer test: {'PASSED' if success1 else 'FAILED'}\n")
    
    print("Testing Simple SNN...")
    success2 = test_simple_snn()
    print(f"SNN test: {'PASSED' if success2 else 'FAILED'}")
    
    overall_success = success1 and success2
    print(f"\nOverall: {'SUCCESS' if overall_success else 'FAILED'}") 