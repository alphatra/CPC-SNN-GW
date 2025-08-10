# Glitch Injection

## Concept

Glitch injection is a data augmentation technique used to improve the robustness and generalization of the CPC-SNN-GW system. Real gravitational wave detector data is plagued by transient, non-astrophysical noise artifacts known as "glitches." These glitches can mimic the characteristics of real GW signals, leading to false alarms if the model is not trained to distinguish between them.


The primary goal of glitch injection is to make the model invariant to these common noise sources. By artificially adding realistic glitches into the training data, the model learns that these patterns are not associated with the positive class (real GW signals) and becomes more robust to them in real-world operation.


This technique transforms the model from a simple pattern matcher into a more sophisticated detector that can differentiate between true astrophysical signals and instrumental artifacts.


## Implementation

The glitch injection module is implemented in `data/glitch_injector.py`. It contains a library of simulated glitch waveforms that mimic common types of LIGO detector noise, such as "blips," "koi fish," and "whistles."

```python
import jax.numpy as jnp
import jax
import numpy as np
from typing import Tuple, Dict

# Define a library of glitch waveforms (simplified examples)
def create_blip_glitch(duration: float, sample_rate: float) -> jnp.ndarray:
    """Create a short, broadband 'blip' glitch."""
    t = jnp.linspace(0, duration, int(duration * sample_rate))
    # A Gaussian envelope modulating white noise
    envelope = jnp.exp(-5 * (t - duration/2)**2 / duration**2)
    noise = jax.random.normal(jax.random.PRNGKey(0), t.shape)
    glitch = envelope * noise
    return glitch

def create_whistle_glitch(duration: float, sample_rate: float, start_freq: float, end_freq: float) -> jnp.ndarray:
    """Create a frequency-modulated 'whistle' glitch."""
    t = jnp.linspace(0, duration, int(duration * sample_rate))
    # Chirp signal
    instantaneous_freq = start_freq + (end_freq - start_freq) * (t / duration)
    phase = 2 * jnp.pi * jnp.cumsum(instantaneous_freq) / sample_rate
    glitch = jnp.sin(phase)
    # Apply a Gaussian envelope
    envelope = jnp.exp(-5 * (t - duration/2)**2 / duration**2)
    return envelope * glitch

def create_koi_fish_glitch(duration: float, sample_rate: float) -> jnp.ndarray:
    """Create a 'koi fish' glitch (a blip with a tail)."""
    t = jnp.linspace(0, duration, int(duration * sample_rate))
    # Main blip
    main_duration = duration * 0.3
    main_t = t[t < main_duration]
    main_envelope = jnp.exp(-5 * (main_t - main_duration/2)**2 / main_duration**2)
    main_noise = jax.random.normal(jax.random.PRNGKey(1), main_t.shape)
    main_glitch = jnp.zeros_like(t).at[:len(main_t)].set(main_envelope * main_noise)
    
    # Exponential decay tail
    tail_start = main_duration
    tail_t = t[t >= tail_start] - tail_start
    tail = jnp.exp(-tail_t / (duration * 0.1)) * jnp.sin(2 * jnp.pi * 100 * tail_t)  # 100 Hz oscillation
    tail_glitch = jnp.zeros_like(t).at[t >= tail_start].set(tail)
    
    return main_glitch + tail_glitch

class GlitchInjector:
    """A module for injecting realistic glitches into GW strain data."""
    def __init__(self, glitch_types: list = None):
        if glitch_types is None:
            glitch_types = ['blip', 'whistle', 'koi_fish']
        self.glitch_types = glitch_types
        self.sample_rate = 4096.0  # Hz
    
    def inject_glitch(
        self, 
        signal: jnp.ndarray, 
        key: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Inject a random glitch into the signal.
        
        Args:
            signal: A 1D array of strain data.
            key: A JAX random key for stochasticity.
            
        Returns:
            A tuple of (augmented_signal, metadata).
        """
        # Choose a random glitch type
        key, subkey = jax.random.split(key)
        glitch_type_idx = jax.random.randint(subkey, (), 0, len(self.glitch_types))
        glitch_type = self.glitch_types[glitch_type_idx]
        
        # Generate the glitch waveform
        if glitch_type == 'blip':
            glitch = create_blip_glitch(0.1, self.sample_rate)  # 0.1 second duration
        elif glitch_type == 'whistle':
            glitch = create_whistle_glitch(0.5, self.sample_rate, 50.0, 300.0)
        elif glitch_type == 'koi_fish':
            glitch = create_koi_fish_glitch(0.3, self.sample_rate)
        else:
            # Fallback to a simple Gaussian noise burst
            glitch = create_blip_glitch(0.1, self.sample_rate)
        
        # Choose a random start time for the glitch within the signal
        key, subkey = jax.random.split(key)
        max_start_idx = len(signal) - len(glitch)
        start_idx = jax.random.randint(subkey, (), 0, max_start_idx + 1)
        
        # Choose a random amplitude for the glitch
        key, subkey = jax.random.split(key)
        # Scale the glitch amplitude relative to the signal's RMS
        signal_rms = jnp.sqrt(jnp.mean(signal**2))
        # Glitch amplitude between 0.5x and 2.0x the signal RMS
        glitch_amp = jax.random.uniform(subkey, (), minval=0.5, maxval=2.0) * signal_rms
        glitch = glitch * glitch_amp
        
        # Inject the glitch by adding it to the signal
        augmented_signal = signal.at[start_idx:start_idx+len(glitch)].add(glitch)
        
        # Create metadata
        metadata = {
            'type': glitch_type,
            'start_time': start_idx / self.sample_rate,
            'duration': len(glitch) / self.sample_rate,
            'amplitude': float(glitch_amp),
            'snr_boost': float(jnp.sum(glitch**2) / jnp.sum(signal**2))  # Approximate SNR contribution
        }
        
        return augmented_signal, metadata
```

## Usage

The `GlitchInjector` is used during the dataset creation process, typically in the `create_real_ligo_dataset` function or in the `run_advanced_pipeline.py` script.

```python
from data.glitch_injector import GlitchInjector

# Initialize the glitch injector
injector = GlitchInjector(glitch_types=['blip', 'whistle'])

# Example: Inject a glitch into a single training signal
key = jax.random.PRNGKey(42)
signal = jnp.array(train_signals[0])  # Example signal

augmented_signal, metadata = injector.inject_glitch(signal, key)
print(f"Injected a '{metadata['type']}' glitch at {metadata['start_time']:.3f}s")
print(f"Original signal RMS: {jnp.sqrt(jnp.mean(signal**2)):.2e}")
print(f"Augmented signal RMS: {jnp.sqrt(jnp.mean(augmented_signal**2)):.2e}")

# To augment the entire training set
augmented_signals = []
augmentation_metadata = []
for i, signal in enumerate(train_signals):
    key = jax.random.PRNGKey(i + 1000)  # Different key for each signal
    aug_signal, meta = injector.inject_glitch(signal, key)
    augmented_signals.append(aug_signal)
    augmentation_metadata.append(meta)

augmented_signals = jnp.array(augmented_signals)
print(f"Augmented {len(augmented_signals)} training signals with glitches")
```

By incorporating glitch injection into the training pipeline, the CPC-SNN-GW system becomes significantly more robust to the real-world noise conditions it will encounter, reducing the false alarm rate and improving its reliability as a detection system.