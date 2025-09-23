"""
SNN AutoEncoder Implementation based on LHC Paper

This module implements SNN AutoEncoder (SNN-AE) with reconstruction loss
based on Dillon et al. "Anomaly detection with spiking neural networks for LHC physics"
arXiv:2508.00063v1 [hep-ph] 31 Jul 2025

Key features:
- Encoder-Decoder architecture with spiking neurons
- MSE reconstruction loss (Equation 7 from paper)
- Multi-step processing with binary latent space
- Optimized for anomaly detection in GW signals

Architecture from paper:
- Encoder: Input → Hidden → Latent (binary spikes)
- Decoder: Latent → Hidden → Reconstruction
- Loss: MSE between input and reconstruction
"""

import logging
from typing import Tuple, Optional, Dict, Any, List
import jax
import jax.numpy as jnp
import flax.linen as nn

from .lhc_optimized import LHCOptimizedLIFLayer

logger = logging.getLogger(__name__)


class SNNEncoder(nn.Module):
    """
    SNN Encoder based on LHC paper methodology.
    
    Encodes input to binary latent space through spiking layers.
    Uses multi-step processing with same input each step.
    """
    
    hidden_sizes: List[int] = (64, 32)  # Hidden layer sizes
    latent_dim: int = 4  # Binary latent dimension
    time_steps: int = 5  # LHC optimal
    threshold: float = 1.2  # LHC optimal
    beta: float = 0.9  # LHC optimal
    
    def setup(self):
        """Initialize encoder layers."""
        # Create hidden layers
        layer_sizes = list(self.hidden_sizes)
        
        # Create layers as module attributes
        for i, size in enumerate(layer_sizes):
            layer = LHCOptimizedLIFLayer(
                features=size,
                time_steps=self.time_steps,
                threshold=self.threshold,
                beta=self.beta,
                name=f'encoder_layer_{i}'
            )
            setattr(self, f'hidden_layer_{i}', layer)
        
        # Latent layer (final encoding)
        self.latent_layer = LHCOptimizedLIFLayer(
            features=self.latent_dim,
            time_steps=self.time_steps,
            threshold=self.threshold,
            beta=self.beta,
            name='latent_layer'
        )
    
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Encode input to binary latent representation.
        
        Args:
            x: Input data [batch_size, input_features]
            
        Returns:
            Tuple of (latent_spikes, metrics)
        """
        current_input = x
        all_metrics = {}
        
        # Process through hidden layers
        for i in range(len(self.hidden_sizes)):
            layer = getattr(self, f'hidden_layer_{i}')
            current_input, layer_metrics = layer(current_input, is_output_layer=False)
            all_metrics[f'encoder_layer_{i}'] = layer_metrics
        
        # Final latent encoding
        latent_spikes, latent_metrics = self.latent_layer(current_input, is_output_layer=False)
        all_metrics['latent'] = latent_metrics
        
        # Aggregate metrics
        aggregate_metrics = self._aggregate_metrics(all_metrics)
        aggregate_metrics['latent_spikes_shape'] = latent_spikes.shape
        aggregate_metrics['binary_latent_configs'] = 2**(self.time_steps * self.latent_dim)
        
        return latent_spikes, aggregate_metrics
    
    def _aggregate_metrics(self, all_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across encoder layers."""
        spike_rates = [metrics['spike_rate_mean'] for metrics in all_metrics.values()]
        
        return {
            'encoder_avg_spike_rate': jnp.mean(jnp.array(spike_rates)),
            'encoder_layers': len(self.hidden_sizes) + 1,  # +1 for latent
            'encoder_time_steps': self.time_steps,
            'encoder_threshold': self.threshold
        }


class SNNDecoder(nn.Module):
    """
    SNN Decoder based on LHC paper methodology.
    
    Decodes binary latent spikes back to input reconstruction.
    Uses multi-step processing to accumulate reconstruction.
    """
    
    hidden_sizes: List[int] = (32, 64)  # Hidden layer sizes (reverse of encoder)
    output_dim: int = 1024  # Reconstruction output dimension
    time_steps: int = 5  # LHC optimal
    threshold: float = 1.2  # LHC optimal
    beta: float = 0.9  # LHC optimal
    
    def setup(self):
        """Initialize decoder layers."""
        # Create hidden layers
        layer_sizes = list(self.hidden_sizes)
        
        # Create layers as module attributes
        for i, size in enumerate(layer_sizes):
            layer = LHCOptimizedLIFLayer(
                features=size,
                time_steps=self.time_steps,
                threshold=self.threshold,
                beta=self.beta,
                name=f'decoder_layer_{i}'
            )
            setattr(self, f'hidden_layer_{i}', layer)
        
        # Output reconstruction layer (no spiking)
        self.output_layer = LHCOptimizedLIFLayer(
            features=self.output_dim,
            time_steps=self.time_steps,
            threshold=self.threshold * 10,  # Very high to prevent spiking
            beta=self.beta,
            name='decoder_output'
        )
    
    def __call__(self, latent_spikes: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Decode latent spikes to reconstruction.
        
        Args:
            latent_spikes: Binary latent spikes [batch_size, latent_dim]
            
        Returns:
            Tuple of (reconstruction, metrics)
        """
        current_input = latent_spikes
        all_metrics = {}
        
        # Process through hidden layers
        for i in range(len(self.hidden_sizes)):
            layer = getattr(self, f'hidden_layer_{i}')
            current_input, layer_metrics = layer(current_input, is_output_layer=False)
            all_metrics[f'decoder_layer_{i}'] = layer_metrics
        
        # Final reconstruction (no spiking)
        reconstruction, output_metrics = self.output_layer(current_input, is_output_layer=True)
        all_metrics['output'] = output_metrics
        
        # Aggregate metrics
        aggregate_metrics = self._aggregate_metrics(all_metrics)
        aggregate_metrics['reconstruction_shape'] = reconstruction.shape
        
        return reconstruction, aggregate_metrics
    
    def _aggregate_metrics(self, all_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across decoder layers."""
        spike_rates = [metrics['spike_rate_mean'] for key, metrics in all_metrics.items() 
                      if 'decoder_layer_' in key]  # Exclude output layer
        
        return {
            'decoder_avg_spike_rate': jnp.mean(jnp.array(spike_rates)) if spike_rates else 0.0,
            'decoder_layers': len(self.hidden_sizes),
            'decoder_time_steps': self.time_steps,
            'decoder_threshold': self.threshold
        }


class SNNAutoEncoder(nn.Module):
    """
    Complete SNN AutoEncoder based on LHC paper.
    
    Implements the full encoder-decoder architecture with:
    - Multi-step processing (T=5-10 optimal)
    - Binary latent space (2^(T*Dz) configurations)
    - MSE reconstruction loss
    - Optimized for anomaly detection
    
    Architecture:
    Input → SNN Encoder → Binary Latent → SNN Decoder → Reconstruction
    """
    
    # Architecture parameters
    encoder_hidden_sizes: List[int] = (64, 32)
    decoder_hidden_sizes: List[int] = (32, 64)
    latent_dim: int = 4
    input_dim: int = 1024
    
    # LHC optimal parameters
    time_steps: int = 5
    threshold: float = 1.2
    beta: float = 0.9
    
    def setup(self):
        """Initialize encoder and decoder."""
        self.encoder = SNNEncoder(
            hidden_sizes=self.encoder_hidden_sizes,
            latent_dim=self.latent_dim,
            time_steps=self.time_steps,
            threshold=self.threshold,
            beta=self.beta
        )
        
        self.decoder = SNNDecoder(
            hidden_sizes=self.decoder_hidden_sizes,
            output_dim=self.input_dim,
            time_steps=self.time_steps,
            threshold=self.threshold,
            beta=self.beta
        )
    
    def __call__(self, x: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Forward pass through SNN AutoEncoder.
        
        Args:
            x: Input data [batch_size, input_dim]
            training: Training mode flag
            
        Returns:
            Tuple of (reconstruction, comprehensive_metrics)
        """
        # Encode to binary latent space
        latent_spikes, encoder_metrics = self.encoder(x)
        
        # Decode to reconstruction
        reconstruction, decoder_metrics = self.decoder(latent_spikes)
        
        # Combine metrics
        combined_metrics = {
            **encoder_metrics,
            **decoder_metrics,
            'architecture': f"SNN-AE: {self.encoder_hidden_sizes} → {self.latent_dim} → {self.decoder_hidden_sizes}",
            'total_time_steps': self.time_steps,
            'total_threshold': self.threshold,
            'binary_latent_configs': 2**(self.time_steps * self.latent_dim),
            'input_shape': x.shape,
            'latent_shape': latent_spikes.shape,
            'reconstruction_shape': reconstruction.shape
        }
        
        return reconstruction, combined_metrics
    
    def encode(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Encode input to latent space only."""
        return self.encoder(x)
    
    def decode(self, latent_spikes: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Decode latent spikes to reconstruction only."""
        return self.decoder(latent_spikes)


def snn_autoencoder_loss(x_input: jnp.ndarray, x_reconstructed: jnp.ndarray, 
                        snn_metrics: Dict[str, Any], 
                        reconstruction_weight: float = 1.0,
                        spike_regularization_weight: float = 0.001,
                        target_spike_rate: float = 0.15) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    SNN AutoEncoder loss function based on LHC paper (Equation 7).
    
    Args:
        x_input: Original input
        x_reconstructed: Reconstructed output
        snn_metrics: Metrics from SNN forward pass
        reconstruction_weight: Weight for MSE reconstruction loss
        spike_regularization_weight: Weight for spike rate regularization
        target_spike_rate: Target spike rate (LHC paper: ~15%)
        
    Returns:
        Tuple of (total_loss, loss_components)
    """
    # ✅ LHC LOSS: MSE reconstruction loss (Equation 7 from paper)
    mse_loss = jnp.mean((x_input - x_reconstructed) ** 2)
    
    # ✅ LHC REGULARIZATION: Spike rate regularization
    encoder_spike_rate = snn_metrics.get('encoder_avg_spike_rate', 0.0)
    decoder_spike_rate = snn_metrics.get('decoder_avg_spike_rate', 0.0)
    avg_spike_rate = (encoder_spike_rate + decoder_spike_rate) / 2
    
    spike_reg_loss = spike_regularization_weight * jnp.abs(avg_spike_rate - target_spike_rate)
    
    # Total loss
    total_loss = reconstruction_weight * mse_loss + spike_reg_loss
    
    # Loss components for monitoring
    loss_components = {
        'total_loss': total_loss,
        'mse_reconstruction_loss': mse_loss,
        'spike_regularization_loss': spike_reg_loss,
        'encoder_spike_rate': encoder_spike_rate,
        'decoder_spike_rate': decoder_spike_rate,
        'avg_spike_rate': avg_spike_rate,
        'target_spike_rate': target_spike_rate,
        'binary_latent_configs': snn_metrics.get('binary_latent_configs', 0)
    }
    
    return total_loss, loss_components


def create_snn_autoencoder(
    input_dim: int = 1024,
    latent_dim: int = 4,
    encoder_hidden_sizes: List[int] = (64, 32),
    decoder_hidden_sizes: List[int] = (32, 64),
    time_steps: int = 5,
    threshold: float = 1.2,
    beta: float = 0.9
) -> SNNAutoEncoder:
    """
    Factory function to create SNN AutoEncoder with LHC-optimized parameters.
    
    Args:
        input_dim: Input feature dimension
        latent_dim: Latent space dimension (binary)
        encoder_hidden_sizes: Encoder layer sizes
        decoder_hidden_sizes: Decoder layer sizes (should be reverse of encoder)
        time_steps: Number of processing steps (LHC optimal: 5-10)
        threshold: Spike threshold (LHC optimal: 1.2)
        beta: Membrane decay factor (LHC optimal: 0.9)
        
    Returns:
        Configured SNN AutoEncoder
    """
    # ✅ LHC VALIDATION: Ensure parameters are in optimal ranges
    if not (5 <= time_steps <= 10):
        logger.warning(f"Time steps {time_steps} outside LHC optimal range [5,10]")
    
    if not (1.0 <= threshold <= 1.5):
        logger.warning(f"Threshold {threshold} outside LHC optimal range [1.0,1.5]")
    
    if latent_dim > 8:
        logger.warning(f"Latent dim {latent_dim} may be too large for efficient binary space")
    
    # Validate encoder/decoder symmetry
    if list(reversed(encoder_hidden_sizes)) != list(decoder_hidden_sizes):
        logger.warning("Decoder sizes should be reverse of encoder for symmetric architecture")
    
    logger.info("Creating SNN AutoEncoder with LHC-optimized parameters:")
    logger.info(f"  📊 Architecture: {input_dim} → {encoder_hidden_sizes} → {latent_dim} → {decoder_hidden_sizes} → {input_dim}")
    logger.info(f"  ⚡ Time steps: {time_steps} (LHC optimal)")
    logger.info(f"  🎯 Threshold: {threshold} (LHC optimal)")
    logger.info(f"  🔄 Beta decay: {beta} (LHC optimal)")
    logger.info(f"  🔥 Binary latent configs: {2**(time_steps * latent_dim):,}")
    
    return SNNAutoEncoder(
        encoder_hidden_sizes=encoder_hidden_sizes,
        decoder_hidden_sizes=decoder_hidden_sizes,
        latent_dim=latent_dim,
        input_dim=input_dim,
        time_steps=time_steps,
        threshold=threshold,
        beta=beta
    )


# ✅ LHC ANOMALY DETECTION UTILITIES
def calculate_anomaly_score(x_input: jnp.ndarray, x_reconstructed: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate anomaly score based on reconstruction error (LHC methodology).
    
    Higher reconstruction error = higher anomaly score = more likely to be anomalous.
    
    Args:
        x_input: Original input
        x_reconstructed: Reconstructed output
        
    Returns:
        Anomaly scores per sample
    """
    # Per-sample MSE reconstruction error
    reconstruction_errors = jnp.mean((x_input - x_reconstructed) ** 2, axis=-1)
    
    # Normalize to [0, 1] range for interpretation
    max_error = jnp.max(reconstruction_errors)
    min_error = jnp.min(reconstruction_errors)
    
    if max_error > min_error:
        anomaly_scores = (reconstruction_errors - min_error) / (max_error - min_error)
    else:
        anomaly_scores = jnp.zeros_like(reconstruction_errors)
    
    return anomaly_scores


def snn_ae_anomaly_detection(
    snn_ae: SNNAutoEncoder,
    params: Dict,
    test_data: jnp.ndarray,
    threshold_percentile: float = 95.0
) -> Dict[str, Any]:
    """
    Perform anomaly detection using SNN AutoEncoder (LHC methodology).
    
    Args:
        snn_ae: Trained SNN AutoEncoder
        params: Model parameters
        test_data: Test data for anomaly detection
        threshold_percentile: Percentile threshold for anomaly detection
        
    Returns:
        Anomaly detection results
    """
    # Forward pass
    reconstructions, metrics = snn_ae.apply(params, test_data, training=False)
    
    # Calculate anomaly scores
    anomaly_scores = calculate_anomaly_score(test_data, reconstructions)
    
    # Determine threshold
    threshold = jnp.percentile(anomaly_scores, threshold_percentile)
    
    # Binary anomaly predictions
    is_anomaly = anomaly_scores > threshold
    
    return {
        'anomaly_scores': anomaly_scores,
        'threshold': threshold,
        'is_anomaly': is_anomaly,
        'num_anomalies': jnp.sum(is_anomaly),
        'anomaly_rate': jnp.mean(is_anomaly),
        'reconstruction_mse': jnp.mean((test_data - reconstructions) ** 2),
        'snn_metrics': metrics
    }
