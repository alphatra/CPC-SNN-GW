#!/usr/bin/env python3
"""
🚨 EMERGENCY DEBUGGING: Simplified CPC model without spike bridge

Test if problem is in:
1. CPC encoder itself
2. Spike bridge
3. SNN classifier
4. Loss function
5. Data preprocessing

This bypasses spike bridge to isolate the issue.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class SimpleCPCModel(nn.Module):
    """Simplified CPC model bypassing spike bridge for debugging."""
    
    latent_dim: int = 64
    num_classes: int = 2
    
    def setup(self):
        # Simple CPC encoder (minimal)
        self.conv1 = nn.Conv(64, kernel_size=(7,), strides=(2,), padding='SAME')
        self.conv2 = nn.Conv(128, kernel_size=(5,), strides=(2,), padding='SAME') 
        self.conv3 = nn.Conv(self.latent_dim, kernel_size=(3,), padding='SAME')
        
        # Direct classifier (bypass spike bridge)
        self.classifier = nn.Dense(self.num_classes)
        
    @nn.compact  
    def __call__(self, x, train: bool = True, return_intermediates: bool = False):
        """Simple forward pass without spike encoding."""
        
        # Add channel dimension if needed
        if len(x.shape) == 2:
            x = x[..., None]  # [batch, time, 1]
        
        # Simple conv stack
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))  
        x = nn.gelu(self.conv3(x))
        
        # Global average pooling
        pooled = jnp.mean(x, axis=1)  # [batch, latent_dim]
        
        # Direct classification
        logits = self.classifier(pooled)
        
        if return_intermediates:
            return {
                'logits': logits,
                'cpc_features': x,
                'pooled_features': pooled
            }
        else:
            return logits

def simple_loss_fn(logits, labels):
    """Simple classification loss - identical to AResGW."""
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

def debug_simple_training():
    """Debug training with simplified model."""
    
    logger.info("🚨 EMERGENCY DEBUGGING: Testing simplified CPC model")
    
    # Generate synthetic data (same as CPC-SNN-GW)
    key = jax.random.PRNGKey(42)
    batch_size = 8
    seq_len = 256
    
    # Create synthetic GW-like data
    x = jax.random.normal(key, (100, seq_len)) * 1e-21
    y = jax.random.randint(key, (100,), 0, 2)
    
    logger.info(f"📊 Debug data: {x.shape}, labels: {jnp.bincount(y)}")
    
    # Create simple model
    model = SimpleCPCModel(latent_dim=64, num_classes=2)
    
    # Initialize
    key = jax.random.PRNGKey(42)
    sample_input = x[:1]
    params = model.init(key, sample_input, train=True)
    
    # Simple optimizer (same as AResGW)
    optimizer = optax.adam(learning_rate=1e-4)  # Lower LR
    train_state_obj = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    # Training loop
    logger.info("🚀 Starting debug training...")
    
    for epoch in range(10):
        epoch_losses = []
        epoch_accs = []
        
        # Simple batching
        for i in range(0, len(x), batch_size):
            batch_x = x[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            # Training step
            def loss_fn(params):
                logits = train_state_obj.apply_fn(params, batch_x, train=True)
                loss = simple_loss_fn(logits, batch_y)
                accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch_y)
                return loss, accuracy
            
            (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state_obj.params)
            train_state_obj = train_state_obj.apply_gradients(grads=grads)
            
            epoch_losses.append(float(loss))
            epoch_accs.append(float(accuracy))
        
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        avg_acc = jnp.mean(jnp.array(epoch_accs))
        
        logger.info(f"Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, Acc={avg_acc:.3f}")
        
        # Check if learning
        if epoch > 2 and avg_acc > 0.7:
            logger.info("✅ SIMPLIFIED MODEL LEARNS - Problem is in spike bridge/SNN!")
            return True
        elif epoch > 5 and avg_acc < 0.55:
            logger.info("❌ SIMPLIFIED MODEL DOESN'T LEARN - Problem is in CPC encoder/data!")
            return False
    
    logger.info(f"🤔 UNCLEAR RESULT - Final acc={avg_acc:.3f}")
    return avg_acc > 0.6

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    success = debug_simple_training()
    if success:
        print("\n✅ DIAGNOSIS: Problem is in SPIKE BRIDGE or SNN components")
        print("SOLUTION: Fix surrogate gradients, spike encoding, or SNN architecture")
    else:
        print("\n❌ DIAGNOSIS: Problem is in CPC ENCODER or DATA preprocessing") 
        print("SOLUTION: Fix CPC architecture, InfoNCE loss, or data pipeline")
