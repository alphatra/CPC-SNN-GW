#!/usr/bin/env python3
"""
🚨 EMERGENCY DEBUGGING: AResGW-style simple model for comparison

Test EXACT AResGW architecture in JAX/Flax to see if the fundamental 
approach works with CPC-SNN-GW data pipeline.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import logging
from pathlib import Path
import time
import numpy as np

logger = logging.getLogger(__name__)

class ResBlock(nn.Module):
    """Equivalent to AResGW ResBlock but in JAX/Flax."""
    
    out_channels: int
    stride: int = 1
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        in_channels = x.shape[-1]
        
        # Skip connection
        if self.out_channels != in_channels or self.stride > 1:
            skip = nn.Conv(self.out_channels, kernel_size=(1,), strides=(self.stride,))(x)
        else:
            skip = x
        
        # Main path
        out = nn.Conv(self.out_channels, kernel_size=(3,), strides=(1,), padding='SAME')(x)
        out = nn.LayerNorm()(out)  # Use LayerNorm instead of BatchNorm for simplicity
        out = nn.gelu(out)
        out = nn.Conv(self.out_channels, kernel_size=(3,), strides=(self.stride,), padding=1)(out)
        out = nn.LayerNorm()(out)
        
        return nn.gelu(out + skip)

class SimpleResNet(nn.Module):
    """Simple ResNet equivalent to working AResGW architecture."""
    
    num_classes: int = 2
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        """Forward pass with AResGW-like architecture."""
        
        # Input: [batch, seq_len] → [batch, seq_len, channels]
        if len(x.shape) == 2:
            # Simulate dual-channel like AResGW (H1, L1)
            x_dual = jnp.stack([x, x], axis=-1)  # [batch, seq_len, 2]
        else:
            x_dual = x
        
        # Feature extraction (simplified ResNet54)
        x = ResBlock(16)(x_dual, train)
        x = ResBlock(32, stride=2)(x, train)
        x = ResBlock(64, stride=2)(x, train)
        x = ResBlock(128, stride=2)(x, train)
        x = ResBlock(64)(x, train)
        x = ResBlock(32)(x, train)
        
        # Global pooling + classification (like AResGW)
        pooled = jnp.mean(x, axis=1)  # Global average pooling
        logits = nn.Dense(self.num_classes)(pooled)
        
        # Softmax like AResGW
        return nn.softmax(logits)

def aresgw_style_loss_fn(predictions, labels):
    """AResGW-style BCE loss equivalent."""
    # Convert to one-hot
    labels_onehot = jax.nn.one_hot(labels, predictions.shape[-1])
    
    # BCE equivalent (with epsilon regularization like AResGW)
    epsilon = 1e-6
    dim = predictions.shape[-1]
    reg_A = epsilon  
    reg_B = 1. - epsilon * dim
    transformed_pred = reg_A + reg_B * predictions
    
    # BCE loss
    loss = -jnp.mean(
        labels_onehot * jnp.log(transformed_pred + 1e-8) + 
        (1 - labels_onehot) * jnp.log(1 - transformed_pred + 1e-8)
    )
    
    return loss

def debug_aresgw_style():
    """Debug training with AResGW-style simple model."""
    
    logger.info("🚨 DEBUGGING: Testing AResGW-style simple ResNet")
    
    # Use SAME data as CPC-SNN-GW for fair comparison
    try:
        from data.real_ligo_integration import create_real_ligo_dataset
        (signals, labels), (test_signals, test_labels) = create_real_ligo_dataset(
            num_samples=2000,  # Much larger for proper training
            window_size=256,
            quick_mode=False,  # Use enhanced mode for more data
            return_split=True,
            train_ratio=0.8,
            overlap=0.8  # Higher overlap for more windows
        )
        logger.info(f"📊 Real LIGO data: {signals.shape}, labels: {jnp.bincount(labels)}")
    except Exception as e:
        logger.warning(f"Real LIGO unavailable: {e}")
        # Fallback synthetic
        key = jax.random.PRNGKey(42)
        signals = jax.random.normal(key, (100, 256)) * 1e-21
        labels = jax.random.randint(key, (100,), 0, 2)
        test_signals = jax.random.normal(key, (25, 256)) * 1e-21  
        test_labels = jax.random.randint(key, (25,), 0, 2)
    
    # Create AResGW-style model
    model = SimpleResNet(num_classes=2)
    
    # Initialize
    key = jax.random.PRNGKey(42)
    sample_input = signals[:1]
    params = model.init(key, sample_input, train=True)
    
    # AResGW-style optimizer (Adam with proper LR)
    optimizer = optax.adam(learning_rate=5e-5)  # Same as AResGW default
    train_state_obj = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    logger.info("🚀 Starting AResGW-style debug training...")
    
    batch_size = 8
    best_acc = 0.0
    
    for epoch in range(15):
        epoch_losses = []
        epoch_accs = []
        
        # Shuffle data
        key = jax.random.split(key)[0]
        indices = jax.random.permutation(key, len(signals))
        shuffled_signals = signals[indices]
        shuffled_labels = labels[indices]
        
        # Training batches
        for i in range(0, len(signals), batch_size):
            batch_x = shuffled_signals[i:i+batch_size]
            batch_y = shuffled_labels[i:i+batch_size]
            
            # Training step (AResGW style)
            def loss_fn(params):
                predictions = train_state_obj.apply_fn(params, batch_x, train=True)
                loss = aresgw_style_loss_fn(predictions, batch_y)
                
                # Accuracy calculation
                pred_labels = jnp.argmax(predictions, axis=-1)
                accuracy = jnp.mean(pred_labels == batch_y)
                return loss, accuracy
            
            (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state_obj.params)
            train_state_obj = train_state_obj.apply_gradients(grads=grads)
            
            epoch_losses.append(float(loss))
            epoch_accs.append(float(accuracy))
        
        # Test evaluation
        test_predictions = train_state_obj.apply_fn(train_state_obj.params, test_signals, train=False)
        test_pred_labels = jnp.argmax(test_predictions, axis=-1)  
        test_acc = float(jnp.mean(test_pred_labels == test_labels))
        
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        avg_acc = jnp.mean(jnp.array(epoch_accs))
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        logger.info(f"Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, Train Acc={avg_acc:.3f}, Test Acc={test_acc:.3f}, Best={best_acc:.3f}")
        
        # Check learning progress
        if epoch > 5 and avg_acc > 0.8:
            logger.info("✅ ARESGW-STYLE MODEL LEARNS SUCCESSFULLY!")
            logger.info("🎯 DIAGNOSIS: Problem is in CPC-SNN complexity")
            return True
            
        if epoch > 8 and avg_acc < 0.55:
            logger.info("❌ EVEN SIMPLE MODEL FAILS - Problem is in data or fundamental setup")
            return False
    
    if best_acc > 0.7:
        logger.info("✅ SIMPLE MODEL EVENTUALLY LEARNS")
        logger.info("🎯 DIAGNOSIS: CPC-SNN architecture too complex")
        return True
    else:
        logger.info("❌ SIMPLE MODEL FAILS TOO")  
        logger.info("🎯 DIAGNOSIS: Data or preprocessing issue")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Run debug
    success = debug_aresgw_style()
    
    if success:
        print("\n" + "="*60)
        print("🎯 DIAGNOSIS: CPC-SNN ARCHITECTURE IS TOO COMPLEX")
        print("="*60)
        print("SOLUTIONS:")
        print("1. 🔧 Start with simple CPC encoder + direct classifier")
        print("2. ⚡ Add spike bridge gradually after basic training works") 
        print("3. 🧠 Use proven ResNet backbone instead of custom CPC")
        print("4. 📉 Reduce model complexity and focus on working baseline")
        print("5. 🎯 Copy AResGW architecture exactly, then neuromorphic features")
        
    else:
        print("\n" + "="*60)
        print("🚨 DIAGNOSIS: FUNDAMENTAL DATA/PREPROCESSING ISSUE")
        print("="*60)
        print("SOLUTIONS:")
        print("1. 📊 Check data preprocessing - add whitening like AResGW")
        print("2. 🔍 Validate data quality - ensure proper GW signals")
        print("3. ⚙️ Copy AResGW data pipeline exactly")
        print("4. 📈 Check learning rate and optimizer settings")
        print("5. 🧹 Debug gradient computation and parameter updates")
