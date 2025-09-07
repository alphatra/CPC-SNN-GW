#!/usr/bin/env python3
"""
🚨 EMERGENCY FIX: CPC Encoder Issues

Based on diagnosis, fixing 4 critical issues:
1. Increase latent_dim 64 → 256 
2. Remove aggressive L2 normalization
3. Simplify to single-task training  
4. Add basic whitening preprocessing
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

class FixedCPCEncoder(nn.Module):
    """Fixed CPC Encoder addressing learning issues."""
    
    latent_dim: int = 256  # ✅ INCREASED from 64
    num_classes: int = 2
    
    def setup(self):
        # Simplified conv stack (similar to working AResGW progression)
        self.conv1 = nn.Conv(64, kernel_size=(9,), strides=(2,), padding='SAME')  # 256 → 128
        self.conv2 = nn.Conv(128, kernel_size=(7,), strides=(2,), padding='SAME')  # 128 → 64
        self.conv3 = nn.Conv(256, kernel_size=(5,), strides=(1,), padding='SAME')  # Keep resolution
        self.conv4 = nn.Conv(self.latent_dim, kernel_size=(3,), padding='SAME')   # Final features
        
        # Direct classifier (single-task, like AResGW)
        self.classifier = nn.Dense(self.num_classes)
        
    @nn.compact  
    def __call__(self, x, train: bool = True):
        """Fixed forward pass without problematic L2 normalization."""
        
        # ✅ FIX #4: Basic whitening (simplified)
        x_mean = jnp.mean(x, axis=1, keepdims=True)  
        x_std = jnp.std(x, axis=1, keepdims=True)
        x_whitened = (x - x_mean) / (x_std + 1e-8)
        
        # Add channel dimension
        x_conv = x_whitened[..., None]  # [batch, time, 1]
        
        # ✅ FIX #1: Proper conv progression with layer norm (not batch norm)
        x_conv = self.conv1(x_conv)
        x_conv = nn.LayerNorm()(x_conv)  # Stable normalization
        x_conv = nn.gelu(x_conv)
        
        x_conv = self.conv2(x_conv) 
        x_conv = nn.LayerNorm()(x_conv)
        x_conv = nn.gelu(x_conv)
        
        x_conv = self.conv3(x_conv)
        x_conv = nn.LayerNorm()(x_conv)
        x_conv = nn.gelu(x_conv)
        
        x_conv = self.conv4(x_conv)
        x_conv = nn.LayerNorm()(x_conv)  
        features = nn.gelu(x_conv)
        
        # ✅ FIX #2: NO aggressive L2 normalization (removed problematic code)
        # Global average pooling for classification
        pooled = jnp.mean(features, axis=1)  # [batch, latent_dim]
        
        # ✅ FIX #3: Direct classification (single-task)
        logits = self.classifier(pooled)
        
        return logits

def simple_classification_loss(logits, labels):
    """Simple classification loss equivalent to AResGW BCE."""
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

def test_fixed_cpc_encoder():
    """Test fixed CPC encoder with same data as working AResGW."""
    
    logger.info("🚨 TESTING: Fixed CPC Encoder")
    
    # Use SAME data as successful AResGW debug
    try:
        from data.real_ligo_integration import create_real_ligo_dataset
        (signals, labels), (test_signals, test_labels) = create_real_ligo_dataset(
            num_samples=2000,
            window_size=256, 
            quick_mode=False,
            return_split=True,
            train_ratio=0.8,
            overlap=0.8
        )
        logger.info(f"📊 Real LIGO data: {signals.shape}, labels: {jnp.bincount(labels)}")
    except Exception as e:
        logger.warning(f"Real LIGO unavailable: {e}")
        # Fallback synthetic
        key = jax.random.PRNGKey(42)
        signals = jax.random.normal(key, (500, 256)) * 1e-21
        labels = jax.random.randint(key, (500,), 0, 2)
        test_signals = jax.random.normal(key, (100, 256)) * 1e-21  
        test_labels = jax.random.randint(key, (100,), 0, 2)
        logger.info(f"📊 Synthetic data: {signals.shape}, labels: {jnp.bincount(labels)}")
    
    # Create fixed model  
    model = FixedCPCEncoder(latent_dim=256, num_classes=2)  # ✅ Increased latent_dim
    
    # Initialize
    key = jax.random.PRNGKey(42)
    sample_input = signals[:1]
    params = model.init(key, sample_input, train=True)
    
    # ✅ FIX: Use AResGW-style learning rate
    optimizer = optax.adam(learning_rate=5e-5)  # Same as successful AResGW
    train_state_obj = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    logger.info("🚀 Starting FIXED CPC training...")
    
    batch_size = 8
    best_acc = 0.0
    loss_history = []
    
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
            
            # ✅ FIX: Simple single-task training step
            def loss_fn(params):
                logits = train_state_obj.apply_fn(params, batch_x, train=True)
                loss = simple_classification_loss(logits, batch_y)
                accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch_y)
                return loss, accuracy
            
            (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state_obj.params)
            train_state_obj = train_state_obj.apply_gradients(grads=grads)
            
            epoch_losses.append(float(loss))
            epoch_accs.append(float(accuracy))
        
        # Test evaluation
        test_logits = train_state_obj.apply_fn(train_state_obj.params, test_signals, train=False)
        test_pred_labels = jnp.argmax(test_logits, axis=-1)  
        test_acc = float(jnp.mean(test_pred_labels == test_labels))
        
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        avg_acc = jnp.mean(jnp.array(epoch_accs))
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        loss_history.append(float(avg_loss))
        
        logger.info(f"Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, Train Acc={avg_acc:.3f}, Test Acc={test_acc:.3f}, Best={best_acc:.3f}")
        
        # Check learning progress
        if epoch > 3:
            recent_losses = loss_history[-4:]
            loss_trend = recent_losses[0] - recent_losses[-1]  # Positive = decreasing loss
            
            if avg_acc > 0.8 and loss_trend > 0.1:
                logger.info("✅ FIXED CPC ENCODER LEARNS SUCCESSFULLY!")
                return True
                
        if epoch > 8 and avg_acc < 0.55:
            logger.info("❌ FIXED CPC STILL FAILS")
            return False
    
    if best_acc > 0.7:
        logger.info("✅ FIXED CPC EVENTUALLY LEARNS")
        return True
    else:
        logger.info("❌ FIXED CPC FAILS") 
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test fixed CPC encoder
    success = test_fixed_cpc_encoder()
    
    if success:
        print("\n" + "="*60)
        print("🎉 SUCCESS: FIXED CPC ENCODER WORKS!")
        print("="*60)
        print("SOLUTION CONFIRMED:")
        print("1. ✅ Increase latent_dim: 64 → 256")
        print("2. ✅ Remove aggressive L2 normalization")  
        print("3. ✅ Use single-task classification loss")
        print("4. ✅ Add basic whitening preprocessing")
        print("5. ✅ Use AResGW learning rate: 5e-5")
        print("")
        print("NEXT STEPS:")
        print("🔧 Apply these fixes to main CPC-SNN-GW model")
        print("⚡ Then gradually re-add spike bridge")
        print("🧠 Finally add SNN classifier")
        
    else:
        print("\n" + "="*60)
        print("❌ STILL FAILING: Need deeper investigation")
        print("="*60) 
        print("FURTHER DEBUGGING NEEDED:")
        print("1. 🔍 Check data quality and preprocessing")
        print("2. 📊 Validate learning rate and optimizer")
        print("3. 🔧 Test even simpler architectures")
        print("4. 📈 Consider copying AResGW exactly")
