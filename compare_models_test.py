#!/usr/bin/env python3
"""
🔍 MODEL COMPARISON TEST

Compare Simple Model (working) vs CPC-SNN Model (broken) 
to identify root cause of gradient flow issues.
"""

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from training.base_trainer import CPCSNNTrainer, TrainingConfig
import time

print("🔍 MODEL COMPARISON TEST: Simple vs CPC-SNN")
print("=" * 60)

# Test data - same for both models
x = jax.random.normal(jax.random.PRNGKey(123), (1, 512))
y = jnp.array([1])

print(f"📊 Test data: x{x.shape}, y{y.shape}")

# ======================================
# TEST 1: SIMPLE MODEL (WORKING)
# ======================================
print(f"\n🔹 TEST 1: SIMPLE MODEL")
print("-" * 40)

class SimpleModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(2)(x)
        return x

# Initialize simple model
simple_model = SimpleModel()
key = jax.random.PRNGKey(42)
simple_params = simple_model.init(key, x)

def simple_loss_fn(params):
    logits = simple_model.apply(params, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss, logits

# Test simple model gradients
(simple_loss, simple_logits), simple_grads = jax.value_and_grad(simple_loss_fn, has_aux=True)(simple_params)
simple_grad_norm = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda g: jnp.sum(g**2), simple_grads))))

print(f"   Loss: {simple_loss:.6f}")
print(f"   Logits: {simple_logits}")
print(f"   Gradient norm: {simple_grad_norm:.6f}")

# Test simple model parameter update
simple_optimizer = optax.adam(learning_rate=1e-2)
simple_opt_state = simple_optimizer.init(simple_params)
simple_updates, _ = simple_optimizer.update(simple_grads, simple_opt_state, simple_params)
simple_new_params = optax.apply_updates(simple_params, simple_updates)

simple_param_change = sum(jax.tree.leaves(jax.tree.map(
    lambda old, new: jnp.linalg.norm(new - old), 
    simple_params, simple_new_params
)))

print(f"   Parameter change: {simple_param_change:.10f}")
print(f"   ✅ Simple model: {'WORKING' if simple_param_change > 1e-8 else 'BROKEN'}")

# ======================================  
# TEST 2: CPC-SNN MODEL (INVESTIGATION)
# ======================================
print(f"\n🔹 TEST 2: CPC-SNN MODEL")
print("-" * 40)

# Create CPC-SNN trainer and model
config = TrainingConfig(
    batch_size=1,
    learning_rate=1e-2,
    num_epochs=1, 
    num_classes=2,
    use_wandb=False,
    use_tensorboard=False,
    gradient_clipping=100.0,
    optimizer="adam"
)

trainer = CPCSNNTrainer(config)
cpc_model = trainer.create_model()

# Create training state
cpc_train_state = trainer.create_train_state(cpc_model, x)

print(f"   Model type: {type(cpc_model)}")
print(f"   Optimizer type: {type(cpc_train_state.tx)}")

# Test CPC-SNN model gradients - VERSION 1: Simple approach
print(f"\n   🔹 Version 1: Simple Forward Pass")
def cpc_loss_fn_v1(params):
    logits = cpc_train_state.apply_fn(params, x, train=False)  # No RNG for consistency
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss, logits

try:
    (cpc_loss_v1, cpc_logits_v1), cpc_grads_v1 = jax.value_and_grad(cpc_loss_fn_v1, has_aux=True)(cpc_train_state.params)
    cpc_grad_norm_v1 = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda g: jnp.sum(g**2), cpc_grads_v1))))
    
    print(f"     Loss: {cpc_loss_v1:.6f}")
    print(f"     Logits: {cpc_logits_v1}")
    print(f"     Gradient norm: {cpc_grad_norm_v1:.6f}")
    
    # Test parameter update V1
    cpc_updates_v1, _ = cpc_train_state.tx.update(cpc_grads_v1, cpc_train_state.opt_state, cpc_train_state.params)
    cpc_new_params_v1 = optax.apply_updates(cpc_train_state.params, cpc_updates_v1)
    
    cpc_param_change_v1 = sum(jax.tree.leaves(jax.tree.map(
        lambda old, new: jnp.linalg.norm(new - old), 
        cpc_train_state.params, cpc_new_params_v1
    )))
    
    print(f"     Parameter change: {cpc_param_change_v1:.10f}")
    print(f"     ✅ CPC-SNN V1: {'WORKING' if cpc_param_change_v1 > 1e-8 else 'BROKEN'}")
    
except Exception as e:
    print(f"     ❌ CPC-SNN V1 FAILED: {e}")

# Test CPC-SNN model gradients - VERSION 2: With RNG (original approach)
print(f"\n   🔹 Version 2: With RNG (Original)")
def cpc_loss_fn_v2(params):
    logits = cpc_train_state.apply_fn(
        params, x, train=True,
        rngs={'spike_bridge': jax.random.PRNGKey(12345)}  # Fixed RNG for consistency
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss, logits

try:
    (cpc_loss_v2, cpc_logits_v2), cpc_grads_v2 = jax.value_and_grad(cpc_loss_fn_v2, has_aux=True)(cpc_train_state.params)
    cpc_grad_norm_v2 = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda g: jnp.sum(g**2), cpc_grads_v2))))
    
    print(f"     Loss: {cpc_loss_v2:.6f}")
    print(f"     Logits: {cpc_logits_v2}")
    print(f"     Gradient norm: {cpc_grad_norm_v2:.6f}")
    
    # Test parameter update V2  
    cpc_updates_v2, _ = cpc_train_state.tx.update(cpc_grads_v2, cpc_train_state.opt_state, cpc_train_state.params)
    cpc_new_params_v2 = optax.apply_updates(cpc_train_state.params, cpc_updates_v2)
    
    cpc_param_change_v2 = sum(jax.tree.leaves(jax.tree.map(
        lambda old, new: jnp.linalg.norm(new - old), 
        cpc_train_state.params, cpc_new_params_v2
    )))
    
    print(f"     Parameter change: {cpc_param_change_v2:.10f}")
    print(f"     ✅ CPC-SNN V2: {'WORKING' if cpc_param_change_v2 > 1e-8 else 'BROKEN'}")
    
except Exception as e:
    print(f"     ❌ CPC-SNN V2 FAILED: {e}")

# Test CPC-SNN model gradients - VERSION 3: Train step approach
print(f"\n   🔹 Version 3: Train Step (Current Pipeline)")
try:
    batch = (x, y)
    old_params = cpc_train_state.params
    old_param_norm = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda p: jnp.sum(p**2), old_params))))
    
    print(f"     Initial param norm: {old_param_norm:.6f}")
    
    # Use actual train_step
    new_train_state, metrics, enhanced_data = trainer.train_step(cpc_train_state, batch)
    
    new_params = new_train_state.params
    new_param_norm = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda p: jnp.sum(p**2), new_params))))
    
    param_change_v3 = sum(jax.tree.leaves(jax.tree.map(
        lambda old, new: jnp.linalg.norm(new - old), 
        old_params, new_params
    )))
    
    print(f"     Final param norm: {new_param_norm:.6f}")
    print(f"     Parameter change: {param_change_v3:.10f}")
    print(f"     Loss: {metrics.loss:.6f}")
    print(f"     Grad norm: {metrics.grad_norm:.6f}")
    print(f"     ✅ Train Step: {'WORKING' if param_change_v3 > 1e-8 else 'BROKEN'}")
    
except Exception as e:
    print(f"     ❌ Train Step FAILED: {e}")

print(f"\n🔍 COMPARISON SUMMARY")
print("=" * 60)
print(f"Simple Model Parameter Change: {simple_param_change:.10f}")
if 'cpc_param_change_v1' in locals():
    print(f"CPC-SNN V1 Parameter Change:   {cpc_param_change_v1:.10f}")
if 'cpc_param_change_v2' in locals():
    print(f"CPC-SNN V2 Parameter Change:   {cpc_param_change_v2:.10f}")
if 'param_change_v3' in locals():
    print(f"CPC-SNN V3 Parameter Change:   {param_change_v3:.10f}")

print(f"\n🎯 ROOT CAUSE ANALYSIS COMPLETE") 