#!/usr/bin/env python3
"""
🔍 GRADIENT FLOW DEBUGGING SCRIPT

Specialized tool to diagnose why parameters aren't updating despite non-zero gradients.
Examines optimizer state, gradient magnitudes, and parameter update mechanics.
"""

import os
import logging
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from training.base_trainer import CPCSNNTrainer, TrainingConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_gradient_flow():
    """Perform detailed gradient flow analysis."""
    logger.info("🔍 ANALYZING GRADIENT FLOW ISSUES")
    logger.info("=" * 80)
    
    # Create trainer with debug config
    config = TrainingConfig(
        batch_size=1,
        learning_rate=1e-2,  # High LR for visible changes
        num_epochs=1,
        num_classes=2,
        use_wandb=False,
        use_tensorboard=False,
        gradient_clipping=100.0,  # Very high to avoid clipping
        optimizer="adam"  # Adam for debugging
    )
    
    trainer = CPCSNNTrainer(config)
    model = trainer.create_model()
    
    # Create training state
    sample_input = jax.random.normal(jax.random.PRNGKey(42), (1, 512))
    train_state_obj = trainer.create_train_state(model, sample_input)
    
    logger.info(f"✅ Initial setup complete")
    logger.info(f"   Optimizer: {config.optimizer}")
    logger.info(f"   Learning rate: {config.learning_rate}")
    logger.info(f"   Gradient clipping: {config.gradient_clipping}")
    
    # Create test batch
    x = jax.random.normal(jax.random.PRNGKey(45), (1, 512))
    y = jnp.array([1])
    
    logger.info("\n🔍 EXAMINING GRADIENTS")
    logger.info("-" * 50)
    
    # Define loss function for gradient analysis
    def detailed_loss_fn(params):
        logits = train_state_obj.apply_fn(
            params, x, train=True,
            rngs={'spike_bridge': jax.random.PRNGKey(12345)}
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        
        # Return logits for analysis
        return loss, {'logits': logits, 'loss': loss}
    
    # Compute gradients with auxiliary data
    (loss_value, aux_data), grads = jax.value_and_grad(detailed_loss_fn, has_aux=True)(train_state_obj.params)
    
    logger.info(f"   Loss value: {loss_value:.6f}")
    logger.info(f"   Logits: {aux_data['logits']}")
    logger.info(f"   Logits shape: {aux_data['logits'].shape}")
    
    # Analyze gradient magnitudes
    logger.info("\n🔍 GRADIENT ANALYSIS")
    logger.info("-" * 50)
    
    grad_norms = {}
    total_grad_norm = 0.0
    
    for key, grad in grads.items():
        if isinstance(grad, dict):
            for subkey, subgrad in grad.items():
                full_key = f"{key}.{subkey}"
                grad_norm = jnp.linalg.norm(subgrad)
                grad_norms[full_key] = float(grad_norm)
                total_grad_norm += grad_norm**2
                logger.info(f"   {full_key}: norm={grad_norm:.6f}, shape={subgrad.shape}")
        else:
            grad_norm = jnp.linalg.norm(grad)
            grad_norms[key] = float(grad_norm)
            total_grad_norm += grad_norm**2
            logger.info(f"   {key}: norm={grad_norm:.6f}, shape={grad.shape}")
    
    total_grad_norm = jnp.sqrt(total_grad_norm)
    logger.info(f"   TOTAL GRADIENT NORM: {total_grad_norm:.6f}")
    
    # Check if gradients are zero
    if total_grad_norm < 1e-10:
        logger.error("❌ CRITICAL: Gradients are effectively zero!")
        return
    
    logger.info("\n🔍 OPTIMIZER STATE ANALYSIS")
    logger.info("-" * 50)
    
    # Get optimizer state details
    optimizer = trainer.create_optimizer()
    
    # Initialize optimizer state
    opt_state = optimizer.init(train_state_obj.params)
    logger.info(f"   Optimizer state initialized: {type(opt_state)}")
    
    # Apply gradient update
    updates, new_opt_state = optimizer.update(grads, opt_state, train_state_obj.params)
    
    logger.info("\n🔍 PARAMETER UPDATES ANALYSIS") 
    logger.info("-" * 50)
    
    update_norms = {}
    total_update_norm = 0.0
    
    for key, update in updates.items():
        if isinstance(update, dict):
            for subkey, subupdate in update.items():
                full_key = f"{key}.{subkey}"
                update_norm = jnp.linalg.norm(subupdate)
                update_norms[full_key] = float(update_norm)
                total_update_norm += update_norm**2
                logger.info(f"   {full_key}: update_norm={update_norm:.6f}")
        else:
            update_norm = jnp.linalg.norm(update)
            update_norms[key] = float(update_norm)
            total_update_norm += update_norm**2
            logger.info(f"   {key}: update_norm={update_norm:.6f}")
    
    total_update_norm = jnp.sqrt(total_update_norm)
    logger.info(f"   TOTAL UPDATE NORM: {total_update_norm:.6f}")
    
    # Check if updates are too small
    if total_update_norm < 1e-8:
        logger.error("❌ CRITICAL: Parameter updates are too small!")
        logger.error("   This suggests optimizer is scaling gradients down too much")
    else:
        logger.info("✅ Parameter updates have reasonable magnitude")
    
    # Apply updates manually and check parameter change
    logger.info("\n🔍 MANUAL PARAMETER UPDATE TEST")
    logger.info("-" * 50)
    
    # Get current parameters
    old_params = train_state_obj.params
    
    # Apply updates using optax
    new_params = optax.apply_updates(old_params, updates)
    
    # Compute actual parameter changes
    param_changes = jax.tree_map(lambda old, new: jnp.linalg.norm(new - old), old_params, new_params)
    total_param_change = sum(jax.tree_leaves(param_changes))
    
    logger.info(f"   Total parameter change: {total_param_change:.10f}")
    
    # Test with higher learning rate
    logger.info("\n🔍 HIGH LEARNING RATE TEST")
    logger.info("-" * 50)
    
    # Create optimizer with 10x higher learning rate
    high_lr_optimizer = optax.adam(learning_rate=0.1)  # 10x higher
    high_lr_opt_state = high_lr_optimizer.init(train_state_obj.params)
    high_lr_updates, _ = high_lr_optimizer.update(grads, high_lr_opt_state, train_state_obj.params)
    high_lr_new_params = optax.apply_updates(old_params, high_lr_updates)
    
    high_lr_param_changes = jax.tree_map(lambda old, new: jnp.linalg.norm(new - old), old_params, high_lr_new_params)
    high_lr_total_change = sum(jax.tree_leaves(high_lr_param_changes))
    
    logger.info(f"   High LR (0.1) parameter change: {high_lr_total_change:.10f}")
    
    if high_lr_total_change > 1e-6:
        logger.info("✅ High learning rate produces visible parameter changes")
        logger.info("   Issue: Learning rate too low or gradient clipping too aggressive")
    else:
        logger.error("❌ Even high learning rate doesn't help - deeper issue")
    
    logger.info("\n" + "=" * 80)
    logger.info("🔍 GRADIENT FLOW ANALYSIS COMPLETE")

if __name__ == "__main__":
    analyze_gradient_flow() 