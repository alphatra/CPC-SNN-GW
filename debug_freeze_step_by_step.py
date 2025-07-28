#!/usr/bin/env python3
"""
Debug script dla analizy freeze issue w train_step
Testuje każdy komponent step by step
"""

import time
import jax
import jax.numpy as jnp
import jax.random as random
from pathlib import Path
import sys

# Add to path  
sys.path.append(str(Path(__file__).parent))

def test_basic_jax():
    """Test podstawowych operacji JAX"""
    print("🧪 Test 1: Basic JAX operations...")
    start = time.time()
    
    key = random.PRNGKey(42)
    x = random.normal(key, (32, 64))
    y = jnp.sum(x)
    
    print(f"   ✅ JAX basic ops: {time.time() - start:.2f}s")
    return True

def test_model_creation():
    """Test tworzenia minimalnego modelu"""
    print("🧪 Test 2: Model creation...")
    start = time.time()
    
    try:
        from utils.config import load_config
        config = load_config("ultra_minimal_config.yaml")
        print(f"   ✅ Config loaded: {time.time() - start:.2f}s")
        return config
    except Exception as e:
        print(f"   ❌ Config load failed: {e}")
        return None

def test_data_creation():
    """Test tworzenia minimalnych danych"""
    print("🧪 Test 3: Minimal data creation...")
    start = time.time()
    
    try:
        # Minimalne synthetic dane
        key = random.PRNGKey(42)
        batch_size = 1
        sequence_length = 64
        
        signals = random.normal(key, (batch_size, sequence_length))
        labels = jnp.array([0])  # single label
        
        print(f"   ✅ Data created: {signals.shape}, {labels.shape} in {time.time() - start:.2f}s")
        return signals, labels
    except Exception as e:
        print(f"   ❌ Data creation failed: {e}")
        return None, None

def test_model_init():
    """Test inicjalizacji modelu"""
    print("🧪 Test 4: Model initialization...")
    start = time.time()
    
    try:
        from training.complete_enhanced_training import CompleteEnhancedModel, CompleteEnhancedConfig
        
        # Ultra-minimal config
        config = CompleteEnhancedConfig(
            cpc_latent_dim=32,
            snn_hidden_size=32, 
            sequence_length=64,
            simulation_time_steps=8,  # Poprawny parametr zamiast spike_time_steps
            transformer_num_heads=2,
            transformer_num_layers=1,
            multi_scale_kernels=(3,),
            num_threshold_scales=2,
            surrogate_gradient_beta=1.0,
            # Wyłącz wszystkie zaawansowane featury
            use_temporal_infonce=False,  # Poprawna nazwa
            use_adaptive_temperature=False,
            use_phase_preserving_encoding=False,
            use_momentum_negatives=False,
            use_pac_bayes_regularization=False,
            gradient_stability_monitoring=False  # Poprawna nazwa bez use_
        )
        
        model = CompleteEnhancedModel(config)
        print(f"   ✅ Model created: {time.time() - start:.2f}s")
        return model, config
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return None, None

def test_model_params_init(model, signals):
    """Test inicjalizacji parametrów modelu"""
    print("🧪 Test 5: Model params initialization...")
    start = time.time()
    
    try:
        key = random.PRNGKey(42)
        params = model.init(key, signals)
        print(f"   ✅ Params initialized: {time.time() - start:.2f}s")
        return params
    except Exception as e:
        print(f"   ❌ Params init failed: {e}")
        return None

def test_forward_pass(model, params, signals):
    """Test forward pass przez model"""
    print("🧪 Test 6: Forward pass...")
    start = time.time()
    
    try:
        output = model.apply(params, signals, training=False)
        # Model zwraca dict z 'logits'
        if isinstance(output, dict):
            logits = output['logits']
            print(f"   ✅ Forward pass: dict with logits {logits.shape} in {time.time() - start:.2f}s")
            return logits
        else:
            print(f"   ✅ Forward pass: {output.shape} in {time.time() - start:.2f}s") 
            return output
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return None

def test_loss_computation(model, params, signals, labels, config):
    """Test obliczania loss"""
    print("🧪 Test 7: Loss computation...")
    start = time.time()
    
    try:
        from training.complete_enhanced_training import CompleteEnhancedTrainer
        trainer = CompleteEnhancedTrainer(config)
        
        # Simple loss bez gradient accumulation
        def simple_loss_fn(params, batch):
            signals, labels = batch
            output = model.apply(params, signals, training=False)
            # Handle dict output
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output
            loss = jnp.mean(jnp.square(logits - labels.reshape(-1, 1)))
            return loss
        
        batch = (signals, labels)
        loss = simple_loss_fn(params, batch)
        print(f"   ✅ Loss computed: {loss:.6f} in {time.time() - start:.2f}s")
        return loss
    except Exception as e:
        print(f"   ❌ Loss computation failed: {e}")
        return None

def test_gradient_computation(model, params, signals, labels):
    """Test gradient computation"""
    print("🧪 Test 8: Gradient computation...")
    start = time.time()
    
    try:
        def simple_loss_fn(params, batch):
            signals, labels = batch
            output = model.apply(params, signals, training=False)
            # Handle dict output
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output
            loss = jnp.mean(jnp.square(logits - labels.reshape(-1, 1)))
            return loss
        
        batch = (signals, labels)
        loss, grads = jax.value_and_grad(simple_loss_fn)(params, batch)
        print(f"   ✅ Gradients computed: loss={loss:.6f} in {time.time() - start:.2f}s")
        return grads
    except Exception as e:
        print(f"   ❌ Gradient computation failed: {e}")
        return None

def main():
    """Main debug function"""
    print("🚀 FREEZE DEBUG - Step by Step Analysis")
    print("=" * 60)
    
    # Test 1: Basic JAX
    if not test_basic_jax():
        return
        
    # Test 2: Config
    config = test_model_creation()
    if config is None:
        return
        
    # Test 3: Data
    signals, labels = test_data_creation()
    if signals is None:
        return
        
    # Test 4: Model
    model, enhanced_config = test_model_init()
    if model is None:
        return
        
    # Test 5: Params - TUTAJ MOŻE BYĆ FREEZE
    print("⚠️  CRITICAL POINT: Model params init - może tu nastąpi freeze...")
    params = test_model_params_init(model, signals)
    if params is None:
        return
        
    # Test 6: Forward pass - KOLEJNY CRITICAL POINT
    print("⚠️  CRITICAL POINT: Forward pass - może tu nastąpi freeze...")
    output = test_forward_pass(model, params, signals)
    if output is None:
        return
        
    # Test 7: Loss - KOLEJNY CRITICAL POINT  
    print("⚠️  CRITICAL POINT: Loss computation - może tu nastąpi freeze...")
    loss = test_loss_computation(model, params, signals, labels, enhanced_config)
    if loss is None:
        return
        
    # Test 8: Gradients - NAJCZĘSTSZY FREEZE POINT
    print("⚠️  CRITICAL POINT: Gradient computation - tutaj najczęściej freeze...")
    grads = test_gradient_computation(model, params, signals, labels)
    if grads is None:
        return
        
    print("\n🎉 SUCCESS: All tests passed! System should work.")
    print("🔬 Problem może być w gradient accumulation lub bardziej złożonych operacjach.")

if __name__ == "__main__":
    main() 