#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE PIPELINE DEBUGGING SCRIPT

Tests all functionality of the CPC+SNN neuromorphic GW detection pipeline
with 3 epochs and extensive debugging to identify issues from refactoring report:

CRITICAL ISSUES TO CHECK:
- Mock metrics throughout (sophisticated placeholders)
- Broken gradient flow (stop_gradient blocks learning)
- Epoch tracking broken (epoch always = 0)
- No real evaluation (fake ROC-AUC computation)
- Unrealistic strain levels (1000x too loud)
- Perfect balance masking (50/50 hides real performance)
- SNN too shallow (insufficient capacity)
- Gradient issues (vanishing gradients)

EXPECTED BEHAVIOR FOR REAL SYSTEM:
- Gradients should change significantly between steps
- Loss should decrease over epochs (not stay constant)
- Accuracy should improve (not stay at 33% random)
- CPC loss should be > 0 (not exactly 0.000000)
- Test accuracy should differ from train accuracy
- Learning rate should decay according to schedule
"""

import logging
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug_training_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def check_jax_environment():
    """Check JAX environment and device setup"""
    logger.info("=" * 80)
    logger.info("🔍 JAX ENVIRONMENT CHECK")
    logger.info("=" * 80)
    
    # Basic JAX info
    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"Default backend: {jax.default_backend()}")
    
    # GPU memory and XLA settings
    import os
    xla_settings = {
        'XLA_PYTHON_CLIENT_PREALLOCATE': os.getenv('XLA_PYTHON_CLIENT_PREALLOCATE', 'not set'),
        'XLA_PYTHON_CLIENT_MEM_FRACTION': os.getenv('XLA_PYTHON_CLIENT_MEM_FRACTION', 'not set'),
        'XLA_FLAGS': os.getenv('XLA_FLAGS', 'not set')
    }
    logger.info(f"XLA settings: {xla_settings}")
    
    # Test basic JAX operation
    test_array = jax.random.normal(jax.random.PRNGKey(42), (10, 10))
    test_result = jnp.sum(test_array)
    logger.info(f"JAX basic test: {test_result}")
    
    return True

def debug_real_ligo_data():
    """Test real LIGO data integration vs synthetic fallbacks"""
    logger.info("=" * 80)
    logger.info("🌊 REAL LIGO DATA INTEGRATION TEST")
    logger.info("=" * 80)
    
    data_info = {}
    
    # Test real LIGO data integration
    try:
        from data.real_ligo_integration import create_real_ligo_dataset
        logger.info("✅ Real LIGO integration module found")
        
        # Test data creation
        (train_signals, train_labels), (test_signals, test_labels) = create_real_ligo_dataset(
            num_samples=100,  # Small for testing
            window_size=512,
            quick_mode=True,
            return_split=True,
            train_ratio=0.8
        )
        
        logger.info(f"✅ Real LIGO data loaded successfully")
        logger.info(f"   Train: {train_signals.shape}, labels: {train_labels.shape}")
        logger.info(f"   Test: {test_signals.shape}, labels: {test_labels.shape}")
        logger.info(f"   Train label distribution: {jnp.bincount(train_labels)}")
        logger.info(f"   Test label distribution: {jnp.bincount(test_labels)}")
        
        # Check data quality
        strain_mean = jnp.mean(train_signals)
        strain_std = jnp.std(train_signals)
        strain_max = jnp.max(jnp.abs(train_signals))
        
        logger.info(f"   Strain statistics: mean={strain_mean:.2e}, std={strain_std:.2e}, max={strain_max:.2e}")
        
        # Check for suspicious values (from refactoring report: 1e-21 to 1e-23 too loud)
        if strain_max > 1e-19:
            logger.warning(f"⚠️ SUSPICIOUS: Strain levels very high ({strain_max:.2e}) - possibly unrealistic")
        
        data_info['data_source'] = 'real_ligo'
        data_info['train_data'] = (train_signals, train_labels)
        data_info['test_data'] = (test_signals, test_labels)
        
    except Exception as e:
        logger.warning(f"❌ Real LIGO data failed: {e}")
        logger.info("🔄 Falling back to synthetic data...")
        
        # Fallback to synthetic
        try:
            from data.gw_dataset_builder import create_evaluation_dataset
            train_data = create_evaluation_dataset(
                num_samples=100,
                sequence_length=512,
                sample_rate=4096,
                random_seed=42
            )
            
            all_signals = jnp.stack([sample[0] for sample in train_data])
            all_labels = jnp.array([sample[1] for sample in train_data])
            
            # Apply stratified split
            from utils.data_split import create_stratified_split
            (train_signals, train_labels), (test_signals, test_labels) = create_stratified_split(
                all_signals, all_labels, train_ratio=0.8, random_seed=42
            )
            
            logger.info(f"✅ Synthetic data created")
            logger.info(f"   Train: {train_signals.shape}, labels: {train_labels.shape}")
            logger.info(f"   Test: {test_signals.shape}, labels: {test_labels.shape}")
            
            data_info['data_source'] = 'synthetic'
            data_info['train_data'] = (train_signals, train_labels)
            data_info['test_data'] = (test_signals, test_labels)
            
        except Exception as e2:
            logger.error(f"❌ Both real and synthetic data failed: {e2}")
            raise
    
    return data_info

def debug_model_architecture():
    """Test model architecture and component integration"""
    logger.info("=" * 80)
    logger.info("🧠 MODEL ARCHITECTURE DEBUG")
    logger.info("=" * 80)
    
    architecture_info = {}
    
    # Test CPC encoder
    try:
        from models.cpc_encoder import CPCEncoder
        logger.info("✅ CPC encoder found")
        
        # Test model creation using Flax dataclass attributes (not constructor params)
        cpc_encoder = CPCEncoder(
            latent_dim=64,
            conv_channels=(32, 64)
        )
        
        # Test forward pass
        test_input = jax.random.normal(jax.random.PRNGKey(42), (1, 512))
        key = jax.random.PRNGKey(42)
        # ✅ FIX: Use training=False to avoid PRNG issue for testing, or provide RNG
        cpc_params = cpc_encoder.init(key, test_input, train=False) 
        cpc_output = cpc_encoder.apply(cpc_params, test_input, train=False)
        
        logger.info(f"✅ CPC encoder forward pass: {test_input.shape} → {cpc_output.shape}")
        
        # Check for suspicious output (all zeros, all same value, etc.)
        output_std = jnp.std(cpc_output)
        output_mean = jnp.mean(cpc_output)
        if output_std < 1e-6:
            logger.warning(f"⚠️ SUSPICIOUS: CPC output has very low variance ({output_std:.2e})")
        
        architecture_info['cpc_working'] = True
        architecture_info['cpc_output_shape'] = cpc_output.shape
        architecture_info['cpc_output_stats'] = {'mean': float(output_mean), 'std': float(output_std)}
        
    except Exception as e:
        logger.error(f"❌ CPC encoder failed: {e}")
        architecture_info['cpc_working'] = False
    
    # Test SNN classifier
    try:
        from models.snn_classifier import SNNClassifier
        logger.info("✅ SNN classifier found")
        
        # Test model creation using Flax dataclass attributes
        snn_classifier = SNNClassifier(
            hidden_size=64,
            num_classes=2
        )
        
        # Test forward pass with spike input
        test_spikes = jax.random.bernoulli(jax.random.PRNGKey(43), 0.1, (1, 32, 64))
        snn_key = jax.random.PRNGKey(43)
        # ✅ FIX: SNNClassifier doesn't accept training parameter in __call__
        snn_params = snn_classifier.init(snn_key, test_spikes)
        snn_output = snn_classifier.apply(snn_params, test_spikes)
        
        logger.info(f"✅ SNN classifier forward pass: {test_spikes.shape} → {snn_output.shape}")
        
        # Check output distribution
        output_probs = jax.nn.softmax(snn_output)
        logger.info(f"   Output probabilities: {output_probs}")
        
        architecture_info['snn_working'] = True
        architecture_info['snn_output_shape'] = snn_output.shape
        
    except Exception as e:
        logger.error(f"❌ SNN classifier failed: {e}")
        architecture_info['snn_working'] = False
    
    # Test spike bridge
    try:
        from models.spike_bridge import ValidatedSpikeBridge
        logger.info("✅ Spike bridge found")
        
        spike_bridge = ValidatedSpikeBridge(
            time_steps=32,
            spike_encoding='temporal_contrast'
        )
        
        # Test encoding
        test_features = jax.random.normal(jax.random.PRNGKey(44), (1, 64, 64))
        bridge_key = jax.random.PRNGKey(44)
        # ✅ FIX: ValidatedSpikeBridge uses 'training' parameter, not 'train'
        bridge_params = spike_bridge.init(bridge_key, test_features, training=False)
        bridge_output = spike_bridge.apply(bridge_params, test_features, training=False)
        
        logger.info(f"✅ Spike bridge forward pass: {test_features.shape} → {bridge_output.shape}")
        
        # Check spike characteristics
        spike_rate = jnp.mean(bridge_output)
        logger.info(f"   Spike rate: {spike_rate:.4f}")
        
        architecture_info['spike_bridge_working'] = True
        architecture_info['spike_rate'] = float(spike_rate)
        
    except Exception as e:
        logger.error(f"❌ Spike bridge failed: {e}")
        architecture_info['spike_bridge_working'] = False
    
    return architecture_info

def debug_training_components():
    """Test training components for mock metrics and broken gradient flow"""
    logger.info("=" * 80)
    logger.info("🏋️ TRAINING COMPONENTS DEBUG")
    logger.info("=" * 80)
    
    training_info = {}
    
    # Test trainer creation
    try:
        from training.base_trainer import CPCSNNTrainer, TrainingConfig
        
        # ✅ FIX: Increase learning rate for visible parameter changes
        config = TrainingConfig(
            batch_size=1,
            learning_rate=1e-2,  # 100x higher for debugging
            num_epochs=3,
            num_classes=2,
            use_wandb=False,
            use_tensorboard=False,
            gradient_clipping=10.0,  # Higher clipping for higher LR
            optimizer="adam"  # Adam for faster convergence
        )
        
        trainer = CPCSNNTrainer(config)
        logger.info("✅ CPCSNNTrainer created successfully")
        
        # Test model creation
        model = trainer.create_model()
        logger.info("✅ Model created successfully")
        
        # Test training state creation
        sample_input = jax.random.normal(jax.random.PRNGKey(42), (1, 512))
        train_state_obj = trainer.create_train_state(model, sample_input)
        logger.info("✅ Training state created successfully")
        
        # Test single training step
        batch_signals = jax.random.normal(jax.random.PRNGKey(45), (1, 512))
        batch_labels = jnp.array([1])
        batch = (batch_signals, batch_labels)
        
        # Get initial parameters for gradient comparison
        initial_params = train_state_obj.params
        initial_param_norm = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), initial_params))))
        
        logger.info(f"   Initial parameter norm: {initial_param_norm:.6f}")
        
        # Execute training step
        new_state, metrics, enhanced_data = trainer.train_step(train_state_obj, batch)
        
        # Check if parameters actually changed (gradient flow test)
        new_params = new_state.params
        new_param_norm = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), new_params))))
        
        # ✅ FIX: Compare actual parameter changes using norm, not sums
        param_changes = jax.tree.map(lambda old, new: jnp.linalg.norm(new - old), initial_params, new_params)
        total_param_change = sum(jax.tree.leaves(param_changes))
        
        logger.info(f"   New parameter norm: {new_param_norm:.6f}")
        logger.info(f"   Total parameter change: {total_param_change:.6f}")
        
        # CRITICAL CHECK: Are gradients actually flowing?
        if total_param_change < 1e-10:
            logger.warning("⚠️ CRITICAL: Parameters barely changed - possible broken gradient flow!")
        else:
            logger.info("✅ Parameters changed significantly - gradient flow working")
        
        # Check metrics for suspicious values
        logger.info(f"   Training metrics: loss={metrics.loss:.6f}, accuracy={metrics.accuracy:.6f}")
        logger.info(f"   Gradient norm: {metrics.grad_norm:.6f}")
        logger.info(f"   Learning rate: {metrics.learning_rate:.6f}")
        
        # Check for mock/fake metrics (exact values that suggest placeholders)
        if metrics.accuracy in [0.5, 0.333333, 0.0, 1.0]:
            logger.warning(f"⚠️ SUSPICIOUS: Accuracy is exact value ({metrics.accuracy}) - possible mock metric")
        
        if metrics.loss == 0.0:
            logger.warning("⚠️ SUSPICIOUS: Loss is exactly 0.0 - possible mock metric")
        
        training_info['trainer_working'] = True
        training_info['gradient_flow'] = total_param_change > 1e-10
        training_info['initial_metrics'] = {
            'loss': float(metrics.loss),
            'accuracy': float(metrics.accuracy),
            'grad_norm': float(metrics.grad_norm)
        }
        
    except Exception as e:
        logger.error(f"❌ Training component test failed: {e}")
        traceback.print_exc()
        training_info['trainer_working'] = False
    
    return training_info

def debug_cpc_loss_fixes():
    """Test CPC loss fixes for proper contrastive learning"""
    logger.info("=" * 80)
    logger.info("🔧 CPC LOSS FIXES DEBUG")
    logger.info("=" * 80)
    
    cpc_info = {}
    
    try:
        from training.cpc_loss_fixes import calculate_fixed_cpc_loss, validate_cpc_features
        
        # Test CPC loss calculation
        # Create mock CPC features with temporal dimension
        batch_size, time_steps, feature_dim = 1, 32, 64
        mock_cpc_features = jax.random.normal(jax.random.PRNGKey(46), (batch_size, time_steps, feature_dim))
        
        # Test validation
        is_valid = validate_cpc_features(mock_cpc_features)
        logger.info(f"   CPC features validation: {is_valid}")
        
        # Test CPC loss calculation
        cpc_loss = calculate_fixed_cpc_loss(mock_cpc_features, temperature=0.07)
        logger.info(f"   CPC loss: {cpc_loss:.6f}")
        
        # CRITICAL CHECK: CPC loss should NOT be 0.000000 (refactoring report issue)
        if cpc_loss < 1e-10:
            logger.warning("⚠️ CRITICAL: CPC loss is essentially zero - contrastive learning broken!")
        else:
            logger.info("✅ CPC loss is non-zero - contrastive learning working")
        
        # Test with different batch sizes and sequence lengths
        test_cases = [
            (1, 10, 32),  # Short sequence
            (1, 64, 32),  # Medium sequence  
            (2, 32, 32),  # Multiple batch
        ]
        
        for i, (b, t, f) in enumerate(test_cases):
            test_features = jax.random.normal(jax.random.PRNGKey(47+i), (b, t, f))
            test_loss = calculate_fixed_cpc_loss(test_features)
            logger.info(f"   Test case {i+1} (b={b}, t={t}, f={f}): CPC loss = {test_loss:.6f}")
        
        cpc_info['cpc_fixes_working'] = True
        cpc_info['cpc_loss_value'] = float(cpc_loss)
        
    except Exception as e:
        logger.error(f"❌ CPC loss fixes test failed: {e}")
        cpc_info['cpc_fixes_working'] = False
    
    return cpc_info

def run_3_epoch_training_test(data_info: Dict[str, Any]) -> Dict[str, Any]:
    """Run actual 3-epoch training test with comprehensive monitoring"""
    logger.info("=" * 80)
    logger.info("🚀 3-EPOCH TRAINING TEST")
    logger.info("=" * 80)
    
    training_results = {}
    
    try:
        # Get data
        train_signals, train_labels = data_info['train_data']
        test_signals, test_labels = data_info['test_data']
        
        # Use smaller subset for quick testing
        max_samples = 20
        train_signals = train_signals[:max_samples]
        train_labels = train_labels[:max_samples]
        test_signals = test_signals[:min(10, len(test_signals))]
        test_labels = test_labels[:min(10, len(test_labels))]
        
        logger.info(f"   Using {len(train_signals)} training samples, {len(test_signals)} test samples")
        
        # Create trainer
        from training.base_trainer import CPCSNNTrainer, TrainingConfig
        
        config = TrainingConfig(
            batch_size=1,
            learning_rate=1e-4,
            num_epochs=3,
            num_classes=2,
            use_wandb=False,
            use_tensorboard=False,
            gradient_clipping=1.0
        )
        
        trainer = CPCSNNTrainer(config)
        model = trainer.create_model()
        train_state_obj = trainer.create_train_state(model, train_signals[:1])
        
        # Track metrics across epochs
        epoch_metrics = []
        parameter_norms = []
        gradient_norms = []
        
        # Training loop
        for epoch in range(3):
            logger.info(f"   🔥 Epoch {epoch+1}/3")
            
            epoch_losses = []
            epoch_accuracies = []
            epoch_grad_norms = []
            
            # Track parameter norm at start of epoch
            param_norm = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), train_state_obj.params))))
            parameter_norms.append(float(param_norm))
            logger.info(f"      Parameter norm: {param_norm:.6f}")
            
            # Process all training samples
            for i in range(len(train_signals)):
                batch_signals = train_signals[i:i+1]
                batch_labels = train_labels[i:i+1]
                batch = (batch_signals, batch_labels)
                
                # Training step
                train_state_obj, metrics, enhanced_data = trainer.train_step(train_state_obj, batch)
                
                epoch_losses.append(float(metrics.loss))
                epoch_accuracies.append(float(metrics.accuracy))
                epoch_grad_norms.append(float(metrics.grad_norm))
            
            # Compute epoch averages
            avg_loss = float(jnp.mean(jnp.array(epoch_losses)))
            avg_accuracy = float(jnp.mean(jnp.array(epoch_accuracies)))
            avg_grad_norm = float(jnp.mean(jnp.array(epoch_grad_norms)))
            
            gradient_norms.append(avg_grad_norm)
            
            logger.info(f"      Loss: {avg_loss:.6f}, Accuracy: {avg_accuracy:.6f}, Grad norm: {avg_grad_norm:.6f}")
            
            epoch_metrics.append({
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': avg_accuracy,
                'grad_norm': avg_grad_norm
            })
        
        # Test evaluation
        logger.info("   📊 Test evaluation...")
        try:
            from training.test_evaluation import evaluate_on_test_set
            
            test_results = evaluate_on_test_set(
                train_state_obj,
                test_signals,
                test_labels,
                train_signals=train_signals,
                verbose=True
            )
            
            logger.info(f"      Test accuracy: {test_results['test_accuracy']:.6f}")
            logger.info(f"      Model collapse: {test_results['model_collapse']}")
            logger.info(f"      Suspicious patterns: {test_results['suspicious_patterns']}")
            
        except Exception as e:
            logger.warning(f"⚠️ Test evaluation failed: {e}")
            test_results = {'test_accuracy': 0.0, 'model_collapse': True}
        
        # CRITICAL ANALYSIS
        logger.info("   🔍 CRITICAL ANALYSIS:")
        
        # Check if loss is decreasing
        if len(epoch_metrics) >= 2:
            loss_trend = epoch_metrics[-1]['loss'] - epoch_metrics[0]['loss']
            if loss_trend > 0.01:
                logger.warning(f"⚠️ SUSPICIOUS: Loss INCREASED by {loss_trend:.6f} - possible learning issue")
            elif abs(loss_trend) < 1e-6:
                logger.warning(f"⚠️ SUSPICIOUS: Loss barely changed ({loss_trend:.6f}) - possible mock training")
            else:
                logger.info(f"✅ Loss decreased by {-loss_trend:.6f} - learning working")
        
        # Check if accuracy is improving
        final_accuracy = epoch_metrics[-1]['accuracy']
        if final_accuracy == 0.5:
            logger.warning("⚠️ SUSPICIOUS: Accuracy exactly 0.5 - possible random guessing")
        elif final_accuracy in [0.333333, 0.33333]:
            logger.warning("⚠️ SUSPICIOUS: Accuracy exactly 1/3 - possible mock metric")
        
        # Check gradient norms
        if all(g < 1e-8 for g in gradient_norms):
            logger.warning("⚠️ CRITICAL: All gradient norms very small - vanishing gradients")
        elif all(g > 100 for g in gradient_norms):
            logger.warning("⚠️ CRITICAL: All gradient norms very large - exploding gradients")
        
        # Check parameter evolution
        param_norm_change = parameter_norms[-1] - parameter_norms[0] if len(parameter_norms) > 1 else 0
        if abs(param_norm_change) < 1e-6:
            logger.warning("⚠️ CRITICAL: Parameter norms barely changed - possible broken optimization")
        
        training_results = {
            'success': True,
            'epoch_metrics': epoch_metrics,
            'parameter_norms': parameter_norms,
            'gradient_norms': gradient_norms,
            'test_results': test_results,
            'data_source': data_info['data_source']
        }
        
    except Exception as e:
        logger.error(f"❌ 3-epoch training test failed: {e}")
        traceback.print_exc()
        training_results = {'success': False, 'error': str(e)}
    
    return training_results

def main():
    """Run comprehensive pipeline debugging"""
    logger.info("🔍 STARTING COMPREHENSIVE PIPELINE DEBUGGING")
    logger.info("=" * 80)
    
    # Collect all debug results
    debug_results = {}
    
    # 1. JAX environment check
    debug_results['jax_env'] = check_jax_environment()
    
    # 2. Data integration test
    debug_results['data_info'] = debug_real_ligo_data()
    
    # 3. Model architecture test
    debug_results['architecture_info'] = debug_model_architecture()
    
    # 4. Training components test
    debug_results['training_info'] = debug_training_components()
    
    # 5. CPC loss fixes test
    debug_results['cpc_info'] = debug_cpc_loss_fixes()
    
    # 6. 3-epoch training test
    if debug_results['data_info']:
        debug_results['training_results'] = run_3_epoch_training_test(debug_results['data_info'])
    
    # FINAL SUMMARY
    logger.info("=" * 80)
    logger.info("📋 DEBUGGING SUMMARY")
    logger.info("=" * 80)
    
    # Check for critical issues
    critical_issues = []
    warnings = []
    
    # Data issues
    if debug_results['data_info']['data_source'] == 'synthetic':
        warnings.append("Using synthetic data (real LIGO integration failed)")
    
    # Architecture issues
    arch_info = debug_results.get('architecture_info', {})
    if not arch_info.get('cpc_working', False):
        critical_issues.append("CPC encoder not working")
    if not arch_info.get('snn_working', False):
        critical_issues.append("SNN classifier not working")
    if not arch_info.get('spike_bridge_working', False):
        critical_issues.append("Spike bridge not working")
    
    # Training issues
    train_info = debug_results.get('training_info', {})
    if not train_info.get('gradient_flow', False):
        critical_issues.append("Gradient flow broken")
    if not train_info.get('trainer_working', False):
        critical_issues.append("Trainer not working")
    
    # CPC issues
    cpc_info = debug_results.get('cpc_info', {})
    if not cpc_info.get('cpc_fixes_working', False):
        critical_issues.append("CPC loss fixes not working")
    elif cpc_info.get('cpc_loss_value', 0) < 1e-10:
        critical_issues.append("CPC loss is zero (contrastive learning broken)")
    
    # Training results issues
    train_results = debug_results.get('training_results', {})
    if not train_results.get('success', False):
        critical_issues.append("3-epoch training failed")
    
    # Report results
    if critical_issues:
        logger.error("❌ CRITICAL ISSUES FOUND:")
        for issue in critical_issues:
            logger.error(f"   - {issue}")
    else:
        logger.info("✅ No critical issues found")
    
    if warnings:
        logger.warning("⚠️ WARNINGS:")
        for warning in warnings:
            logger.warning(f"   - {warning}")
    
    # Success assessment
    total_tests = 6
    passed_tests = sum([
        debug_results.get('jax_env', False),
        bool(debug_results.get('data_info')),
        arch_info.get('cpc_working', False) and arch_info.get('snn_working', False),
        train_info.get('trainer_working', False) and train_info.get('gradient_flow', False),
        cpc_info.get('cpc_fixes_working', False),
        train_results.get('success', False)
    ])
    
    success_rate = passed_tests / total_tests
    logger.info(f"📊 OVERALL SUCCESS RATE: {success_rate:.1%} ({passed_tests}/{total_tests} tests passed)")
    
    if success_rate >= 0.8:
        logger.info("🎉 Pipeline appears to be working well!")
    elif success_rate >= 0.6:
        logger.warning("⚠️ Pipeline has some issues but basic functionality works")
    else:
        logger.error("❌ Pipeline has serious issues that need to be addressed")
    
    return debug_results

if __name__ == "__main__":
    main() 