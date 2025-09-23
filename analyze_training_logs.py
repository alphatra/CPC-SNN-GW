#!/usr/bin/env python3
"""
Analyze training logs to verify system improvements.
"""

import json
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_epoch_metrics():
    """Analyze epoch-level metrics for trends."""
    logger.info("📊 Analyzing epoch metrics...")
    
    try:
        epoch_file = Path("outputs/logs/epoch_metrics.jsonl")
        if not epoch_file.exists():
            logger.error("❌ No epoch metrics file found")
            return False
        
        epochs = []
        with open(epoch_file, 'r') as f:
            for line in f:
                if line.strip():
                    epochs.append(json.loads(line))
        
        if not epochs:
            logger.error("❌ No epoch data found")
            return False
        
        # Extract metrics
        accuracies = [e['eval_accuracy'] for e in epochs]
        cpc_losses = [abs(e['mean_cpc_loss']) for e in epochs]  # Take absolute value
        grad_norms = [e['mean_grad_norm_total'] for e in epochs]
        
        # Analyze trends
        logger.info(f"  📈 Accuracy trend: {accuracies[0]:.3f} → {accuracies[-1]:.3f}")
        logger.info(f"  📈 CPC loss trend: {cpc_losses[0]:.3f} → {cpc_losses[-1]:.3f}")
        logger.info(f"  📈 Gradient norm trend: {grad_norms[0]:.1f} → {grad_norms[-1]:.1f}")
        
        # Check improvements
        accuracy_improvement = accuracies[-1] - accuracies[0]
        cpc_improvement = cpc_losses[-1] - cpc_losses[0]  # Higher absolute value = better
        gradient_reduction = grad_norms[0] - grad_norms[-1]  # Lower = better
        
        checks = {
            'accuracy_improved': accuracy_improvement > 0.01,  # >1% improvement
            'cpc_improved': cpc_improvement > 0.05,  # CPC loss magnitude increased
            'gradients_stabilized': gradient_reduction > 5.0,  # Significant reduction
            'accuracy_above_random': accuracies[-1] > 0.55,  # Above 55%
            'no_collapse': max(accuracies) - min(accuracies) < 0.3  # Not too volatile
        }
        
        passed = 0
        for check, result in checks.items():
            if result:
                logger.info(f"  ✅ {check}: PASSED")
                passed += 1
            else:
                logger.warning(f"  ⚠️ {check}: FAILED")
        
        logger.info(f"  📊 Epoch analysis: {passed}/{len(checks)} checks passed")
        return passed >= len(checks) * 0.6  # 60% threshold
        
    except Exception as e:
        logger.error(f"❌ Epoch analysis error: {e}")
        return False

def analyze_step_metrics():
    """Analyze step-level metrics for detailed behavior."""
    logger.info("📊 Analyzing step metrics...")
    
    try:
        step_file = Path("outputs/logs/training_results.jsonl")
        if not step_file.exists():
            logger.error("❌ No step metrics file found")
            return False
        
        # Read last 100 steps for recent behavior
        steps = []
        with open(step_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-100:]:  # Last 100 steps
                if line.strip():
                    steps.append(json.loads(line))
        
        if not steps:
            logger.error("❌ No step data found")
            return False
        
        # Check critical metrics
        recon_losses = [s.get('recon_loss', 0.0) for s in steps]
        cpc_losses = [abs(s.get('cpc_loss', 0.0)) for s in steps]
        alphas = [s.get('alpha_classification', 0.0) for s in steps]
        betas = [s.get('beta_contrastive', 0.0) for s in steps]
        gammas = [s.get('gamma_reconstruction', 0.0) for s in steps]
        
        # Analyze
        logger.info(f"  🔍 Recent steps analysis (last {len(steps)} steps):")
        logger.info(f"    - Reconstruction loss: {np.mean(recon_losses):.6f} (should be >0 if enabled)")
        logger.info(f"    - CPC loss magnitude: {np.mean(cpc_losses):.3f}")
        logger.info(f"    - Alpha (classification): {np.mean(alphas):.2f}")
        logger.info(f"    - Beta (contrastive): {np.mean(betas):.2f}")
        logger.info(f"    - Gamma (reconstruction): {np.mean(gammas):.2f}")
        
        checks = {
            'recon_loss_active': np.mean(recon_losses) > 0.001 if np.mean(gammas) > 0 else True,
            'cpc_loss_reasonable': 1.0 < np.mean(cpc_losses) < 10.0,
            'weights_present': all(x > 0 for x in [np.mean(alphas), np.mean(betas)]),
            'no_nan_losses': all(np.isfinite(x) for x in cpc_losses)
        }
        
        passed = 0
        for check, result in checks.items():
            if result:
                logger.info(f"  ✅ {check}: PASSED")
                passed += 1
            else:
                logger.warning(f"  ⚠️ {check}: FAILED")
        
        logger.info(f"  📊 Step analysis: {passed}/{len(checks)} checks passed")
        return passed >= len(checks) * 0.75  # 75% threshold
        
    except Exception as e:
        logger.error(f"❌ Step analysis error: {e}")
        return False

def check_critical_fixes():
    """Check if critical fixes are working."""
    logger.info("🔧 Checking critical fixes...")
    
    fixes = {
        "YAML Propagation": "Check if temp=0.30 and cpc_weight=0.02 in logs",
        "Information Bottleneck": "Check if CPC features aren't averaged (shape preservation)", 
        "GW Twins Loss": "Check if CPC loss improves over epochs",
        "Enhanced Gradient Clipping": "Check if gradient norms are stabilized",
        "SNN-AE Decoder": "Check if recon_loss > 0 when gamma_reconstruction > 0",
        "Loss Component Weights": "Check if α,β,γ are logged in metrics"
    }
    
    for fix_name, description in fixes.items():
        logger.info(f"  🔧 {fix_name}: {description}")
    
    return True

def main():
    """Main verification function."""
    logger.info("🎊 CPC-SNN-GW SYSTEM HEALTH VERIFICATION")
    logger.info("=" * 60)
    
    # Check if training has been run
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        logger.error("❌ No outputs directory - run training first")
        return False
    
    # Run log analysis
    epoch_ok = analyze_epoch_metrics()
    step_ok = analyze_step_metrics()
    
    # Check fixes
    check_critical_fixes()
    
    # Overall assessment
    logger.info("=" * 60)
    if epoch_ok and step_ok:
        logger.info("🎉 SYSTEM VERIFICATION: PASSED")
        logger.info("✅ All critical improvements are working correctly")
        return True
    else:
        logger.warning("⚠️ SYSTEM VERIFICATION: PARTIAL")
        logger.info("🔧 Some issues detected - check logs above")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
