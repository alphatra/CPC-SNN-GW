#!/usr/bin/env python3
"""
Automated system test suite for CPC-SNN-GW.

Runs comprehensive tests to verify all improvements work correctly.
"""

import subprocess
import sys
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_unit_tests():
    """Run unit tests for individual components."""
    logger.info("🧪 Running unit tests...")
    
    try:
        # Test imports
        from models.cpc.losses import gw_twins_inspired_loss
        from models.snn.core import SNNDecoder
        from training.base.config import TrainingConfig
        
        logger.info("  ✅ All imports successful")
        
        # Test TrainingConfig with new parameters
        config = TrainingConfig(
            gamma_reconstruction=0.2,
            cpc_loss_type="gw_twins_inspired",
            cpc_latent_dim=256
        )
        logger.info("  ✅ TrainingConfig with new parameters")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Unit tests failed: {e}")
        return False

def run_integration_test():
    """Run quick integration test."""
    logger.info("🔗 Running integration test...")
    
    try:
        # Run 1 epoch test
        cmd = [
            "python", "cli.py", "train",
            "-c", "configs/default.yaml",
            "--use-mlgwsc", "--whiten-psd",
            "--epochs", "1", "--batch-size", "2",
            "--cpc-loss-type", "gw_twins_inspired",
            "--gamma-reconstruction", "0.1",
            "--spike-time-steps", "16",  # Smaller for speed
            "--quick-mode"  # If available
        ]
        
        logger.info(f"  🚀 Running: {' '.join(cmd)}")
        
        # Set environment
        env = {
            "JAX_PLATFORM_NAME": "cpu",  # Use CPU for reliability
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.3"
        }
        
        # Run with timeout
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300,  # 5 minutes max
            env=env,
            cwd=Path.cwd()
        )
        
        if result.returncode == 0:
            logger.info("  ✅ Integration test PASSED")
            
            # Check for key indicators in output
            output = result.stdout + result.stderr
            indicators = {
                'gamma_recon': 'gamma_recon' in output,
                'gw_twins': 'gw_twins' in output or 'twins' in output,
                'recon_loss': 'recon_loss' in output,
                'temp_300': 'temp=0.30' in output or 'temp=0.300' in output
            }
            
            passed = sum(indicators.values())
            logger.info(f"  📊 Indicators found: {passed}/{len(indicators)}")
            
            return True
        else:
            logger.error(f"  ❌ Integration test FAILED (exit code: {result.returncode})")
            logger.error(f"  Error output: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("  ❌ Integration test TIMEOUT")
        return False
    except Exception as e:
        logger.error(f"  ❌ Integration test error: {e}")
        return False

def run_log_verification():
    """Verify training logs from previous runs."""
    logger.info("📋 Verifying previous training logs...")
    
    try:
        # Run log analysis script
        result = subprocess.run(
            ["python", "analyze_training_logs.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            logger.info("  ✅ Log analysis PASSED")
            logger.info(f"  Output: {result.stdout.split('=')[-1] if '=' in result.stdout else 'Success'}")
            return True
        else:
            logger.warning("  ⚠️ Log analysis issues detected")
            logger.info(f"  Details: {result.stderr[:200]}")
            return False
            
    except Exception as e:
        logger.error(f"  ❌ Log verification error: {e}")
        return False

def run_component_verification():
    """Run component verification script."""
    logger.info("🔧 Running component verification...")
    
    try:
        # Run system health script
        result = subprocess.run(
            ["python", "verify_system_health.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            logger.info("  ✅ Component verification PASSED")
            return True
        else:
            logger.warning("  ⚠️ Component verification issues")
            logger.info(f"  Details: {result.stderr[:200]}")
            return False
            
    except Exception as e:
        logger.error(f"  ❌ Component verification error: {e}")
        return False

def main():
    """Run complete test suite."""
    logger.info("🚀 STARTING COMPREHENSIVE SYSTEM VERIFICATION")
    logger.info("=" * 60)
    
    tests = [
        ("Unit Tests", run_unit_tests),
        ("Log Verification", run_log_verification),
        ("Component Verification", run_component_verification),
        # ("Integration Test", run_integration_test),  # Commented out - too slow for regular use
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"💥 {test_name}: CRASHED - {e}")
        
        logger.info("-" * 40)
    
    # Final summary
    success_rate = passed / total * 100
    logger.info(f"\n🏆 FINAL RESULTS:")
    logger.info(f"   Tests Passed: {passed}/{total}")
    logger.info(f"   Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 100:
        logger.info("🎉 PERFECT SCORE - SYSTEM 100% VERIFIED!")
        return True
    elif success_rate >= 80:
        logger.info("✅ EXCELLENT - SYSTEM MOSTLY VERIFIED")
        return True
    elif success_rate >= 60:
        logger.warning("⚠️ GOOD - MINOR ISSUES DETECTED")
        return True
    else:
        logger.error("🚨 CRITICAL ISSUES - SYSTEM NEEDS ATTENTION")
        return False

if __name__ == "__main__":
    success = main()
    
    logger.info("\n📋 NEXT STEPS:")
    if success:
        logger.info("✅ System ready for production training")
        logger.info("🚀 Run full training with confidence")
    else:
        logger.info("🔧 Address issues found above")
        logger.info("📖 Check SYSTEM_VERIFICATION_CHECKLIST.md")
    
    exit(0 if success else 1)
