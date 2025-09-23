# 🎊 CPC-SNN-GW System Verification Checklist

## 🔍 MANUAL VERIFICATION STEPS

### ✅ **1. YAML Configuration Propagation**
**Check in training logs:**
- [ ] `EVAL (full test) epoch=X | ... | cpc_weight=0.020 temp=0.300`
- [ ] NOT `temp=0.200` or `cpc_weight=0.000`
- [ ] Values match `configs/default.yaml` training section

### ✅ **2. GW Twins Loss Functionality**
**Check in epoch_metrics.jsonl:**
- [ ] `mean_cpc_loss` improves over epochs (becomes more negative)
- [ ] Example: `-2.90 → -3.02` or similar trend
- [ ] No plateau or stagnation in CPC loss

### ✅ **3. SNN-AE Decoder Activation**
**Check in training_results.jsonl:**
- [ ] `recon_loss > 0.0` when `gamma_reconstruction > 0`
- [ ] `gamma_reconstruction: 0.2` is logged in metrics
- [ ] NOT `recon_loss: 0.0` throughout training

### ✅ **4. Enhanced Gradient Clipping**
**Check in epoch_metrics.jsonl:**
- [ ] `mean_grad_norm_total` decreases significantly
- [ ] Example: `58.8 → 10-15` (50%+ reduction)
- [ ] No gradient explosions (>100) in later epochs

### ✅ **5. Loss Component Weights (α,β,γ)**
**Check in training_results.jsonl:**
- [ ] `alpha_classification: 1.0` is logged
- [ ] `beta_contrastive: 1.0` is logged  
- [ ] `gamma_reconstruction: 0.2` is logged
- [ ] Values match CLI arguments

### ✅ **6. Information Bottleneck Fix**
**Check accuracy improvement:**
- [ ] Accuracy improves from ~53% to ~60%+ 
- [ ] NOT stuck at random (~50%) throughout training
- [ ] CPC improvements translate to classification improvements

### ✅ **7. Model Architecture Integrity**
**Check training logs:**
- [ ] No shape mismatch errors
- [ ] No `AttributeError` for new parameters
- [ ] Model initialization successful
- [ ] Forward pass completes without errors

### ✅ **8. Memory and Performance**
**Check training performance:**
- [ ] No CUDA OOM errors
- [ ] Training speed reasonable (not dramatically slower)
- [ ] Memory usage stable throughout training
- [ ] No memory leaks or accumulation

### ✅ **9. Numerical Stability**
**Check for numerical issues:**
- [ ] No NaN/Inf in any loss values
- [ ] Gradient norms finite and reasonable
- [ ] Spike rates in reasonable range (10-30%)
- [ ] All metrics finite throughout training

### ✅ **10. CLI Integration**
**Check command execution:**
- [ ] All new CLI arguments recognized
- [ ] No `unrecognized arguments` errors
- [ ] Parameter values propagated correctly
- [ ] Help text shows new options

## 🚨 **CRITICAL FAILURE INDICATORS**

### **🔴 IMMEDIATE ATTENTION REQUIRED:**
- [ ] `recon_loss: 0.0` when `gamma_reconstruction > 0`
- [ ] Accuracy stuck at ~50% (random) 
- [ ] CPC loss not improving over epochs
- [ ] Gradient norms >100 consistently
- [ ] NaN/Inf in any metrics
- [ ] CUDA OOM or memory errors

### **🟡 INVESTIGATION NEEDED:**
- [ ] Accuracy plateau below 60%
- [ ] High variance in training accuracy (0% to 100%)
- [ ] Very slow training speed
- [ ] Inconsistent metric values

### **🟢 ACCEPTABLE ISSUES:**
- [ ] Accuracy plateau above 65%
- [ ] Gradual learning (not dramatic jumps)
- [ ] Minor memory usage increase
- [ ] Slightly slower compilation

## 🚀 **VERIFICATION COMMANDS**

### **Quick Health Check:**
```bash
python CPC-SNN-GW/verify_system_health.py
```

### **Log Analysis:**
```bash
python CPC-SNN-GW/analyze_training_logs.py
```

### **Manual Log Inspection:**
```bash
# Check epoch trends
tail -20 outputs/logs/epoch_metrics.jsonl

# Check recent step metrics  
tail -10 outputs/logs/training_results.jsonl

# Check for SNN-AE debug messages
grep -i "snn-ae\|recon\|decoder" outputs/logs/training.log

# Check YAML propagation
grep "temp=0.300\|cpc_weight=0.02" outputs/logs/training.log
```

## 🎯 **SUCCESS CRITERIA**

### **MINIMUM REQUIREMENTS (Must Pass):**
1. ✅ YAML values propagated correctly
2. ✅ No critical errors or crashes
3. ✅ Accuracy > 55% (above random)
4. ✅ CPC loss improving over time
5. ✅ Gradients stable (not exploding)

### **OPTIMAL PERFORMANCE (Target):**
1. 🎯 Accuracy > 60% and improving
2. 🎯 `recon_loss > 0` when SNN-AE enabled
3. 🎯 CPC loss steady improvement (-2.9 → -3.1+)
4. 🎯 Gradient norms <20 in later epochs
5. 🎯 All α,β,γ weights logged correctly

### **BREAKTHROUGH INDICATORS (Excellent):**
1. 🚀 Accuracy > 70%
2. 🚀 Stable learning without plateau
3. 🚀 All loss components active and balanced
4. 🚀 Memory efficient operation
5. 🚀 Fast convergence

## 📋 **POST-TRAINING VERIFICATION PROTOCOL**

After each training run:

1. **Immediate Check:**
   ```bash
   tail -5 outputs/logs/epoch_metrics.jsonl
   ```

2. **Quick Metrics:**
   ```bash
   grep "EVAL.*epoch.*acc" outputs/logs/training.log | tail -5
   ```

3. **Error Check:**
   ```bash
   grep -i "error\|failed\|nan\|inf" outputs/logs/training.log | tail -10
   ```

4. **Performance Summary:**
   ```bash
   cat outputs/standard_training/training_results.json
   ```

5. **Component Status:**
   ```bash
   grep -E "(recon_loss|alpha_|beta_|gamma_)" outputs/logs/training_results.jsonl | tail -3
   ```

---

**Use this checklist after every training run to ensure 100% system health!** 🎉
