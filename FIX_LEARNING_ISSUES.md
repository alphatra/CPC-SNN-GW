# 🚨 CRITICAL: Fix Learning Issues in CPC-SNN-GW

## **✅ DIAGNOSIS COMPLETE**

Po systematycznym debugging z porównaniem do working AResGW, znalazłem **ROOT CAUSE** problemów z uczeniem:

### **🎯 KEY FINDINGS:**

1. **✅ AResGW-style simple ResNet WORKS** - osiąga 84% train accuracy w 7 epochs
2. **❌ CPC simplified model FAILS** - zostaje na ~53% accuracy
3. **🎯 CONCLUSION:** Problem jest w **CPC encoder architecture**, NIE w spike bridge

---

## **🚨 ROOT CAUSE: 4 Critical Issues in CPC Encoder**

### **Issue #1: Ultra-Minimal Latent Dimension**
```yaml
# config.yaml
latent_dim: 64   # ✅ ULTRA-MINIMAL: GPU memory optimization
```
**Problem:** 64 może być za mało dla complex GW patterns  
**AResGW equivalent:** ~500+ effective features w ResNet54

### **Issue #2: Aggressive L2 Normalization** 
```python
# models/cpc_encoder.py:292-294
z_norm = jnp.linalg.norm(z, axis=-1, keepdims=True)
z_normalized = z / (z_norm + 1e-8)  # Może niszczyć gradient flow!
```
**Problem:** L2 normalization może zerować gradients  
**AResGW:** Brak agresywnej normalizacji

### **Issue #3: Complex Multi-Stage Training**
```python
# Multi-task loss with potentially unstable weighting
total_loss = ce_loss_weight * classification_loss + cpc_aux_weight * cpc_loss
# CPC loss często = 0.0, więc model nie uczy się representations
```
**AResGW:** Single-task BCE loss (stable)

### **Issue #4: Missing Proven Preprocessing**
**AResGW has:**
- Whitening module z PSD estimation
- DAIN adaptive normalization 
- Proper data augmentation

**CPC-SNN-GW:** Basic preprocessing bez proven LIGO-specific steps

---

## **🛠️ IMMEDIATE SOLUTIONS**

### **Solution #1: Fix CPC Encoder Architecture**
