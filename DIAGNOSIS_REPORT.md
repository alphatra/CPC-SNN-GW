# 🚨 CPC-SNN-GW Learning Failure: Complete Diagnosis Report

## **Current Status: Model nie uczy się (accuracy ~50% = random)**

```
Epoch 1: Loss=0.6130, Accuracy=0.4885, Test=0.499
Epoch 2: Loss=0.6237, Accuracy=0.4969, Test=0.534  
Epoch 3: Loss=0.6304, Accuracy=0.5062, Test=0.476
```

**Wniosek:** Model oscyluje wokół random performance bez learning progress.

---

## **🔍 Root Cause Analysis: AResGW vs CPC-SNN-GW**

### **✅ AResGW (Working Model)**

**Architecture:**
```python
Raw Strain [B, T] 
  → Whitening (PSD-based)
  → DAIN Normalization (adaptive)
  → ResNet54 (27 blocks, proven depth)
  → Global pooling 
  → Softmax [B, 2]
```

**Training:**
```python
loss = reg_BCELoss(dim=2)  # Simple, stable
optimizer = Adam(lr=5e-5)  # Proven LR
training_loss = loss(output, labels)  # Direct supervised
```

**Key Success Factors:**
1. ✅ **Proven architecture** (ResNet54 with 27 blocks)
2. ✅ **Proper preprocessing** (whitening + DAIN)
3. ✅ **Simple loss function** (regularized BCE)
4. ✅ **Stable gradients** (no spike discontinuities)
5. ✅ **Optimized hyperparameters** (LR=5e-5, proper scheduling)

---

### **❌ CPC-SNN-GW (Failing Model)**

**Architecture:**
```python
Raw Strain [B, T] 
  → CPC Encoder (latent_dim=64, minimal?)
  → Spike Bridge (temporal-contrast, surrogate grads) 
  → SNN Classifier (LIF neurons, complex dynamics)
  → Multi-task loss (InfoNCE + CE)
```

**Training:**
```python
classification_loss = cross_entropy(logits, labels)
cpc_loss = temporal_info_nce_loss(cpc_features)  # Often = 0.0!
total_loss = ce_weight * classification_loss + cpc_weight * cpc_loss
```

---

## **🚨 6 Critical Problems Identified**

### **Problem #1: Over-Engineering vs Proven Simplicity**

**AResGW:** Jeden sprawdzony komponent (ResNet54)  
**CPC-SNN-GW:** 3 eksperimentalne komponenty (CPC + SpikeBridge + SNN)

**Risk:** Każdy komponent wprowadza dodatkowe źródło błędów.

### **Problem #2: Loss Function Instability**

**AResGW:**
```python
loss = reg_BCELoss(output, labels)  # Stable, always > 0
```

**CPC-SNN-GW:**
```python
cpc_loss = temporal_info_nce_loss(...)  # Often = 0.0 ❌
total_loss = 0.8 * ce_loss + 0.2 * cpc_loss  # Unstable weighting
```

**Problem:** CPC loss frequently = 0.0, więc model nie uczy się representations.

### **Problem #3: Gradient Flow Disruption**

**AResGW:** Continuous backprop through standard layers  
**CPC-SNN-GW:** Gradients muszą przejść przez spike discontinuities z surrogate functions

**Risk:** Surrogate gradients mogą być zbyt słabe lub nieprawidłowe.

### **Problem #4: Missing Proven Preprocessing**

**AResGW has critical preprocessing:**
```python
class Whiten(nn.Module):  # PSD-based whitening
class DAIN_Layer(nn.Module):  # Adaptive normalization
```

**CPC-SNN-GW:** Brak equivalent preprocessing pipeline.

### **Problem #5: Architecture Parameter Mismatch**

**AResGW:** Optimized dla 1.25s windows, proper channel progression  
**CPC-SNN-GW:** Ultra-minimal latent_dim=64, może za mały dla GW detection

### **Problem #6: Training Hyperparameters**

**AResGW:** lr=5e-5 (proven), proper warmup, curriculum SNR  
**CPC-SNN-GW:** lr=1e-3 (może za wysoki), complex multi-task balancing

---

## **📋 Debugging Protocol**

### **Step 1: Architecture Isolation** 

Run debugging scripts:

```bash
# Test simplified CPC without spike bridge
python debug_simple_model.py

# Test AResGW-style architecture  
python debug_aresgw_style.py
```

**Expected outcomes:**
- If simplified CPC works → Problem is in spike bridge/SNN
- If AResGW-style works → Problem is in CPC architecture  
- If both fail → Problem is in data/preprocessing

### **Step 2: Component-by-Component Validation**

1. **Data Pipeline:** Czy preprocessing jest equivalent do AResGW?
2. **CPC Encoder:** Czy generuje sensible features?
3. **Spike Bridge:** Czy preserves information?
4. **SNN Classifier:** Czy gradient flow działa?
5. **Loss Functions:** Czy InfoNCE vs BCE comparison?

---

## **🛠️ Immediate Solutions**

### **Solution #1: Emergency Fallback - Copy AResGW Exactly**

Zamiast fix CPC-SNN, skopiuj working AResGW architecture:

```python
# Convert AResGW PyTorch → JAX/Flax
class WorkingResNet54JAX(nn.Module):
    """Direct port of working AResGW to JAX."""
    
    @nn.compact 
    def __call__(self, x):
        # Exact ResNet54Double architecture
        # Proven to work on GW detection
```

### **Solution #2: Gradual Integration**

1. **Phase 1:** Get AResGW-style working in JAX ✅
2. **Phase 2:** Add CPC encoder gradually  
3. **Phase 3:** Add spike bridge as optional component
4. **Phase 4:** Full neuromorphic when stable

### **Solution #3: Fix Current CPC-SNN**

Jeśli insist on neuromorphic approach:

1. **Fix CPC loss:** Ensure InfoNCE ≠ 0.0
2. **Fix spike bridge:** Better surrogate gradients
3. **Fix learning rate:** Reduce to 1e-4 or 5e-5
4. **Add whitening:** Copy AResGW preprocessing
5. **Simplify multi-task:** Focus on classification first

---

## **⚡ Quick Test Commands**

```bash
# 1. Debug simplified architecture
cd /teamspace/studios/this_studio/CPC-SNN-GW
python debug_simple_model.py

# 2. Debug AResGW-style 
python debug_aresgw_style.py

# 3. Check if data pipeline works
python -c "
from data.real_ligo_integration import create_real_ligo_dataset
data = create_real_ligo_dataset(num_samples=50, window_size=256, quick_mode=True)
print('Data shape:', data[0][0].shape)
print('Label distribution:', jnp.bincount(data[0][1]))
print('Data range:', jnp.min(data[0][0]), 'to', jnp.max(data[0][0]))
"
```

---

## **🎯 Recommended Action Plan**

### **Immediate (24h):**
1. ✅ Run debugging scripts to isolate problem
2. 🔧 If architecture issue: Implement AResGW-JAX hybrid  
3. 📊 If data issue: Fix preprocessing pipeline

### **Short-term (1 week):**  
1. 🏗️ Get stable baseline working (60%+ accuracy)
2. 🧪 Add neuromorphic components gradually
3. 📈 Validate each component before adding next

### **Long-term (1 month):**
1. 🚀 Full CPC-SNN-GW working with proven components
2. 📊 Performance comparison vs AResGW baseline
3. 🔬 Scientific analysis of neuromorphic advantages

---

## **💡 Key Insight**

**AResGW succeeds because it's SIMPLE and PROVEN.**  
**CPC-SNN-GW fails because it's COMPLEX and EXPERIMENTAL.**

**Solution:** Start simple, add complexity gradually, validate each step.
