# Legacy Components

This directory contains deprecated components that are no longer part of the main API but are kept for reference.

## simple_snn.py

**Status**: DEPRECATED  
**Replaced by**: `ligo_cpc_snn.models.snn_classifier` (Spyx-based implementation)  
**Reason**: Unified to single SNN implementation following ML4GW standards

The `simple_snn.py` contained a basic JAX/Flax implementation of LIF neurons and SNN classifier. This was replaced with a Spyx-based implementation (`snn_classifier.py`) that:

- Uses Google Haiku + Spyx for better stability
- Follows ML4GW neuromorphic standards
- Provides more robust training utilities
- Has better integration with the overall system

### Migration Guide

If you were using `simple_snn.py` components:

```python
# OLD (deprecated)
from ligo_cpc_snn.models.simple_snn import SimpleSNN, create_simple_snn

# NEW (recommended)
from ligo_cpc_snn.models.snn_classifier import SNNClassifier, create_snn_classifier
```

The new API is similar but uses Haiku transforms:

```python
# OLD
snn = create_simple_snn(hidden_size=64, num_classes=2)
params = snn.init(key, spikes)
logits = snn.apply(params, spikes)

# NEW
snn_fn = create_snn_classifier(hidden_size=64, num_classes=2)
params = snn_fn.init(key, spikes)
logits = snn_fn.apply(params, None, spikes)
```

Training utilities have also changed:

```python
# OLD
from ligo_cpc_snn.models.simple_snn import SimpleSNNTrainer
trainer = SimpleSNNTrainer(learning_rate=1e-3)

# NEW
from ligo_cpc_snn.models.snn_classifier import SNNTrainer
trainer = SNNTrainer(snn_fn=snn_fn, learning_rate=1e-3)
``` 