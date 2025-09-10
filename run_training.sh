#!/bin/bash
# Simple training launcher with all fixes

echo "=================================="
echo "🚀 CPC-SNN-GW Training Launcher"
echo "=================================="
echo ""
echo "JAX Backend: $(python -c 'import jax; print(jax.default_backend())')"
echo "GPU Available: $(python -c 'import jax; print(jax.devices())')"
echo ""

# Set memory optimization
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

echo "Starting training with all fixes applied..."
echo ""

# Run training
python cli.py train \
    --mode standard \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 5e-4 \
    --output-dir outputs/fixed_training \
    --quick-mode \
    --balanced-early-stop \
    --opt-threshold \
    --spike-time-steps 16 \
    --snn-hidden 256 \
    --cpc-layers 3 \
    --cpc-heads 4 \
    --device gpu

echo ""
echo "Training complete! Check outputs/fixed_training for results."
