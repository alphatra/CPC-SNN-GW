#!/bin/bash
# Real training using CLI with proper parameters

echo "======================================="
echo "🚀 REAL CPC-SNN-GW Training"
echo "======================================="
echo ""
echo "Using REAL data (not quick mode):"
echo "  - Full epochs: 50"
echo "  - Batch size: 16"
echo "  - No quick mode"
echo ""

# Use GPU memory efficiently
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

# Run WITHOUT quick mode to use more data
python cli.py train \
    --mode standard \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 5e-4 \
    --output-dir outputs/real_training_full \
    --balanced-early-stop \
    --opt-threshold \
    --spike-time-steps 16 \
    --snn-hidden 256 \
    --cpc-layers 3 \
    --cpc-heads 4 \
    --device gpu

echo ""
echo "✅ Real training complete!"
echo "Check outputs/real_training_full for results."
