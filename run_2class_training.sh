#!/bin/bash

# Simple script to run training with correct 2-class configuration
# Uses CLI which already handles all configuration properly

echo "=========================================="
echo "🚀 CPC-SNN-GW Training with 2-class Fix"
echo "=========================================="
echo ""
echo "✅ Key configuration:"
echo "  - num_classes = 2 (binary: noise vs GW)"
echo "  - batch_size = 32"
echo "  - learning_rate = 3e-4"
echo "  - 3 SNN layers with LayerNorm"
echo ""

# Kill any existing training
pkill -f "python.*cli.py train" 2>/dev/null
sleep 2

# Run training with correct parameters
# CRITICAL: Using standard mode which should use 2 classes
python cli.py train \
    --mode standard \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 3e-4 \
    --output-dir outputs/2class_training \
    --device gpu \
    --balanced-early-stop \
    --opt-threshold \
    --spike-time-steps 16 \
    --snn-hidden 256 \
    --cpc-layers 3 \
    --cpc-heads 4 \
    --num-classes 2 \
    2>&1 | tee outputs/2class_training_log.txt
