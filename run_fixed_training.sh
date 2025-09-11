#!/bin/bash

# Fixed training script with corrected configuration
# ✅ CRITICAL FIX: num_classes=2 to match dataset

echo "=========================================="
echo "🚀 CPC-SNN-GW FIXED Training Launcher"
echo "=========================================="
echo ""
echo "✅ Key fixes applied:"
echo "  - num_classes changed from 3 to 2"
echo "  - Increased batch size to 32"
echo "  - Better GPU utilization settings"
echo "  - Proper class distribution handling"
echo ""

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export JAX_ENABLE_X64=True
export TF_CPP_MIN_LOG_LEVEL=2

# Create output directory
mkdir -p outputs/fixed_training

# Check if old training is still running
if pgrep -f "python.*cli.py train" > /dev/null; then
    echo "⚠️  Old training process still running. Stopping it..."
    pkill -f "python.*cli.py train"
    sleep 2
fi

echo "📊 Starting fixed training with proper configuration..."
echo ""

# Run the fixed training script
python scripts/train_fixed.py 2>&1 | tee outputs/fixed_training/training.log

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo "📁 Results saved in: outputs/fixed_training/"
else
    echo ""
    echo "❌ Training failed. Check the log above for errors."
    exit 1
fi
