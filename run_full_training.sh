#!/bin/bash
# FULL training with MLGWSC-1 dataset (100k+ samples)

echo "==========================================="
echo "🚀 CPC-SNN-GW FULL Training (MLGWSC-1)"
echo "==========================================="
echo ""
echo "⚠️  This will use the FULL MLGWSC-1 dataset"
echo "   Expected: 100,000+ training samples"
echo "   Training time: ~2-4 hours on GPU"
echo ""
echo "JAX Backend: $(python -c 'import jax; print(jax.default_backend())')"
echo "GPU Available: $(python -c 'import jax; print(jax.devices())')"
echo ""

# Set memory optimization for large dataset
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

echo "Loading MLGWSC-1 dataset..."
echo ""

# Run FULL training without quick mode
python cli.py train \
    --mode standard \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 5e-4 \
    --output-dir outputs/full_mlgwsc_training \
    --balanced-early-stop \
    --opt-threshold \
    --spike-time-steps 16 \
    --snn-hidden 256 \
    --cpc-layers 3 \
    --cpc-heads 4 \
    --device gpu \
    --use-mlgwsc \
    --mlgwsc-samples 100000 \
    --mlgwsc-background-hdf /teamspace/studios/this_studio/data/dataset-4/v2/train_background_s24w61w_1.hdf

echo ""
echo "Full training complete! Check outputs/full_mlgwsc_training for REAL results."
