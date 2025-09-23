"""
Standard training runner implementation.

Extracted from cli.py for better modularity.
"""

import logging
import time
from typing import Dict, Any
import numpy as np
import jax.numpy as jnp
import jax

# ✅ UNIFIED FILTERING: Import professional unified filtering system
from data.filtering.unified import antialias_downsample

logger = logging.getLogger(__name__)


def run_standard_training(config: Dict, args) -> Dict[str, Any]:
    """
    Run real CPC+SNN training using CPCSNNTrainer.
    
    Args:
        config: Training configuration dictionary
        args: Command line arguments
        
    Returns:
        Training results dictionary
    """
    logger.info("🚀 Starting STANDARD CPC+SNN Training")
    
    try:
        # Runtime performance optimizations (memory fraction, XLA flags)
        try:
            from utils.config import apply_performance_optimizations
            apply_performance_optimizations()
        except Exception:
            pass
        
        # Import training modules
        from training.base.trainer import CPCSNNTrainer
        from training.base.config import TrainingConfig
        # Use unified data router to respect --use-mlgwsc and config data_dir
        from cli.commands.training.data_loader import load_training_data
        
        # Create training config
        training_config = TrainingConfig(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            # ✅ Ensure we pull SNN num_classes from nested model.snn
            num_classes=(
                config.get('model', {}).get('snn', {}).get('num_classes',
                    config.get('model', {}).get('num_classes', 2)
                )
            ),
            spike_time_steps=args.spike_time_steps,
            spike_threshold=args.spike_threshold,
            spike_learnable=args.spike_learnable,
            spike_threshold_levels=args.spike_threshold_levels,
            spike_surrogate_type=args.spike_surrogate_type,
            spike_surrogate_beta=args.spike_surrogate_beta,
            # ✅ CPC from YAML training section
            cpc_temperature=float(config.get('training', {}).get('cpc_temperature', 0.20)),
            cpc_aux_weight=float(config.get('training', {}).get('cpc_aux_weight', 0.05)),
            # ✅ Eval batch size from YAML to stabilize per-epoch metrics
            eval_batch_size=int(config.get('training', {}).get('eval_batch_size', 64)),
            
            # ✅ NEW: Advanced loss configuration from CLI
            cpc_loss_type=getattr(args, 'cpc_loss_type', 'temporal_info_nce'),
            gw_twins_redundancy_weight=getattr(args, 'gw_twins_redundancy_weight', 0.1),
            
            # ✅ NEW: Loss component weights (α,β,γ) from CLI
            alpha_classification=getattr(args, 'alpha_classification', 1.0),
            beta_contrastive=getattr(args, 'beta_contrastive', 1.0),
            gamma_reconstruction=getattr(args, 'gamma_reconstruction', 0.0),
            
            # ✅ NEW: Advanced gradient clipping from CLI
            adaptive_grad_clip_threshold=getattr(args, 'adaptive_grad_clip_threshold', 0.5),
            per_module_grad_clip=(getattr(args, 'per_module_grad_clip', 'true') == 'true'),
            cpc_grad_clip_multiplier=getattr(args, 'cpc_grad_clip_multiplier', 0.8),
            snn_grad_clip_multiplier=getattr(args, 'snn_grad_clip_multiplier', 1.0),
            bridge_grad_clip_multiplier=getattr(args, 'bridge_grad_clip_multiplier', 1.2),
            
            # ✅ CPC model parameters from YAML
            cpc_latent_dim=int(config.get('models', {}).get('cpc', {}).get('latent_dim', 256))
        )
        
        logger.info(f"   📊 Config: {args.epochs} epochs, batch={args.batch_size}")
        logger.info(f"   🧠 Model: {training_config.num_classes} classes")
        logger.info(f"   ⚡ Spikes: {args.spike_time_steps} steps, threshold={args.spike_threshold}")
        
        # Create trainer
        trainer = CPCSNNTrainer(training_config)
        
        # Load dataset via router (MLGWSC / synthetic / real)
        logger.info("📊 Loading training dataset...")
        train_signals, train_labels, test_signals, test_labels = load_training_data(args)
        # Ensure CPC input shape [B, T, F]
        if train_signals.ndim == 2:
            train_signals = train_signals[..., None]
        if test_signals.ndim == 2:
            test_signals = test_signals[..., None]
        # Reduce channels → single feature for CPC if multi-channel (on CPU to avoid device OOM)
        if train_signals.shape[-1] > 1:
            train_signals = np.mean(train_signals, axis=-1, keepdims=True)
        if test_signals.shape[-1] > 1:
            test_signals = np.mean(test_signals, axis=-1, keepdims=True)
        # Downsample long sequences to stabilize attention memory (configurable target T with anti-aliasing)
        def _design_lowpass_kernel(decim_factor: int, taps: int) -> jnp.ndarray:
            # Normalized cutoff for anti-aliasing (Nyquist/decim_factor)
            fc = 0.5 / float(max(1, decim_factor))
            n = jnp.arange(taps)
            m = (taps - 1) / 2.0
            # Sinc low-pass (normalized sinc: jnp.sinc uses sin(pi x)/(pi x))
            h = 2.0 * fc * jnp.sinc(2.0 * fc * (n - m))
            # Hann window
            w = 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * n / (taps - 1))
            h = h * w
            # Normalize to unity gain at DC
            h = h / (jnp.sum(h) + 1e-8)
            return h.astype(train_signals.dtype)

        def _antialias_downsample(x: jnp.ndarray, target_t: int = 512, max_taps: int = 97) -> jnp.ndarray:
            # x: [B, T, F]
            t = x.shape[1]
            if t <= target_t:
                return x
            factor = int(jnp.ceil(t / target_t))
            taps = int(min(max_taps, max(31, 6 * factor + 1)))
            kernel = _design_lowpass_kernel(factor, taps)
            # Convolve along time axis per feature
            # Expand kernel for broadcasting over batch and features
            kernel_ = kernel[None, :, None]
            pad = (taps // 2)
            # Pad reflect to reduce edge artifacts
            x_pad = jnp.pad(x, ((0, 0), (pad, pad), (0, 0)), mode='reflect')
            # Depthwise 1D conv via explicit convolution per feature
            def conv_one_feature(feat_idx):
                xf = x_pad[:, :, feat_idx]
                yf = jax.vmap(lambda row: jnp.convolve(row, kernel, mode='valid'))(xf)
                return yf
            feats = x.shape[-1]
            y_list = [conv_one_feature(f) for f in range(feats)]
            y = jnp.stack(y_list, axis=-1)
            # Decimate
            y = y[:, ::factor, :]
            # Trim or pad to exact target_t
            t_new = y.shape[1]
            if t_new > target_t:
                y = y[:, :target_t, :]
            elif t_new < target_t:
                pad_back = target_t - t_new
                y = jnp.pad(y, ((0, 0), (0, pad_back), (0, 0)), mode='edge')
            return y

        # ✅ UNIFIED: Use professional unified filtering system
        # Get target length from config (fallback to 512)
        target_t = int(config.get('data', {}).get('downsample_target_t', 512))
        # Downsample in chunks to avoid device OOM for large N
        def _downsample_in_chunks(arr_np: np.ndarray, target_len: int, chunk_size: int = 128) -> np.ndarray:
            outputs = []
            n = arr_np.shape[0]
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk = jnp.asarray(arr_np[start:end])
                y = antialias_downsample(chunk, target_length=target_len)
                outputs.append(np.asarray(y))
            return np.concatenate(outputs, axis=0)

        target_t = int(config.get('data', {}).get('downsample_target_t', 512))
        train_signals = _downsample_in_chunks(train_signals, target_t)
        test_signals = _downsample_in_chunks(test_signals, target_t)
        logger.info(f"   ⏬ Downsampled T: train={train_signals.shape[1]}, test={test_signals.shape[1]}, F={train_signals.shape[-1]}")
        logger.info(f"   📊 Train: {len(train_signals)} samples")
        logger.info(f"   📊 Test: {len(test_signals)} samples")
        
        # Run training
        start_time = time.time()
        training_results = trainer.train(
            train_signals=train_signals,
            train_labels=train_labels,
            test_signals=test_signals,
            test_labels=test_labels
        )
        training_time = time.time() - start_time
        
        logger.info(f"✅ Standard training completed in {training_time:.1f}s")
        logger.info(f"   📊 Final accuracy: {training_results.get('test_accuracy', 0.0):.3f}")
        
        # Save results
        results_dir = args.output_dir / "standard_training"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        results_file = results_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        logger.info(f"📊 Results saved to: {results_file}")
        
        return training_results
        
    except Exception as e:
        logger.error(f"❌ Standard training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
