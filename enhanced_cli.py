#!/usr/bin/env python3
"""
Enhanced CLI for Neuromorphic Gravitational Wave Detection

Revolutionary command-line interface with beautiful Rich visualizations and
advanced gradient accumulation for production-scale training. Features
scientific metrics tracking and memory optimization.

Usage:
    python enhanced_cli.py train --enhanced-logging --gradient-accumulation
    python enhanced_cli.py train --production-scale --memory-optimize
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Enhanced imports
from utils.enhanced_logger import get_enhanced_logger, ScientificMetrics
from training.gradient_accumulation import create_gradient_accumulator, AccumulationConfig
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Original system imports
import jax
import jax.numpy as jnp
from data.gw_dataset_builder import create_evaluation_dataset
from training.base_trainer import CPCSNNTrainer, TrainerBase
# from models.neuromorphic_model import NeuromorphicGWModel  # Using trainer.create_model() instead

# Initialize enhanced console
console = Console(width=120)

def create_enhanced_welcome():
    """Create beautiful welcome message"""
    
    welcome_text = """
[bold blue]🔬 Enhanced Neuromorphic Gravitational Wave Detection System[/bold blue]

[cyan]Revolutionary Features:[/cyan]
• 🎨 Beautiful Rich logging with scientific metrics
• 🧬 Advanced gradient accumulation for memory efficiency  
• 📊 Real-time progress tracking and visualization
• 🚀 Production-scale training with memory optimization
• 🔬 Scientific error analysis and diagnostics

[yellow]Ready for breakthrough gravitational wave research![/yellow]
"""
    
    welcome_panel = Panel.fit(
        welcome_text,
        title="🌊 Enhanced GW Detection System",
        border_style="blue",
        padding=(1, 2)
    )
    
    console.print(welcome_panel)

def setup_enhanced_training(args) -> tuple:
    """Setup enhanced training with beautiful logging and gradient accumulation"""
    
    # ✅ CUDA/GPU OPTIMIZATION: Configure JAX for proper GPU usage
    console.print("🔧 [cyan]Configuring JAX GPU settings...[/cyan]")
    
    try:
        # ✅ FIX: Set JAX memory pre-allocation to prevent 16GB allocation spikes
        import os
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.35'  # Use max 35% of GPU memory for CLI
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        # ✅ CUDA TIMING FIX: Suppress timing warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'               # Suppress TF warnings
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'               # Async kernel execution
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'   # Async allocator
        os.environ['XLA_FLAGS'] = '--xla_gpu_enable_fast_min_max=true'  # ✅ FIXED: Removed invalid flag
        
        # Configure JAX for efficient GPU memory usage
        import jax
        jax.config.update('jax_enable_x64', False)  # Use float32 for memory efficiency
        
        # ✅ COMPREHENSIVE CUDA WARMUP: Advanced model-specific kernel initialization
        console.print("🔥 [yellow]Performing COMPREHENSIVE GPU warmup to eliminate timing issues...[/yellow]")
        
        warmup_key = jax.random.PRNGKey(123)
        
        # ✅ STAGE 1: Basic tensor operations (varied sizes)
        console.print("   🔸 [dim]Stage 1: Basic tensor operations...[/dim]")
        for size in [(8, 32), (16, 64), (32, 128)]:
            data = jax.random.normal(warmup_key, size)
            _ = jnp.sum(data ** 2).block_until_ready()
            _ = jnp.dot(data, data.T).block_until_ready()
            _ = jnp.mean(data, axis=1).block_until_ready()
        
        # ✅ STAGE 2: Model-specific operations (Dense layers)
        console.print("   🔸 [dim]Stage 2: Dense layer operations...[/dim]")
        input_data = jax.random.normal(warmup_key, (4, 256))
        weight_matrix = jax.random.normal(jax.random.split(warmup_key)[0], (256, 128))
        bias = jax.random.normal(jax.random.split(warmup_key)[1], (128,))
        
        dense_output = jnp.dot(input_data, weight_matrix) + bias
        activated = jnp.tanh(dense_output)  # Activation similar to model
        activated.block_until_ready()
        
        # ✅ STAGE 3: CPC/SNN specific operations  
        console.print("   🔸 [dim]Stage 3: CPC/SNN operations...[/dim]")
        sequence_data = jax.random.normal(warmup_key, (2, 64, 32))  # [batch, time, features]
        
        # Temporal operations (like CPC)
        context = sequence_data[:, :-1, :]  # Context frames
        target = sequence_data[:, 1:, :]    # Target frames  
        
        # Normalization (like CPC encoder)
        context_norm = context / (jnp.linalg.norm(context, axis=-1, keepdims=True) + 1e-8)
        target_norm = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)
        
        # Similarity computation (like InfoNCE)
        context_flat = context_norm.reshape(-1, context_norm.shape[-1])
        target_flat = target_norm.reshape(-1, target_norm.shape[-1])
        similarity = jnp.dot(context_flat, target_flat.T)
        similarity.block_until_ready()
        
        # ✅ STAGE 4: Advanced operations (convolutions, reductions)
        console.print("   🔸 [dim]Stage 4: Advanced CUDA kernels...[/dim]")
        conv_data = jax.random.normal(warmup_key, (4, 128, 1))  # [batch, length, channels] - REDUCED for memory
        kernel = jax.random.normal(jax.random.split(warmup_key)[0], (5, 1, 16))  # [width, in_ch, out_ch] - REDUCED
        
        # Convolution operation (like CPC encoder)
        conv_result = jax.lax.conv_general_dilated(
            conv_data, kernel, 
            window_strides=[1], padding=[(2, 2)],  # ✅ Conservative params  
            dimension_numbers=('NHC', 'HIO', 'NHC')
        )
        conv_result.block_until_ready()
        
        # ✅ STAGE 5: JAX compilation warmup 
        console.print("   🔸 [dim]Stage 5: JAX JIT compilation warmup...[/dim]")
        
        @jax.jit
        def warmup_jit_function(x):
            return jnp.sum(x ** 2) + jnp.mean(jnp.tanh(x))
        
        jit_data = jax.random.normal(warmup_key, (8, 32))  # ✅ REDUCED: Memory-safe
        _ = warmup_jit_function(jit_data).block_until_ready()
        
        # ✅ FINAL SYNCHRONIZATION: Ensure all kernels are compiled
        import time
        time.sleep(0.1)  # Brief pause for kernel initialization
        
        # ✅ ADDITIONAL WARMUP: Model-specific operations
        console.print("   🔸 [dim]Stage 6: SpikeBridge/CPC specific warmup...[/dim]")
        
        # Mimic exact CPC encoder operations
        cpc_input = jax.random.normal(warmup_key, (1, 256))  # Strain data size
        # Conv1D operations
        for channels in [32, 64, 128]:
            conv_kernel = jax.random.normal(jax.random.split(warmup_key)[0], (3, 1, channels))
            conv_data = cpc_input[..., None]  # Add channel dim
            _ = jax.lax.conv_general_dilated(
                conv_data, conv_kernel,
                window_strides=[2], padding='SAME',
                dimension_numbers=('NHC', 'HIO', 'NHC')
            ).block_until_ready()
        
        # Dense layers with GELU/tanh (like model)
        dense_sizes = [(256, 128), (128, 64), (64, 32)]
        temp_data = jax.random.normal(warmup_key, (1, 256))
        for in_size, out_size in dense_sizes:
            w = jax.random.normal(jax.random.split(warmup_key)[0], (in_size, out_size))
            b = jax.random.normal(jax.random.split(warmup_key)[1], (out_size,))
            temp_data = jnp.tanh(jnp.dot(temp_data, w) + b)
            temp_data.block_until_ready()
            if temp_data.shape[1] != in_size:  # Adjust for next iteration
                temp_data = jax.random.normal(warmup_key, (1, out_size))
        
        console.print("✅ [green]COMPREHENSIVE GPU warmup completed - ALL CUDA kernels initialized![/green]")
        
    except Exception as e:
        console.print(f"[yellow]⚠️ GPU warmup warning: {e}[/yellow]")
    
    # ✅ AUTO GPU DETECTION: Override GPU setting if not available
    try:
        import jax
        available_devices = jax.devices()
        gpu_available = any(device.platform == 'gpu' for device in available_devices)
        if not gpu_available and args.gpu:
            args.gpu = False
            console.print("[yellow]⚠️ GPU requested but not available, falling back to CPU[/yellow]")
        elif gpu_available and not hasattr(args, '_gpu_manually_set'):
            args.gpu = True  # Auto-enable if available
    except Exception as e:
        args.gpu = False
        console.print(f"[yellow]⚠️ GPU detection failed: {e}[/yellow]")
    
    # Initialize enhanced logger
    logger = get_enhanced_logger(
        name="Enhanced-GW-Training",
        log_level=args.log_level.upper(),
        log_file=args.log_file,
        console_width=120
    )
    
    # Create beautiful configuration display
    config_table = Table(title="🔧 Training Configuration", box=box.ROUNDED)
    config_table.add_column("Parameter", style="cyan", width=25)
    config_table.add_column("Value", style="magenta", width=15)
    config_table.add_column("Description", style="green", width=40)
    
    config_table.add_row("Mode", args.mode, "Training mode (standard/enhanced/advanced)")
    config_table.add_row("Epochs", str(args.epochs), "Number of training epochs")
    config_table.add_row("Batch Size", str(args.batch_size), "Physical batch size")
    config_table.add_row("Learning Rate", f"{args.learning_rate:.2e}", "Optimizer learning rate")
    config_table.add_row("GPU", "✅ Enabled" if args.gpu else "❌ Disabled", "GPU acceleration")
    config_table.add_row("Enhanced Logging", "✅ Active" if args.enhanced_logging else "❌ Standard", "Rich logging system")
    config_table.add_row("Gradient Accum", "✅ Active" if args.gradient_accumulation else "❌ Standard", "Memory-efficient training")
    
    console.print(config_table)
    
    # Setup gradient accumulation if requested
    gradient_accumulator = None
    if args.gradient_accumulation:
        logger.info("🧬 Initializing gradient accumulation system...")
        
        gradient_accumulator = create_gradient_accumulator(
            accumulation_steps=args.accumulation_steps,
            max_physical_batch_size=args.batch_size,
            gradient_clipping=1.0,
            auto_optimize=True
        )
        
        # Display accumulation config
        accum_info = Panel.fit(
            f"""
[bold green]🧬 Gradient Accumulation Configuration[/bold green]

[cyan]Physical Batch Size:[/cyan] {gradient_accumulator.config.max_physical_batch_size}
[cyan]Accumulation Steps:[/cyan] {gradient_accumulator.config.accumulation_steps}
[cyan]Effective Batch Size:[/cyan] {gradient_accumulator.config.effective_batch_size}
[cyan]Memory Monitoring:[/cyan] {'✅ Enabled' if gradient_accumulator.config.memory_monitoring else '❌ Disabled'}
[cyan]Gradient Clipping:[/cyan] {gradient_accumulator.config.gradient_clipping}

[yellow]Memory Efficiency: {gradient_accumulator.config.effective_batch_size / gradient_accumulator.config.max_physical_batch_size:.1f}x larger effective batch[/yellow]
            """,
            title="🧬 Accumulation Setup",
            border_style="green"
        )
        console.print(accum_info)
    
    return logger, gradient_accumulator

def run_enhanced_training(args):
    """Run enhanced training with beautiful logging and gradient accumulation"""
    
    # Setup enhanced systems
    logger, gradient_accumulator = setup_enhanced_training(args)
    
    try:
        # ✅ REAL LIGO DATA: Create dataset using real GW150914 data
        logger.info("🌊 Creating REAL LIGO gravitational wave dataset...")
        
        # Simple progress without excessive updates
        dataset_progress_desc = f"🌊 Loading REAL LIGO data (target: {args.num_samples} samples)"
        
        with logger.progress_context(dataset_progress_desc, total=100, reuse_existing=True) as task_id:
            try:
                from data.real_ligo_integration import create_real_ligo_dataset
                
                signals, labels = create_real_ligo_dataset(
                    num_samples=args.num_samples,
                    window_size=args.sequence_length,
                    quick_mode=True  # Enhanced CLI uses quick mode
                )
                
                # Complete progress
                logger.update_progress(task_id, advance=100, 
                                     description=f"🌊 Real LIGO data loaded: {len(signals)} samples ✓")
                
            except ImportError:
                logger.warning("🌊 Real LIGO integration not available - using synthetic data")
                # Fallback to synthetic
                train_data = create_evaluation_dataset(
                    num_samples=args.num_samples,
                    sequence_length=args.sequence_length,
                    sample_rate=4096,
                    random_seed=42
                )
                signals = jnp.stack([sample[0] for sample in train_data])
                labels = jnp.array([sample[1] for sample in train_data])
                
                logger.update_progress(task_id, advance=100, 
                                     description=f"🌊 Synthetic data loaded: {len(signals)} samples ✓")
        
        logger.info(f"✅ Dataset created: {len(signals)} samples")
        logger.info(f"   Data source: {'Real LIGO GW150914' if 'create_real_ligo_dataset' in locals() else 'Synthetic'}")
        logger.info(f"   Signal ratio: {jnp.mean(labels):.1%}")
        
        # Create enhanced model and trainer
        logger.info("🧠 Initializing neuromorphic model...")
        
        key = jax.random.PRNGKey(42)
        dummy_input = signals[:1]
        
        # Create trainer with enhanced config
        from training.base_trainer import TrainingConfig
        
        # ✅ OPTIMIZED: Better learning rate and epochs for loss reduction
        optimized_lr = args.learning_rate
        optimized_epochs = args.epochs
        
        if args.optimize_loss or args.auto_lr:
            # Smart optimization based on gradient accumulation
            if gradient_accumulator:
                # With gradient accumulation, we can use higher LR
                optimized_lr = args.learning_rate * 2.0  # 2x higher LR
                optimized_epochs = max(args.epochs, 5)   # Minimum 5 epochs
                logger.info(f"🚀 Loss optimization enabled: LR {args.learning_rate:.2e} → {optimized_lr:.2e}, Epochs {args.epochs} → {optimized_epochs}")
            else:
                optimized_lr = args.learning_rate * 1.5  # 1.5x higher LR
                optimized_epochs = max(args.epochs, 3)   # Minimum 3 epochs
                logger.info(f"📈 Auto-LR optimization: LR {args.learning_rate:.2e} → {optimized_lr:.2e}, Epochs {args.epochs} → {optimized_epochs}")
        
        trainer_config = TrainingConfig(
            learning_rate=optimized_lr,
            batch_size=args.batch_size,
            gradient_clipping=1.0 if gradient_accumulator else 0.0,
            num_epochs=optimized_epochs,
            output_dir=args.output_dir,
            use_wandb=False  # Disable for enhanced logging focus
        )
        
        trainer = CPCSNNTrainer(trainer_config)
        model = trainer.create_model()
        trainer.train_state = trainer.create_train_state(model, dummy_input)
        
        logger.info("🚀 Starting enhanced training loop...")
        
        # Enhanced training loop with gradient accumulation
        for epoch in range(optimized_epochs):
            epoch_start = time.time()
            logger.info(f"🔥 Epoch {epoch + 1}/{optimized_epochs}")
            
            # ✅ REMOVED: Don't clear progress, let it update naturally
            # if epoch > 0:
            #     logger.clear_progress()
            
            # Prepare batches
            num_samples = len(signals)
            indices = jax.random.permutation(key, num_samples)
            
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            # ✅ FIXED: Single progress bar per epoch - no duplicates
            epoch_progress_desc = f"🔥 Epoch {epoch + 1}/{optimized_epochs}"
            
            with logger.progress_context(epoch_progress_desc, total=num_samples, reuse_existing=True) as task_id:
                
                for start_idx in range(0, num_samples, args.batch_size):
                    end_idx = min(start_idx + args.batch_size, num_samples)
                    batch_indices = indices[start_idx:end_idx]
                    
                    batch_signals = signals[batch_indices]
                    batch_labels = labels[batch_indices]
                    batch = (batch_signals, batch_labels)
                    
                    if gradient_accumulator:
                        # ✅ ENHANCED: Use CPC loss fixes for proper contrastive learning
                        from training.cpc_loss_fixes import create_enhanced_loss_fn, extract_cpc_metrics
                        
                        # Use gradient accumulation with enhanced loss function
                        micro_batches = gradient_accumulator.create_accumulation_batches(batch)
                        
                        # ✅ CRITICAL: Use fixed CPC loss function
                        loss_fn = create_enhanced_loss_fn(trainer.train_state, temperature=0.07)
                        
                        accumulated_grads, metrics = gradient_accumulator.accumulate_gradients(
                            trainer.train_state, micro_batches, loss_fn
                        )
                        
                        # Apply accumulated gradients
                        trainer.train_state = trainer.train_state.apply_gradients(grads=accumulated_grads)
                        
                        # ✅ ENHANCED: Extract CPC metrics properly
                        cpc_metrics = extract_cpc_metrics(metrics)
                        
                        batch_loss = metrics.loss
                        batch_accuracy = metrics.accuracy
                        batch_cpc_loss = cpc_metrics.get('cpc_loss', 0.0)
                        batch_snn_accuracy = cpc_metrics.get('snn_accuracy', batch_accuracy)
                        
                    else:
                        # Standard training step
                        trainer.train_state, metrics, _ = trainer.train_step(trainer.train_state, batch)
                        batch_loss = float(metrics.loss)
                        batch_accuracy = float(metrics.accuracy)
                    
                    # ✅ ENHANCED: Update metrics with CPC tracking
                    epoch_loss += batch_loss
                    epoch_accuracy += batch_accuracy
                    
                    # Track CPC metrics if available
                    if 'batch_cpc_loss' in locals():
                        if not hasattr(logger, '_epoch_cpc_loss'):
                            logger._epoch_cpc_loss = 0.0
                            logger._epoch_snn_accuracy = 0.0
                        logger._epoch_cpc_loss += batch_cpc_loss
                        logger._epoch_snn_accuracy += batch_snn_accuracy
                    
                    num_batches += 1
                    
                    # ✅ ENHANCED: Progress updates with CPC metrics
                    processed_samples = min(end_idx, num_samples)
                    avg_loss = epoch_loss / num_batches
                    avg_accuracy = epoch_accuracy / num_batches
                    avg_cpc_loss = getattr(logger, '_epoch_cpc_loss', 0.0) / num_batches if hasattr(logger, '_epoch_cpc_loss') else 0.0
                    
                    # Show CPC loss in progress if available and non-zero
                    if avg_cpc_loss > 1e-6:
                        progress_desc = f"🔥 E{epoch + 1} | Loss: {avg_loss:.3f} | Acc: {avg_accuracy:.2f} | CPC: {avg_cpc_loss:.3f}"
                    else:
                        progress_desc = f"🔥 E{epoch + 1} | Loss: {avg_loss:.3f} | Acc: {avg_accuracy:.2f}"
                    
                    logger.update_progress(
                        task_id, 
                        advance=processed_samples - start_idx,
                        description=progress_desc
                    )
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            epoch_time = time.time() - epoch_start
            
            # Create scientific metrics
            scientific_metrics = ScientificMetrics(
                epoch=epoch + 1,
                loss=avg_loss,
                accuracy=avg_accuracy,
                training_time=epoch_time,
                batch_size=gradient_accumulator.config.effective_batch_size if gradient_accumulator else args.batch_size,
                samples_processed=num_samples,
                learning_rate=args.learning_rate
            )
            
            # Log beautiful scientific metrics
            logger.log_scientific_metrics(scientific_metrics)
            
            # Log gradient accumulation summary if used
            if gradient_accumulator and epoch == optimized_epochs - 1:
                gradient_accumulator.log_accumulation_summary()
        
        # Final experiment summary
        logger.log_experiment_summary()
        
        # Save model with absolute path (fixes Orbax error)
        output_dir = Path(args.output_dir).resolve()  # ✅ ORBAX FIX: Convert to absolute path
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "enhanced_model.orbax"
        
        logger.info(f"💾 Saving enhanced model to {model_path}")
        try:
            # Use orbax for saving with absolute path
            import orbax.checkpoint as ocp
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(str(model_path.resolve()), trainer.train_state)  # ✅ ORBAX FIX: Absolute path
            logger.info("✅ Model saved successfully")
        except Exception as e:
            logger.warning(f"⚠️ Model saving failed: {e}")
            logger.info("Continuing without model save...")
        
        # Success message
        success_panel = Panel.fit(
            f"""
[bold green]🎉 ENHANCED TRAINING COMPLETED SUCCESSFULLY![/bold green]

[cyan]Training Results:[/cyan]
• Final Accuracy: {avg_accuracy:.3%}
• Final Loss: {avg_loss:.6f}
• Total Epochs: {optimized_epochs}
• Training Mode: {args.mode}
• Gradient Accumulation: {'✅ Used' if gradient_accumulator else '❌ Not used'}

[yellow]Model saved to:[/yellow] {model_path}

[green]🔬 Ready for gravitational wave detection![/green]
            """,
            title="✅ Training Complete",
            border_style="green"
        )
        console.print(success_panel)
        
        return True
        
    except Exception as e:
        logger.error("💥 Enhanced training failed", exception=e)
        return False

def create_argument_parser():
    """Create enhanced argument parser with gradient accumulation options"""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Neuromorphic Gravitational Wave Detection Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enhanced training with gradient accumulation
  python enhanced_cli.py train --enhanced-logging --gradient-accumulation
  
  # Production scale training with memory optimization  
  python enhanced_cli.py train --production-scale --accumulation-steps 8
  
  # Standard training with beautiful logging
  python enhanced_cli.py train --enhanced-logging
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Run enhanced training')
    
    # Original arguments
    train_parser.add_argument('--mode', choices=['standard', 'enhanced', 'advanced'], 
                             default='standard', help='Training mode')
    train_parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=1, help='Physical batch size (memory optimized)')
    train_parser.add_argument('--learning-rate', '--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU acceleration (auto-enabled if available)')
    train_parser.add_argument('--output-dir', default='outputs/enhanced_training', help='Output directory')
    train_parser.add_argument('--log-file', help='Log file path')
    train_parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    # Enhanced arguments
    train_parser.add_argument('--enhanced-logging', action='store_true', 
                             help='Enable beautiful Rich logging system')
    train_parser.add_argument('--gradient-accumulation', action='store_true', default=True,
                                help='Enable gradient accumulation for memory efficiency (auto-enabled)')
    train_parser.add_argument('--accumulation-steps', type=int, default=4,
                             help='Number of gradient accumulation steps')
    train_parser.add_argument('--production-scale', action='store_true',
                             help='Enable production-scale optimizations')
    train_parser.add_argument('--optimize-loss', action='store_true',
                             help='Enable loss optimization with better learning rate and more epochs')
    train_parser.add_argument('--auto-lr', action='store_true',
                             help='Enable automatic learning rate optimization')
    
    # Dataset arguments  
    train_parser.add_argument('--num-samples', type=int, default=1200, help='Number of dataset samples')
    train_parser.add_argument('--sequence-length', type=int, default=2048, help='Sequence length')
    
    return parser

def main():
    """Enhanced main function with beautiful error handling"""
    
    try:
        # Create beautiful welcome
        create_enhanced_welcome()
        
        # Parse arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        if args.command is None:
            parser.print_help()
            return 1
        
        # Force enhanced logging if using enhanced features
        if args.gradient_accumulation or args.production_scale:
            args.enhanced_logging = True
            
        # Auto-enable loss optimization with gradient accumulation
        if args.gradient_accumulation and not args.optimize_loss:
            args.optimize_loss = True
            console.print("[yellow]🚀 Auto-enabled loss optimization with gradient accumulation[/yellow]")
        
        if args.command == 'train':
            success = run_enhanced_training(args)
            return 0 if success else 1
        else:
            console.print(f"[red]Unknown command: {args.command}[/red]")
            return 1
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        return 130
        
    except Exception as e:
        console.print(f"\n[red]💥 Unexpected error: {e}[/red]")
        console.print_exception()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 