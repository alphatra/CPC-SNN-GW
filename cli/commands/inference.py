"""
Inference command implementation.

Extracted from cli.py for better modularity.
"""

import logging
from pathlib import Path

from ..parsers.base import get_base_parser

logger = logging.getLogger(__name__)


def infer_cmd():
    """Main inference command entry point."""
    parser = get_base_parser()
    parser.description = "Run inference with CPC+SNN neuromorphic gravitational wave detector"
    
    # Inference specific arguments
    parser.add_argument(
        "--model-path", "-m",
        type=Path,
        required=True,
        help="Path to trained model"
    )
    
    parser.add_argument(
        "--input-data",
        type=Path,
        required=True,
        help="Input data file or directory"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path, 
        default=Path("./inference"),
        help="Output directory for inference results"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size"
    )
    
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Enable real-time inference mode"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    from utils import setup_logging
    setup_logging(
        level=logging.INFO if args.verbose == 0 else logging.DEBUG,
        log_file=args.log_file
    )
    
    logger.info("⚡ Starting CPC+SNN inference")
    logger.info(f"   Model: {args.model_path}")
    logger.info(f"   Input: {args.input_data}")
    logger.info(f"   Output: {args.output_dir}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    from ...utils.config import load_config
    config = load_config(args.config)
    
    try:
        # Load trained model parameters
        logger.info("📂 Loading trained model parameters...")
        if not args.model_path.exists():
            logger.error(f"❌ Model path does not exist: {args.model_path}")
            return 1
            
        # Load input data
        logger.info("📊 Loading input data...")
        if not args.input_data.exists():
            logger.error(f"❌ Input data does not exist: {args.input_data}")
            return 1
            
        # Run the inference pipeline
        logger.info("⚡ Running inference pipeline...")
        logger.info(f"   - Input: {args.input_data}")
        logger.info(f"   - Batch size: {args.batch_size}")
        logger.info(f"   - Real-time mode: {args.real_time}")
        
        # Professional MLGWSC-1 inference pipeline
        from ...data.mlgwsc_data_loader import MLGWSCDataLoader
        from ...models import create_enhanced_cpc_encoder, create_enhanced_snn_classifier
        from ...models.bridge.core import create_default_spike_bridge
        from ...utils.config_loader import load_config
        
        try:
            # Load configuration
            config = load_config()
            
            # Initialize MLGWSC-1 data loader with config
            data_loader = MLGWSCDataLoader(
                mode="inference",
                config=config
            )
            logger.info("✅ MLGWSC-1 data loader initialized")
            
            # Load inference data
            if args.input_data:
                inference_data = data_loader.load_hdf5_file(str(args.input_data))
                logger.info(f"📊 Loaded inference data: {inference_data['H1'].shape}")
            else:
                # Use validation data as default
                inference_data = data_loader.load_validation_data()
                logger.info("📊 Using validation data for inference")
            
            # Load trained model
            model_components = load_trained_model(args.model_path, logger)
            if not model_components:
                return 1
                
            # Run inference pipeline
            results = run_neuromorphic_inference(
                data=inference_data,
                model_components=model_components,
                batch_size=args.batch_size,
                real_time=args.real_time,
                logger=logger
            )
            
            # Save results
            output_file = args.output_dir / f"inference_results_{int(time.time())}.hdf5"
            save_inference_results(results, output_file, logger)
            
            logger.info(f"✅ Inference completed successfully")
            logger.info(f"📄 Results saved to: {output_file}")
            return 0
            
        except Exception as e:
            logger.error(f"❌ Inference pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 2
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def load_trained_model(model_path, logger):
    """Load trained CPC+SNN model components."""
    import jax
    import jax.numpy as jnp
    import pickle
    from pathlib import Path
    
    try:
        model_path = Path(model_path)
        if not model_path.exists():
            logger.error(f"❌ Model file not found: {model_path}")
            return None
            
        # Load model checkpoint
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
            
        logger.info(f"✅ Model checkpoint loaded from {model_path}")
        return checkpoint
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None


def run_neuromorphic_inference(data, model_components, batch_size, real_time, logger):
    """Run complete neuromorphic inference pipeline."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ...models.bridge.core import create_default_spike_bridge
    
    try:
        # Extract H1 and L1 strain data
        h1_strain = data.get('H1', data.get('h1'))
        l1_strain = data.get('L1', data.get('l1'))
        
        if h1_strain is None or l1_strain is None:
            logger.error("❌ Missing H1 or L1 strain data")
            return None
            
        logger.info(f"📊 Processing {len(h1_strain)} samples")
        
        # Create model components if needed
        if 'cpc_encoder' not in model_components:
            from ...models.cpc.factory import create_enhanced_cpc_encoder
            cpc_encoder = create_enhanced_cpc_encoder()
            logger.info("✅ CPC encoder created")
        else:
            cpc_encoder = model_components['cpc_encoder']
            
        if 'snn_classifier' not in model_components:
            from ...models.snn.factory import create_enhanced_snn_classifier  
            snn_classifier = create_enhanced_snn_classifier()
            logger.info("✅ SNN classifier created")
        else:
            snn_classifier = model_components['snn_classifier']
            
        # Create spike bridge
        spike_bridge = create_default_spike_bridge()
        logger.info("✅ Spike bridge created")
        
        # Process data in batches
        results = {
            'timestamps': [],
            'predictions': [],
            'confidence_scores': [],
            'spike_patterns': []
        }
        
        num_batches = (len(h1_strain) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(h1_strain))
            
            # Extract batch
            h1_batch = h1_strain[start_idx:end_idx]
            l1_batch = l1_strain[start_idx:end_idx]
            
            # Combine strain data
            strain_batch = jnp.stack([h1_batch, l1_batch], axis=-1)
            
            # CPC encoding
            if hasattr(cpc_encoder, 'params'):
                cpc_features = cpc_encoder.apply(cpc_encoder.params, strain_batch)
            else:
                # Fallback for different model structures
                cpc_features = cpc_encoder(strain_batch)
                
            # Spike encoding
            spikes = spike_bridge.encode(cpc_features)
            
            # SNN classification
            if hasattr(snn_classifier, 'params'):
                predictions = snn_classifier.apply(snn_classifier.params, spikes)
            else:
                predictions = snn_classifier(spikes)
                
            # Store results
            results['predictions'].extend(predictions.tolist())
            results['confidence_scores'].extend(jnp.max(predictions, axis=-1).tolist())
            results['timestamps'].extend(np.arange(start_idx, end_idx).tolist())
            results['spike_patterns'].extend(spikes.tolist())
            
            if (i + 1) % 10 == 0:
                logger.info(f"📊 Processed {i + 1}/{num_batches} batches")
                
        logger.info(f"✅ Inference completed on {len(results['predictions'])} samples")
        return results
        
    except Exception as e:
        logger.error(f"❌ Neuromorphic inference failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def save_inference_results(results, output_file, logger):
    """Save inference results in MLGWSC-1 compatible format."""
    import h5py
    import numpy as np
    from pathlib import Path
    
    try:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_file, 'w') as f:
            # MLGWSC-1 format requirements
            f.create_dataset('times', data=np.array(results['timestamps']))
            f.create_dataset('ranking_statistic', data=np.array(results['confidence_scores']))
            f.create_dataset('predictions', data=np.array(results['predictions']))
            
            # Additional neuromorphic data
            f.create_dataset('spike_patterns', data=np.array(results['spike_patterns']))
            
            # Metadata
            f.attrs['format_version'] = '1.0'
            f.attrs['algorithm'] = 'CPC-SNN Neuromorphic'
            f.attrs['num_detections'] = len(results['predictions'])
            
        logger.info(f"✅ Results saved to {output_file}")
        logger.info(f"📊 Saved {len(results['predictions'])} predictions")
        
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
        raise
