"""
MLGWSC-1 Data Loader for CPC-SNN Neuromorphic GW Detection

Professional data loader for MLGWSC-1 mock data challenge datasets.
Handles HDF5 files with H1/L1 strain data and provides standardized interface.
"""

import h5py
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union, Any
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class MLGWSCDataLoader:
    """
    Professional data loader for MLGWSC-1 challenge datasets.
    
    Handles the standard MLGWSC-1 HDF5 format with H1/L1 strain data
    and provides preprocessing for neuromorphic CPC+SNN pipeline.
    """
    
    def __init__(self, 
                 data_dir: Optional[str] = None,
                 mode: str = "training",
                 sample_rate: Optional[int] = None,
                 segment_length: Optional[float] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize MLGWSC-1 data loader.
        
        Args:
            data_dir: Directory containing MLGWSC-1 HDF5 files (uses config if None)
            mode: Loading mode ("training", "validation", "inference")
            sample_rate: Data sample rate in Hz (uses config if None)
            segment_length: Segment length in seconds (uses config if None)
            config: Configuration dictionary (loads default if None)
        """
        # Load configuration if not provided
        if config is None:
            from utils.config_loader import load_config
            config = load_config()
            
        # Use config values as defaults
        if data_dir is None:
            data_dir = config['system']['data_dir']
        if sample_rate is None:
            sample_rate = config['data']['sample_rate']
        if segment_length is None:
            segment_length = config['data']['segment_length']
            
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(segment_length * sample_rate)
        self.config = config
        
        # Validate data directory
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
            
        # Find available files
        self.available_files = self._discover_files()
        logger.info(f"✅ MLGWSC data loader initialized")
        logger.info(f"   - Data directory: {data_dir}")
        logger.info(f"   - Mode: {mode}")
        logger.info(f"   - Available files: {len(self.available_files)}")
        
    def _discover_files(self) -> Dict[str, List[Path]]:
        """Discover available MLGWSC-1 files."""
        files = {
            'train_background': [],
            'train_foreground': [], 
            'train_injections': [],
            'val_background': [],
            'val_foreground': [],
            'val_injections': []
        }
        
        for pattern in files.keys():
            matching_files = list(self.data_dir.glob(f"{pattern}*.hdf"))
            files[pattern] = matching_files
            
        return files
    
    def load_hdf5_file(self, filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Load HDF5 file with H1/L1 strain data.
        
        Args:
            filepath: Path to HDF5 file
            
        Returns:
            Dictionary with 'H1' and 'L1' strain data arrays
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {filepath}")
            
        try:
            with h5py.File(filepath, 'r') as f:
                data = {}
                
                # Handle HDF5 groups (MLGWSC-1 format)
                if 'H1' in f and 'L1' in f:
                    h1_data = f['H1']
                    l1_data = f['L1']
                    
                    # If groups, get the actual data arrays
                    if hasattr(h1_data, 'keys'):
                        h1_keys = list(h1_data.keys())
                        l1_keys = list(l1_data.keys())
                        if h1_keys and l1_keys:
                            data['H1'] = np.array(h1_data[h1_keys[0]])
                            data['L1'] = np.array(l1_data[l1_keys[0]])
                        else:
                            raise ValueError(f"Empty H1/L1 groups in {filepath}")
                    else:
                        # Direct arrays
                        data['H1'] = np.array(h1_data)
                        data['L1'] = np.array(l1_data)
                        
                elif 'h1' in f and 'l1' in f:
                    data['H1'] = np.array(f['h1'])
                    data['L1'] = np.array(f['l1'])
                else:
                    # Try to find strain data with different naming
                    keys = list(f.keys())
                    if len(keys) >= 2:
                        data['H1'] = np.array(f[keys[0]])
                        data['L1'] = np.array(f[keys[1]])
                    else:
                        raise ValueError(f"Cannot find H1/L1 strain data in {filepath}")
                
                logger.info(f"✅ Loaded HDF5 file: {filepath.name}")
                logger.info(f"   - H1 shape: {data['H1'].shape}")
                logger.info(f"   - L1 shape: {data['L1'].shape}")
                
                return data
                
        except Exception as e:
            logger.error(f"❌ Failed to load HDF5 file {filepath}: {e}")
            raise
    
    def load_training_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Load training data (background + injections).
        
        Returns:
            Tuple of (background_data, injection_data)
        """
        background_data = None
        injection_data = None
        
        # Load background data
        if self.available_files['train_background']:
            bg_file = self.available_files['train_background'][0]
            background_data = self.load_hdf5_file(bg_file)
            logger.info("✅ Training background data loaded")
            
        # Load injection data  
        if self.available_files['train_injections']:
            inj_file = self.available_files['train_injections'][0]
            injection_data = self.load_hdf5_file(inj_file)
            logger.info("✅ Training injection data loaded")
            
        return background_data, injection_data
    
    def load_validation_data(self) -> Dict[str, np.ndarray]:
        """
        Load validation data for inference/evaluation.
        
        Returns:
            Dictionary with validation strain data
        """
        if self.available_files['val_background']:
            val_file = self.available_files['val_background'][0]
            val_data = self.load_hdf5_file(val_file)
            logger.info("✅ Validation data loaded")
            return val_data
        else:
            raise FileNotFoundError("No validation files found")
    
    def create_segments(self, 
                       data: Dict[str, np.ndarray], 
                       overlap: float = 0.5) -> List[Dict[str, np.ndarray]]:
        """
        Create overlapping segments from continuous strain data.
        
        Args:
            data: Dictionary with H1/L1 strain data
            overlap: Overlap fraction between segments (0.0 - 1.0)
            
        Returns:
            List of segment dictionaries
        """
        h1_strain = data['H1']
        l1_strain = data['L1']
        
        # Calculate step size
        step_size = int(self.segment_samples * (1.0 - overlap))
        
        segments = []
        start_idx = 0
        
        while start_idx + self.segment_samples <= len(h1_strain):
            end_idx = start_idx + self.segment_samples
            
            segment = {
                'H1': h1_strain[start_idx:end_idx],
                'L1': l1_strain[start_idx:end_idx],
                'start_time': start_idx / self.sample_rate,
                'end_time': end_idx / self.sample_rate,
                'segment_id': len(segments)
            }
            
            segments.append(segment)
            start_idx += step_size
            
        logger.info(f"✅ Created {len(segments)} segments")
        logger.info(f"   - Segment length: {self.segment_length}s")
        logger.info(f"   - Overlap: {overlap*100:.1f}%")
        
        return segments
    
    def preprocess_for_neuromorphic(self, 
                                   data: Dict[str, np.ndarray]) -> jnp.ndarray:
        """
        Preprocess strain data for neuromorphic CPC+SNN pipeline.
        
        Args:
            data: Dictionary with H1/L1 strain data
            
        Returns:
            JAX array ready for CPC encoding [batch, time, detectors]
        """
        h1_strain = data['H1']
        l1_strain = data['L1']
        
        # Ensure same length
        min_length = min(len(h1_strain), len(l1_strain))
        h1_strain = h1_strain[:min_length]
        l1_strain = l1_strain[:min_length]
        
        # Stack detectors
        strain_data = np.stack([h1_strain, l1_strain], axis=-1)
        
        # Convert to JAX array
        strain_jax = jnp.array(strain_data)
        
        # Add batch dimension if needed
        if strain_jax.ndim == 2:
            strain_jax = strain_jax[None, ...]
            
        logger.info(f"✅ Preprocessed for neuromorphic pipeline")
        logger.info(f"   - Output shape: {strain_jax.shape}")
        
        return strain_jax
    
    def create_labeled_dataset(self) -> Tuple[List[jnp.ndarray], List[int]]:
        """
        Create labeled dataset for training.
        
        Returns:
            Tuple of (data_segments, labels) where labels: 0=noise, 1=signal
        """
        data_segments = []
        labels = []
        
        # Load and process background data (label = 0)
        background_data, injection_data = self.load_training_data()
        
        if background_data:
            bg_segments = self.create_segments(background_data)
            for segment in bg_segments:
                processed = self.preprocess_for_neuromorphic(segment)
                data_segments.append(processed[0])  # Remove batch dim
                labels.append(0)  # Background/noise
                
        # Load and process injection data (label = 1)  
        if injection_data:
            inj_segments = self.create_segments(injection_data)
            for segment in inj_segments:
                processed = self.preprocess_for_neuromorphic(segment)
                data_segments.append(processed[0])  # Remove batch dim
                labels.append(1)  # Signal
        
        # ✅ NEW: Load and process foreground data (label = 1)
        if self.available_files.get('train_foreground'):
            for fg_file in self.available_files['train_foreground']:
                try:
                    fg_data = self.load_hdf5_file(fg_file)
                    fg_segments = self.create_segments(fg_data)
                    for segment in fg_segments:
                        processed = self.preprocess_for_neuromorphic(segment)
                        data_segments.append(processed[0])
                        labels.append(1)
                    logger.info(f"✅ Foreground file processed: {fg_file.name} → {len(fg_segments)} segments")
                except Exception as e:
                    logger.warning(f"⚠️ Skipping foreground file {fg_file}: {e}")
                
        logger.info(f"✅ Created labeled dataset")
        logger.info(f"   - Total segments: {len(data_segments)}")
        logger.info(f"   - Background segments: {labels.count(0)}")
        logger.info(f"   - Signal segments: {labels.count(1)}")
        
        return data_segments, labels
    
    def get_dataset_info(self) -> Dict:
        """Get comprehensive dataset information."""
        info = {
            'data_dir': str(self.data_dir),
            'mode': self.mode,
            'sample_rate': self.sample_rate,
            'segment_length': self.segment_length,
            'segment_samples': self.segment_samples,
            'available_files': {k: len(v) for k, v in self.available_files.items()}
        }
        
        return info


# Factory function for easy instantiation
def create_mlgwsc_loader(data_dir: Optional[str] = None, 
                        mode: str = "training",
                        config: Optional[Dict[str, Any]] = None) -> MLGWSCDataLoader:
    """
    Factory function to create MLGWSC data loader.
    
    Args:
        data_dir: Data directory path (uses config if None)
        mode: Loading mode
        config: Configuration dictionary (loads default if None)
        
    Returns:
        Configured MLGWSCDataLoader instance
    """
    return MLGWSCDataLoader(data_dir=data_dir, mode=mode, config=config)
