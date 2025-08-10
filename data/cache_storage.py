"""
Cache Storage: Storage Engine and Serialization
Extracted from cache_manager.py for modular architecture.
"""

import pickle
import json
from pathlib import Path
from typing import Any, Optional, Callable
import logging
import jax.numpy as jnp

logger = logging.getLogger(__name__)

# Optional dependencies for different formats
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class StorageEngine:
    """
    Unified storage engine for different data formats.
    
    Supports:
    - NumPy arrays (.npy)
    - JAX arrays (converted to/from NumPy)
    - HDF5 files (.h5, .hdf5)
    - Pickle files (.pkl)
    - JSON files (.json)
    """
    
    def __init__(self, enable_compression: bool = True):
        """
        Initialize storage engine.
        
        Args:
            enable_compression: Whether to enable compression for supported formats
        """
        self.enable_compression = enable_compression
        
    def save(self, data: Any, file_path: Path, extension: str = ".npy") -> bool:
        """
        Save data to file using appropriate format.
        
        Args:
            data: Data to save
            file_path: Path to save file
            extension: File extension determining format
            
        Returns:
            True if successful
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if extension == ".npy":
                return self._save_numpy(data, file_path)
            elif extension in [".h5", ".hdf5"]:
                return self._save_hdf5(data, file_path)
            elif extension == ".pkl":
                return self._save_pickle(data, file_path)
            elif extension == ".json":
                return self._save_json(data, file_path)
            else:
                logger.error(f"Unsupported file extension: {extension}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to save data to {file_path}: {e}")
            return False
    
    def load(self, file_path: Path, extension: str = ".npy") -> Optional[Any]:
        """
        Load data from file using appropriate format.
        
        Args:
            file_path: Path to load file
            extension: File extension determining format
            
        Returns:
            Loaded data or None if failed
        """
        try:
            if not file_path.exists():
                return None
                
            if extension == ".npy":
                return self._load_numpy(file_path)
            elif extension in [".h5", ".hdf5"]:
                return self._load_hdf5(file_path)
            elif extension == ".pkl":
                return self._load_pickle(file_path)
            elif extension == ".json":
                return self._load_json(file_path)
            else:
                logger.error(f"Unsupported file extension: {extension}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            return None
    
    def _save_numpy(self, data: Any, file_path: Path) -> bool:
        """Save data as NumPy array."""
        try:
            # Convert JAX arrays to NumPy
            if hasattr(data, '__array__'):
                data_np = jnp.asarray(data)
                if hasattr(data_np, 'device'):  # JAX array
                    data_np = np.array(data_np)
            else:
                data_np = np.array(data)
            
            if self.enable_compression:
                np.savez_compressed(file_path.with_suffix('.npz'), data=data_np)
            else:
                np.save(file_path, data_np)
            
            return True
        except Exception as e:
            logger.error(f"NumPy save failed: {e}")
            return False
    
    def _load_numpy(self, file_path: Path) -> Optional[Any]:
        """Load NumPy array data."""
        try:
            if file_path.suffix == '.npz':
                loaded = np.load(file_path)
                data = loaded['data']
            else:
                data = np.load(file_path)
            
            # Convert to JAX array
            return jnp.array(data)
        except Exception as e:
            logger.error(f"NumPy load failed: {e}")
            return None
    
    def _save_hdf5(self, data: Any, file_path: Path) -> bool:
        """Save data to HDF5 format."""
        if not HDF5_AVAILABLE:
            logger.error("HDF5 not available")
            return False
        
        try:
            compression = 'gzip' if self.enable_compression else None
            
            with h5py.File(file_path, 'w') as f:
                self._save_pytree_to_hdf5(f, data, compression)
            
            return True
        except Exception as e:
            logger.error(f"HDF5 save failed: {e}")
            return False
    
    def _load_hdf5(self, file_path: Path) -> Optional[Any]:
        """Load data from HDF5 format."""
        if not HDF5_AVAILABLE:
            logger.error("HDF5 not available")
            return None
        
        try:
            with h5py.File(file_path, 'r') as f:
                return self._load_pytree_from_hdf5(f)
        except Exception as e:
            logger.error(f"HDF5 load failed: {e}")
            return None
    
    def _save_pickle(self, data: Any, file_path: Path) -> bool:
        """Save data using pickle."""
        try:
            # Convert JAX arrays to NumPy for pickling
            serializable_data = self._convert_jax_to_numpy(data)
            
            with open(file_path, 'wb') as f:
                pickle.dump(serializable_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            return True
        except Exception as e:
            logger.error(f"Pickle save failed: {e}")
            return False
    
    def _load_pickle(self, file_path: Path) -> Optional[Any]:
        """Load data using pickle."""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Convert NumPy arrays back to JAX
            return self._convert_numpy_to_jax(data)
        except Exception as e:
            logger.error(f"Pickle load failed: {e}")
            return None
    
    def _save_json(self, data: Any, file_path: Path) -> bool:
        """Save data as JSON."""
        try:
            # Convert to JSON-serializable format
            serializable_data = self._convert_for_json(data)
            
            with open(file_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"JSON save failed: {e}")
            return False
    
    def _load_json(self, file_path: Path) -> Optional[Any]:
        """Load data from JSON."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert back from JSON format
            return self._convert_from_json(data)
        except Exception as e:
            logger.error(f"JSON load failed: {e}")
            return None
    
    def _save_pytree_to_hdf5(self, group, data: Any, compression: Optional[str] = None):
        """Recursively save pytree to HDF5."""
        if isinstance(data, dict):
            for key, value in data.items():
                subgroup = group.create_group(str(key))
                self._save_pytree_to_hdf5(subgroup, value, compression)
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                subgroup = group.create_group(str(i))
                self._save_pytree_to_hdf5(subgroup, item, compression)
        else:
            # Convert to numpy array
            if hasattr(data, '__array__'):
                array_data = np.array(data)
            else:
                array_data = np.array(data)
            
            group.create_dataset('data', data=array_data, compression=compression)
            group.attrs['type'] = type(data).__name__
    
    def _load_pytree_from_hdf5(self, group) -> Any:
        """Recursively load pytree from HDF5."""
        if 'data' in group:
            # Leaf node
            data = group['data'][...]
            return jnp.array(data)
        else:
            # Branch node - check if it's a list/tuple or dict
            keys = list(group.keys())
            
            # Try to determine if it's a sequence (all integer keys)
            try:
                int_keys = [int(k) for k in keys]
                int_keys.sort()
                if int_keys == list(range(len(keys))):
                    # It's a sequence
                    result = []
                    for i in int_keys:
                        result.append(self._load_pytree_from_hdf5(group[str(i)]))
                    return result
            except ValueError:
                pass
            
            # It's a dictionary
            result = {}
            for key in keys:
                result[key] = self._load_pytree_from_hdf5(group[key])
            return result
    
    def _convert_jax_to_numpy(self, data: Any) -> Any:
        """Convert JAX arrays to NumPy arrays recursively."""
        if hasattr(data, '__array__') and hasattr(data, 'device'):
            # JAX array
            return np.array(data)
        elif isinstance(data, dict):
            return {k: self._convert_jax_to_numpy(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            converted = [self._convert_jax_to_numpy(item) for item in data]
            return type(data)(converted)
        else:
            return data
    
    def _convert_numpy_to_jax(self, data: Any) -> Any:
        """Convert NumPy arrays to JAX arrays recursively."""
        if isinstance(data, np.ndarray):
            return jnp.array(data)
        elif isinstance(data, dict):
            return {k: self._convert_numpy_to_jax(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            converted = [self._convert_numpy_to_jax(item) for item in data]
            return type(data)(converted)
        else:
            return data
    
    def _convert_for_json(self, data: Any) -> Any:
        """Convert data to JSON-serializable format."""
        if hasattr(data, '__array__'):
            # Array-like (JAX or NumPy)
            array_data = np.array(data)
            return {
                '__array__': array_data.tolist(),
                '__dtype__': str(array_data.dtype),
                '__shape__': array_data.shape
            }
        elif isinstance(data, dict):
            return {k: self._convert_for_json(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._convert_for_json(item) for item in data]
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            # Fallback to string representation
            return str(data)
    
    def _convert_from_json(self, data: Any) -> Any:
        """Convert data from JSON format back to original types."""
        if isinstance(data, dict):
            if '__array__' in data:
                # Reconstruct array
                array_data = np.array(data['__array__'], dtype=data['__dtype__'])
                array_data = array_data.reshape(data['__shape__'])
                return jnp.array(array_data)
            else:
                return {k: self._convert_from_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_from_json(item) for item in data]
        else:
            return data


def cache_jax_function(storage_engine: StorageEngine, cache_dir: Path):
    """
    Decorator to cache JAX function results.
    
    Args:
        storage_engine: Storage engine instance
        cache_dir: Cache directory
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Generate cache key
            from .cache_metadata import generate_cache_key
            cache_key = generate_cache_key(func.__name__, *args, **kwargs)
            cache_file = cache_dir / f"{cache_key}.npy"
            
            # Try to load from cache
            if cache_file.exists():
                try:
                    result = storage_engine.load(cache_file, ".npy")
                    if result is not None:
                        logger.debug(f"Cache hit for {func.__name__}")
                        return result
                except Exception as e:
                    logger.warning(f"Cache load failed: {e}")
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Save to cache
            try:
                storage_engine.save(result, cache_file, ".npy")
                logger.debug(f"Cached result for {func.__name__}")
            except Exception as e:
                logger.warning(f"Cache save failed: {e}")
            
            return result
        return wrapper
    return decorator


def test_storage_engine():
    """Test storage engine functionality."""
    try:
        import tempfile
        
        engine = StorageEngine()
        
        # Test data
        test_data = {
            'array': jnp.array([1, 2, 3, 4, 5]),
            'nested': {
                'numbers': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
                'string': 'test'
            },
            'list': [1, 2, 3]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test different formats
            formats = [".npy", ".pkl", ".json"]
            if HDF5_AVAILABLE:
                formats.append(".h5")
            
            for ext in formats:
                test_file = temp_path / f"test{ext}"
                
                # Save and load
                success = engine.save(test_data, test_file, ext)
                assert success, f"Failed to save {ext}"
                
                loaded_data = engine.load(test_file, ext)
                assert loaded_data is not None, f"Failed to load {ext}"
                
                logger.info(f"✅ {ext} format test passed")
        
        logger.info("✅ Storage engine tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Storage engine test failed: {e}")
        return False 