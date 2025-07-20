"""
Cache Manager: Main Interface for Professional Cache Management
Extracted and refactored from original cache_manager.py for modular architecture.
"""

from pathlib import Path
from typing import Any, Optional, Callable, Dict, Union
import logging
import time

from .cache_metadata import (
    CacheMetadata, CacheStatistics, LRUCacheWithTTL,
    get_cache_dir, get_cache_size, load_metadata_file, 
    save_metadata_file, cleanup_expired_entries,
    compute_file_hash, verify_file_integrity, generate_cache_key
)
from .cache_storage import StorageEngine, cache_jax_function

logger = logging.getLogger(__name__)


class ProfessionalCacheManager:
    """
    Professional cache manager with metadata tracking and multiple storage formats.
    
    Features:
    - Automatic size management with LRU eviction
    - File integrity verification with SHA256 hashes
    - Multiple storage formats (NumPy, HDF5, Pickle, JSON)
    - TTL support and automatic cleanup
    - Comprehensive statistics and monitoring
    """
    
    def __init__(self,
                 app_name: str = "ligo-cpc-snn",
                 max_cache_size_gb: float = 5.0,
                 max_age_days: int = 30,
                 cleanup_threshold: float = 0.9,
                 enable_verification: bool = True,
                 cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize professional cache manager.
        
        Args:
            app_name: Application name for cache directory
            max_cache_size_gb: Maximum cache size in GB
            max_age_days: Maximum age for cache entries in days
            cleanup_threshold: Cleanup when cache reaches this fraction of max size
            enable_verification: Enable file integrity verification
            cache_dir: Custom cache directory (optional)
        """
        self.app_name = app_name
        self.max_cache_size_bytes = int(max_cache_size_gb * 1e9)
        self.max_age_days = max_age_days
        self.cleanup_threshold = cleanup_threshold
        self.enable_verification = enable_verification
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = get_cache_dir(app_name)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
        # Initialize components
        self.storage_engine = StorageEngine(enable_compression=True)
        self.in_memory_cache = LRUCacheWithTTL(max_size=100, ttl_seconds=3600)
        self.statistics = CacheStatistics()
        
        # Load existing metadata
        self.metadata = load_metadata_file(self.metadata_file)
        
        logger.info(f"Initialized ProfessionalCacheManager")
        logger.info(f"  Cache directory: {self.cache_dir}")
        logger.info(f"  Max size: {max_cache_size_gb:.1f} GB")
        logger.info(f"  Existing entries: {len(self.metadata)}")


    def get_cache_file_path(self, cache_key: str, extension: str = ".npy") -> Path:
        """Get full path for cache file."""
        return self.cache_dir / f"{cache_key}{extension}"


    def put(self, cache_key: str, data: Any, extension: str = ".npy") -> bool:
        """
        Store data in cache with metadata tracking.
        
        Args:
            cache_key: Unique cache key
            data: Data to store
            extension: File extension determining storage format
            
        Returns:
            True if successful
        """
        try:
            file_path = self.get_cache_file_path(cache_key, extension)
            
            # Save to storage
            success = self.storage_engine.save(data, file_path, extension)
            if not success:
                return False
            
            # Compute file hash for integrity verification
            file_hash = compute_file_hash(file_path) if self.enable_verification else ""
            file_size = file_path.stat().st_size
            
            # Create metadata
            current_time = time.time()
            metadata = CacheMetadata(
                filename=file_path.name,
                sha256=file_hash,
                created_at=current_time,
                last_accessed=current_time,
                file_size=file_size,
                cache_key=cache_key
            )
            
            # Store metadata
            self.metadata[cache_key] = metadata
            save_metadata_file(self.metadata, self.metadata_file)
            
            # Update statistics
            self.statistics.total_entries = len(self.metadata)
            self.statistics.total_size_bytes = get_cache_size(self.cache_dir)
            
            # Cache in memory for fast access
            self.in_memory_cache.put(cache_key, data)
            
            # Check if cleanup is needed
            self._maybe_cleanup()
            
            logger.debug(f"Cached data with key: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache data: {e}")
            return False


    def get(self, cache_key: str, extension: str = ".npy") -> Optional[Any]:
        """
        Retrieve data from cache.
        
        Args:
            cache_key: Cache key to retrieve
            extension: File extension
            
        Returns:
            Cached data or None if not found
        """
        try:
            # Try in-memory cache first
            data = self.in_memory_cache.get(cache_key)
            if data is not None:
                self.statistics.hit_count += 1
                if cache_key in self.metadata:
                    self.metadata[cache_key].update_access_time()
                return data
            
            # Check if key exists in metadata
            if cache_key not in self.metadata:
                self.statistics.miss_count += 1
                return None
            
            metadata = self.metadata[cache_key]
            file_path = self.cache_dir / metadata.filename
            
            # Check if file exists
            if not file_path.exists():
                logger.warning(f"Cache file missing: {file_path}")
                self._remove_cache_entry(cache_key)
                self.statistics.miss_count += 1
                return None
            
            # Verify file integrity if enabled
            if self.enable_verification and metadata.sha256:
                if not verify_file_integrity(file_path, metadata.sha256):
                    logger.warning(f"Cache file corrupted: {file_path}")
                    self._remove_cache_entry(cache_key)
                    self.statistics.miss_count += 1
                    return None
            
            # Load data
            data = self.storage_engine.load(file_path, extension)
            if data is None:
                logger.warning(f"Failed to load cache file: {file_path}")
                self._remove_cache_entry(cache_key)
                self.statistics.miss_count += 1
                return None
            
            # Update access time and cache in memory
            metadata.update_access_time()
            self.in_memory_cache.put(cache_key, data)
            save_metadata_file(self.metadata, self.metadata_file)
            
            self.statistics.hit_count += 1
            logger.debug(f"Cache hit for key: {cache_key}")
            return data
            
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
            self.statistics.miss_count += 1
            return None


    def _remove_cache_entry(self, cache_key: str) -> None:
        """Remove cache entry and its file."""
        try:
            if cache_key in self.metadata:
                metadata = self.metadata[cache_key]
                file_path = self.cache_dir / metadata.filename
                
                # Remove file
                if file_path.exists():
                    file_path.unlink()
                
                # Remove from metadata
                del self.metadata[cache_key]
                save_metadata_file(self.metadata, self.metadata_file)
                
                self.statistics.eviction_count += 1
                
        except Exception as e:
            logger.error(f"Failed to remove cache entry {cache_key}: {e}")


    def _maybe_cleanup(self) -> None:
        """Check if cleanup is needed and perform it."""
        current_size = get_cache_size(self.cache_dir)
        if current_size > self.max_cache_size_bytes * self.cleanup_threshold:
            self.cleanup()


    def cleanup(self, days: Optional[int] = None) -> Dict[str, Any]:
        """
        Clean up expired and old cache entries.
        
        Args:
            days: Custom age threshold (uses default if None)
            
        Returns:
            Cleanup statistics
        """
        try:
            cleanup_days = days or self.max_age_days
            
            # Remove expired entries
            expired_count = cleanup_expired_entries(
                self.metadata, self.cache_dir, cleanup_days
            )
            
            # Remove LRU entries if still over size limit
            lru_removed = 0
            current_size = get_cache_size(self.cache_dir)
            
            if current_size > self.max_cache_size_bytes:
                # Sort by last accessed time (LRU)
                sorted_entries = sorted(
                    self.metadata.items(),
                    key=lambda x: x[1].last_accessed
                )
                
                for cache_key, metadata in sorted_entries:
                    if current_size <= self.max_cache_size_bytes * 0.8:
                        break
                    
                    self._remove_cache_entry(cache_key)
                    current_size -= metadata.file_size
                    lru_removed += 1
            
            # Update metadata file
            save_metadata_file(self.metadata, self.metadata_file)
            
            # Update statistics
            self.statistics.total_entries = len(self.metadata)
            self.statistics.total_size_bytes = get_cache_size(self.cache_dir)
            
            cleanup_stats = {
                'expired_removed': expired_count,
                'lru_removed': lru_removed,
                'total_removed': expired_count + lru_removed,
                'remaining_entries': len(self.metadata),
                'cache_size_mb': self.statistics.size_mb
            }
            
            logger.info(f"Cache cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            return {'error': str(e)}


    def clear_all(self) -> bool:
        """Clear all cache entries."""
        try:
            # Remove all files
            for metadata in self.metadata.values():
                file_path = self.cache_dir / metadata.filename
                if file_path.exists():
                    file_path.unlink()
            
            # Clear metadata
            self.metadata.clear()
            save_metadata_file(self.metadata, self.metadata_file)
            
            # Clear in-memory cache
            self.in_memory_cache.clear()
            
            # Reset statistics
            self.statistics = CacheStatistics()
            
            logger.info("Cache cleared completely")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False


    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        self.statistics.total_entries = len(self.metadata)
        self.statistics.total_size_bytes = get_cache_size(self.cache_dir)
        
        return {
            **self.statistics.to_dict(),
            'cache_dir': str(self.cache_dir),
            'max_size_gb': self.max_cache_size_bytes / 1e9,
            'max_age_days': self.max_age_days,
            'in_memory_entries': self.in_memory_cache.size()
        }


    def print_statistics(self) -> None:
        """Print formatted cache statistics."""
        stats = self.get_statistics()
        
        print("📊 Cache Statistics")
        print("=" * 50)
        print(f"📁 Directory: {stats['cache_dir']}")
        print(f"📦 Total entries: {stats['total_entries']}")
        print(f"💾 Total size: {stats['total_size_mb']:.1f} MB")
        print(f"🎯 Hit rate: {stats['hit_rate']:.1%}")
        print(f"🔥 Memory cached: {stats['in_memory_entries']}")
        print(f"📤 Evictions: {stats['eviction_count']}")
        print(f"🔧 Max size: {stats['max_size_gb']:.1f} GB")


def get_cache_manager(**kwargs) -> ProfessionalCacheManager:
    """Get global cache manager instance."""
    # Singleton pattern
    if not hasattr(get_cache_manager, '_instance'):
        get_cache_manager._instance = ProfessionalCacheManager(**kwargs)
    return get_cache_manager._instance


def cache_decorator(key_func: Optional[Callable] = None, extension: str = ".npy"):
    """
    Decorator for caching function results.
    
    Args:
        key_func: Function to generate cache key (uses generate_cache_key if None)
        extension: File extension for storage format
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = generate_cache_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            result = cache_manager.get(cache_key, extension)
            if result is not None:
                return result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result
            cache_manager.put(cache_key, result, extension)
            
            return result
        return wrapper
    return decorator 