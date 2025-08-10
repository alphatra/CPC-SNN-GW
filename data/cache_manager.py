"""
Professional Cache Manager with LRU Eviction Policies
Addresses Executive Summary Priority 3: Memory Leak Prevention
"""

import jax
import jax.numpy as jnp
import numpy as np
import logging
import threading
import weakref
import time
import psutil
import os
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict
import pickle
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Enhanced cache entry with LRU tracking and metadata."""
    data: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    key_hash: str = ""
    data_type: str = ""
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access = time.time()

@dataclass 
class CacheStatistics:
    """Cache performance and memory statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    max_size_bytes: int = 0
    entry_count: int = 0
    memory_pressure: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate

class LRUCache:
    """
    Least Recently Used cache with memory pressure monitoring.
    Implements automatic eviction to prevent memory leaks.
    """
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 2048):
        """
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self.stats = CacheStatistics(max_size_bytes=self.max_memory_bytes)
        
        logger.info(f"Initialized LRU cache: max_size={max_size}, max_memory={max_memory_mb}MB")
    
    def _compute_size(self, data: Any) -> int:
        """Estimate memory size of data object."""
        try:
            if isinstance(data, (jnp.ndarray, np.ndarray)):
                return data.nbytes
            elif isinstance(data, dict):
                return sum(self._compute_size(v) for v in data.values()) + \
                       sum(len(str(k).encode()) for k in data.keys())
            elif isinstance(data, (list, tuple)):
                return sum(self._compute_size(item) for item in data)
            elif isinstance(data, str):
                return len(data.encode())
            else:
                # Fallback: pickle size estimation
                return len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as e:
            logger.warning(f"Size estimation failed: {e}")
            return 1024  # Default 1KB estimate
    
    def _generate_key(self, key: str, data: Any) -> str:
        """Generate consistent hash key for cache entry."""
        try:
            # Create hash from key and data characteristics
            if isinstance(data, (jnp.ndarray, np.ndarray)):
                data_sig = f"{data.shape}_{data.dtype}_{np.sum(data.flat[:10])}"
            else:
                data_sig = str(type(data).__name__)
                
            combined = f"{key}_{data_sig}"
            return hashlib.md5(combined.encode()).hexdigest()[:16]
        except Exception:
            return hashlib.md5(key.encode()).hexdigest()[:16]
    
    def _check_memory_pressure(self) -> float:
        """Check system memory pressure."""
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        except Exception:
            return 0.0
    
    def _evict_lru_entries(self, target_reduction_bytes: int = 0):
        """Evict least recently used entries to free memory."""
        if not self._cache:
            return
            
        # Sort by last access time (oldest first)
        sorted_items = sorted(
            self._cache.items(),
            key=lambda x: x[1].last_access
        )
        
        bytes_freed = 0
        entries_evicted = 0
        
        for key, entry in sorted_items:
            if (target_reduction_bytes > 0 and bytes_freed >= target_reduction_bytes) or \
               (len(self._cache) <= self.max_size // 2):
                break
                
            # Remove entry
            del self._cache[key]
            bytes_freed += entry.size_bytes
            entries_evicted += 1
            
        self.stats.evictions += entries_evicted
        self.stats.total_size_bytes -= bytes_freed
        
        if entries_evicted > 0:
            logger.info(f"LRU eviction: {entries_evicted} entries, {bytes_freed/1024/1024:.1f}MB freed")
    
    def _should_evict(self) -> bool:
        """Determine if eviction is needed."""
        # Size-based eviction
        if len(self._cache) >= self.max_size:
            return True
            
        # Memory-based eviction
        if self.stats.total_size_bytes >= self.max_memory_bytes:
            return True
            
        # System memory pressure eviction
        memory_pressure = self._check_memory_pressure()
        if memory_pressure > 0.85:  # 85% system memory usage
            logger.warning(f"High system memory pressure: {memory_pressure:.1%}")
            return True
            
        return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with LRU update."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry.update_access()
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self.stats.hits += 1
                return entry.data
            else:
                self.stats.misses += 1
                return None
    
    def put(self, key: str, data: Any, data_type: str = "unknown"):
        """Put item in cache with automatic eviction."""
        with self._lock:
            # Compute entry metadata
            size_bytes = self._compute_size(data)
            key_hash = self._generate_key(key, data)
            
            # Create cache entry
            entry = CacheEntry(
                data=data,
                timestamp=time.time(),
                size_bytes=size_bytes,
                key_hash=key_hash,
                data_type=data_type
            )
            
            # Check if eviction needed before adding
            if self._should_evict():
                # Evict at least 25% of current memory or 100MB, whichever is larger
                target_reduction = max(
                    self.stats.total_size_bytes // 4,
                    100 * 1024 * 1024  # 100MB
                )
                self._evict_lru_entries(target_reduction)
            
            # Add/update entry
            if key in self._cache:
                old_size = self._cache[key].size_bytes
                self.stats.total_size_bytes = self.stats.total_size_bytes - old_size + size_bytes
            else:
                self.stats.total_size_bytes += size_bytes
                self.stats.entry_count += 1
                
            self._cache[key] = entry
            self._cache.move_to_end(key)  # Mark as most recently used
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self.stats.total_size_bytes = 0
            self.stats.entry_count = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> CacheStatistics:
        """Get current cache statistics."""
        with self._lock:
            self.stats.entry_count = len(self._cache)
            self.stats.memory_pressure = self._check_memory_pressure()
            return self.stats

class ProfessionalCacheManager:
    """
    Professional cache manager with LRU eviction and memory leak prevention.
    Addresses all Executive Summary memory-related issues.
    """
    
    def __init__(self, 
                 cache_dir: Optional[Path] = None,
                 memory_limit_mb: int = 2048,
                 enable_disk_cache: bool = True,
                 enable_auto_cleanup: bool = True):
        """
        Args:
            cache_dir: Directory for persistent cache
            memory_limit_mb: Memory limit for in-memory cache
            enable_disk_cache: Enable persistent disk caching
            enable_auto_cleanup: Enable automatic cleanup
        """
        self.cache_dir = cache_dir or Path("./cache")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Memory cache with LRU eviction
        self.memory_cache = LRUCache(max_size=1000, max_memory_mb=memory_limit_mb)
        
        # Disk cache management
        self.enable_disk_cache = enable_disk_cache
        self.disk_cache_dir = self.cache_dir / "disk_cache"
        if enable_disk_cache:
            self.disk_cache_dir.mkdir(exist_ok=True)
        
        # Auto-cleanup configuration
        self.enable_auto_cleanup = enable_auto_cleanup
        self.cleanup_interval = 3600  # 1 hour
        self.last_cleanup = time.time()
        
        # Thread-safe operations
        self._global_lock = threading.RLock()
        
        # Weak references to prevent circular references
        self._weak_refs: weakref.WeakSet = weakref.WeakSet()
        
        logger.info(f"Professional cache manager initialized: "
                   f"memory_limit={memory_limit_mb}MB, disk_cache={enable_disk_cache}")
    
    def _should_auto_cleanup(self) -> bool:
        """Check if auto cleanup should run."""
        return (self.enable_auto_cleanup and 
                time.time() - self.last_cleanup > self.cleanup_interval)
    
    def _auto_cleanup(self):
        """Perform automatic cache cleanup."""
        if not self._should_auto_cleanup():
            return
            
        logger.info("Starting automatic cache cleanup")
        
        try:
            # Clean memory cache if under memory pressure
            memory_pressure = self.memory_cache._check_memory_pressure()
            if memory_pressure > 0.75:
                logger.warning(f"Memory pressure {memory_pressure:.1%}, triggering cleanup")
                self.memory_cache._evict_lru_entries(
                    self.memory_cache.stats.total_size_bytes // 3
                )
            
            # Clean disk cache if enabled
            if self.enable_disk_cache:
                self._cleanup_disk_cache()
                
            self.last_cleanup = time.time()
            logger.info("Automatic cleanup completed")
            
        except Exception as e:
            logger.error(f"Auto cleanup failed: {e}")
    
    def _cleanup_disk_cache(self):
        """Clean old entries from disk cache."""
        if not self.disk_cache_dir.exists():
            return
            
        try:
            # Remove files older than 24 hours
            cutoff_time = time.time() - 24 * 3600
            removed_count = 0
            removed_size = 0
            
            for cache_file in self.disk_cache_dir.glob("*.pkl"):
                if cache_file.stat().st_mtime < cutoff_time:
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    removed_count += 1
                    removed_size += file_size
                    
            if removed_count > 0:
                logger.info(f"Disk cache cleanup: {removed_count} files, "
                           f"{removed_size/1024/1024:.1f}MB removed")
                           
        except Exception as e:
            logger.error(f"Disk cache cleanup failed: {e}")
    
    def _disk_cache_key(self, key: str) -> Path:
        """Generate disk cache file path."""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.disk_cache_dir / f"{safe_key}.pkl"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache (memory first, then disk)."""
        with self._global_lock:
            # Auto cleanup if needed
            self._auto_cleanup()
            
            # Try memory cache first
            result = self.memory_cache.get(key)
            if result is not None:
                return result
                
            # Try disk cache if enabled
            if self.enable_disk_cache:
                disk_file = self._disk_cache_key(key)
                if disk_file.exists():
                    try:
                        with open(disk_file, 'rb') as f:
                            data = pickle.load(f)
                        
                        # Load back into memory cache
                        self.memory_cache.put(key, data, "disk_loaded")
                        logger.debug(f"Loaded from disk cache: {key}")
                        return data
                        
                    except Exception as e:
                        logger.warning(f"Failed to load disk cache {key}: {e}")
                        # Remove corrupted file
                        try:
                            disk_file.unlink()
                        except:
                            pass
            
            return default
    
    def put(self, key: str, data: Any, persist_to_disk: bool = False,
            data_type: str = "unknown"):
        """
        Put item in cache with optional disk persistence.
        
        Args:
            key: Cache key
            data: Data to cache
            persist_to_disk: Whether to save to disk
            data_type: Type of data for diagnostics
        """
        with self._global_lock:
            # Store in memory cache
            self.memory_cache.put(key, data, data_type)
            
            # Optionally persist to disk
            if persist_to_disk and self.enable_disk_cache:
                try:
                    disk_file = self._disk_cache_key(key)
                    with open(disk_file, 'wb') as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.debug(f"Persisted to disk: {key}")
                except Exception as e:
                    logger.warning(f"Failed to persist {key} to disk: {e}")
    
    def invalidate(self, key: str):
        """Remove item from both memory and disk cache."""
        with self._global_lock:
            # Remove from memory
            if key in self.memory_cache._cache:
                entry = self.memory_cache._cache[key]
                self.memory_cache.stats.total_size_bytes -= entry.size_bytes
                del self.memory_cache._cache[key]
                
            # Remove from disk
            if self.enable_disk_cache:
                disk_file = self._disk_cache_key(key)
                if disk_file.exists():
                    try:
                        disk_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove disk cache {key}: {e}")
    
    def clear_all(self):
        """Clear all caches (memory and disk)."""
        with self._global_lock:
            # Clear memory cache
            self.memory_cache.clear()
            
            # Clear disk cache
            if self.enable_disk_cache and self.disk_cache_dir.exists():
                try:
                    for cache_file in self.disk_cache_dir.glob("*.pkl"):
                        cache_file.unlink()
                    logger.info("Disk cache cleared")
                except Exception as e:
                    logger.error(f"Failed to clear disk cache: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory and cache statistics."""
        memory_stats = self.memory_cache.get_stats()
        
        # System memory info
        try:
            system_memory = psutil.virtual_memory()
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info()
        except Exception:
            system_memory = None
            process_memory = None
        
        # Disk cache stats
        disk_stats = {'enabled': False}
        if self.enable_disk_cache and self.disk_cache_dir.exists():
            try:
                disk_files = list(self.disk_cache_dir.glob("*.pkl"))
                disk_size = sum(f.stat().st_size for f in disk_files)
                disk_stats = {
                    'enabled': True,
                    'file_count': len(disk_files),
                    'total_size_mb': disk_size / (1024 * 1024)
                }
            except Exception as e:
                disk_stats = {'enabled': True, 'error': str(e)}
        
        return {
            'memory_cache': {
                'hit_rate': memory_stats.hit_rate,
                'miss_rate': memory_stats.miss_rate,
                'entry_count': memory_stats.entry_count,
                'size_mb': memory_stats.total_size_bytes / (1024 * 1024),
                'max_size_mb': memory_stats.max_size_bytes / (1024 * 1024),
                'evictions': memory_stats.evictions
            },
            'disk_cache': disk_stats,
            'system_memory': {
                'percent_used': system_memory.percent if system_memory else None,
                'available_gb': system_memory.available / (1024**3) if system_memory else None,
                'process_rss_mb': process_memory.rss / (1024**2) if process_memory else None
            }
        }
    
    def force_cleanup(self):
        """Force immediate cleanup of all caches."""
        with self._global_lock:
            logger.info("Forcing cache cleanup")
            
            # Aggressive memory cache cleanup
            self.memory_cache._evict_lru_entries(
                self.memory_cache.stats.total_size_bytes // 2
            )
            
            # Disk cache cleanup
            if self.enable_disk_cache:
                self._cleanup_disk_cache()
                
            self.last_cleanup = time.time()
            
            # Log final stats
            stats = self.get_memory_stats()
            logger.info(f"Cleanup complete. Memory cache: {stats['memory_cache']['size_mb']:.1f}MB")

# Factory functions and utilities

def create_professional_cache(memory_limit_mb: int = 2048, 
                            cache_dir: Optional[Path] = None) -> ProfessionalCacheManager:
    """Create professional cache manager with optimized settings."""
    return ProfessionalCacheManager(
        cache_dir=cache_dir,
        memory_limit_mb=memory_limit_mb,
        enable_disk_cache=True,
        enable_auto_cleanup=True
    )

def monitor_memory_usage(cache_manager: ProfessionalCacheManager) -> Dict[str, Any]:
    """Monitor and log cache memory usage."""
    stats = cache_manager.get_memory_stats()
    
    # Log warnings for high usage
    memory_pct = stats['memory_cache']['size_mb'] / stats['memory_cache']['max_size_mb']
    if memory_pct > 0.8:
        logger.warning(f"High cache memory usage: {memory_pct:.1%}")
        
    system_pct = stats['system_memory'].get('percent_used', 0)
    if system_pct > 85:
        logger.warning(f"High system memory usage: {system_pct:.1f}%")
        
    return stats 