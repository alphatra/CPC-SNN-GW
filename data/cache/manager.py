"""
Professional cache manager implementation.

This module contains the main caching functionality consolidated from
cache_manager.py, cache_storage.py, and cache_metadata.py for better modularity.

Split from multiple cache files for better maintainability.
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
        """Calculate cache hit rate."""
        total_requests = self.hits + self.misses
        return self.hits / total_requests if total_requests > 0 else 0.0
    
    @property 
    def memory_usage_mb(self) -> float:
        """Get memory usage in MB."""
        return self.total_size_bytes / 1024 / 1024


@dataclass
class CacheMetadata:
    """Metadata for cached items."""
    created_at: float
    data_type: str
    source_info: Dict[str, Any]
    quality_metrics: Optional[Dict[str, float]] = None
    dependencies: List[str] = field(default_factory=list)
    
    def is_expired(self, max_age_seconds: float) -> bool:
        """Check if cache entry is expired."""
        return (time.time() - self.created_at) > max_age_seconds


class ProfessionalCacheManager:
    """
    Professional cache manager with LRU eviction and memory management.
    
    Features:
    - LRU (Least Recently Used) eviction policy
    - Memory pressure monitoring
    - Thread-safe operations
    - Persistent storage support
    - Cache statistics and performance monitoring
    - Automatic cleanup and maintenance
    """
    
    def __init__(self, 
                 max_size_bytes: int = 2 * 1024**3,  # 2GB default
                 max_entries: int = 10000,
                 enable_persistence: bool = True,
                 cache_dir: Optional[Path] = None,
                 memory_pressure_threshold: float = 0.8):
        """
        Initialize professional cache manager.
        
        Args:
            max_size_bytes: Maximum cache size in bytes
            max_entries: Maximum number of cache entries
            enable_persistence: Enable persistent cache storage
            cache_dir: Directory for persistent cache files
            memory_pressure_threshold: Memory pressure threshold for aggressive cleanup
        """
        self.max_size_bytes = max_size_bytes
        self.max_entries = max_entries
        self.enable_persistence = enable_persistence
        self.memory_pressure_threshold = memory_pressure_threshold
        
        # Cache storage (OrderedDict for LRU)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Statistics
        self.stats = CacheStatistics(max_size_bytes=max_size_bytes)
        
        # Cache directory setup
        if cache_dir is None:
            cache_dir = Path.home() / '.cpc_snn_cache'
        self.cache_dir = Path(cache_dir)
        
        if self.enable_persistence:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Background maintenance
        self._maintenance_thread = None
        self._shutdown_event = threading.Event()
        self._start_maintenance()
        
        logger.info(f"ProfessionalCacheManager initialized: "
                   f"max_size={max_size_bytes/1024**3:.1f}GB, "
                   f"max_entries={max_entries}, "
                   f"persistence={enable_persistence}")
    
    def _start_maintenance(self):
        """Start background maintenance thread."""
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True,
            name="CacheMaintenanceThread"
        )
        self._maintenance_thread.start()
    
    def _maintenance_loop(self):
        """Background maintenance loop."""
        while not self._shutdown_event.is_set():
            try:
                # Check memory pressure
                self._check_memory_pressure()
                
                # Cleanup expired entries
                self._cleanup_expired_entries()
                
                # Update statistics
                self._update_statistics()
                
                # Sleep for maintenance interval
                self._shutdown_event.wait(30.0)  # 30 second intervals
                
            except Exception as e:
                logger.error(f"Cache maintenance error: {e}")
                self._shutdown_event.wait(60.0)  # Wait longer on error
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache with LRU update.
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None if not found
        """
        with self._lock:
            if key in self._cache:
                # Update LRU order and access stats
                entry = self._cache[key]
                entry.update_access()
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                
                self.stats.hits += 1
                
                logger.debug(f"Cache HIT: {key}")
                return entry.data
            else:
                self.stats.misses += 1
                logger.debug(f"Cache MISS: {key}")
                return None
    
    def put(self, key: str, data: Any, metadata: Optional[CacheMetadata] = None) -> bool:
        """
        Put item in cache with automatic eviction if needed.
        
        Args:
            key: Cache key
            data: Data to cache
            metadata: Optional metadata
            
        Returns:
            True if successfully cached, False otherwise
        """
        with self._lock:
            # Calculate data size
            try:
                if hasattr(data, 'nbytes'):
                    data_size = data.nbytes
                elif isinstance(data, (list, tuple)):
                    data_size = sum(getattr(item, 'nbytes', 256) for item in data)
                else:
                    data_size = len(pickle.dumps(data))
            except:
                data_size = 1024  # Default estimate
            
            # Check if we can fit this entry
            if data_size > self.max_size_bytes:
                logger.warning(f"Data too large for cache: {data_size} bytes")
                return False
            
            # Evict entries if necessary
            self._evict_if_necessary(data_size)
            
            # Create cache entry
            entry = CacheEntry(
                data=data,
                timestamp=time.time(),
                size_bytes=data_size,
                key_hash=hashlib.md5(key.encode()).hexdigest(),
                data_type=type(data).__name__
            )
            
            # Store in cache
            if key in self._cache:
                # Remove old entry size from stats
                old_entry = self._cache[key]
                self.stats.total_size_bytes -= old_entry.size_bytes
            
            self._cache[key] = entry
            self.stats.total_size_bytes += data_size
            self.stats.entry_count = len(self._cache)
            
            # Persistent storage (if enabled)
            if self.enable_persistence and metadata:
                self._save_persistent_entry(key, data, metadata)
            
            logger.debug(f"Cache PUT: {key} ({data_size} bytes)")
            return True
    
    def _evict_if_necessary(self, incoming_size: int):
        """Evict entries using LRU policy if necessary."""
        # Check size constraint
        while (self.stats.total_size_bytes + incoming_size > self.max_size_bytes and 
               len(self._cache) > 0):
            self._evict_lru_entry()
        
        # Check entry count constraint  
        while len(self._cache) >= self.max_entries:
            self._evict_lru_entry()
    
    def _evict_lru_entry(self):
        """Evict least recently used entry."""
        if len(self._cache) == 0:
            return
        
        # Get LRU entry (first in OrderedDict)
        lru_key, lru_entry = self._cache.popitem(last=False)
        
        # Update statistics
        self.stats.total_size_bytes -= lru_entry.size_bytes
        self.stats.evictions += 1
        self.stats.entry_count = len(self._cache)
        
        logger.debug(f"Cache EVICT: {lru_key} ({lru_entry.size_bytes} bytes)")
    
    def _check_memory_pressure(self):
        """Check system memory pressure and trigger cleanup if needed."""
        try:
            # Get system memory info
            memory_info = psutil.virtual_memory()
            memory_pressure = memory_info.percent / 100.0
            
            self.stats.memory_pressure = memory_pressure
            
            # Aggressive cleanup under memory pressure
            if memory_pressure > self.memory_pressure_threshold:
                logger.warning(f"High memory pressure: {memory_pressure:.1%}")
                
                # Evict 25% of entries
                num_to_evict = len(self._cache) // 4
                for _ in range(num_to_evict):
                    if len(self._cache) > 0:
                        self._evict_lru_entry()
                
                logger.info(f"Emergency eviction: removed {num_to_evict} entries")
                
        except Exception as e:
            logger.error(f"Error checking memory pressure: {e}")
    
    def _cleanup_expired_entries(self, max_age_hours: float = 24.0):
        """Remove expired cache entries."""
        max_age_seconds = max_age_hours * 3600
        current_time = time.time()
        
        expired_keys = []
        
        with self._lock:
            for key, entry in self._cache.items():
                if (current_time - entry.timestamp) > max_age_seconds:
                    expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            self.evict(key)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _update_statistics(self):
        """Update cache statistics."""
        with self._lock:
            self.stats.entry_count = len(self._cache)
            self.stats.total_size_bytes = sum(entry.size_bytes for entry in self._cache.values())
    
    def _save_persistent_entry(self, key: str, data: Any, metadata: CacheMetadata):
        """Save cache entry to persistent storage."""
        try:
            cache_file = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.pkl"
            
            persistent_data = {
                'data': data,
                'metadata': metadata,
                'timestamp': time.time()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(persistent_data, f)
                
        except Exception as e:
            logger.warning(f"Failed to save persistent cache entry: {e}")
    
    def evict(self, key: str) -> bool:
        """Manually evict specific cache entry."""
        with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                self.stats.total_size_bytes -= entry.size_bytes
                self.stats.evictions += 1
                self.stats.entry_count = len(self._cache)
                logger.debug(f"Manual eviction: {key}")
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            num_entries = len(self._cache)
            self._cache.clear()
            self.stats.total_size_bytes = 0
            self.stats.entry_count = 0
            self.stats.evictions += num_entries
            
            logger.info(f"Cache cleared: {num_entries} entries removed")
    
    def get_statistics(self) -> CacheStatistics:
        """Get current cache statistics."""
        self._update_statistics()
        return self.stats
    
    def shutdown(self):
        """Shutdown cache manager and cleanup resources."""
        logger.info("Shutting down cache manager...")
        
        # Stop maintenance thread
        if self._maintenance_thread and self._maintenance_thread.is_alive():
            self._shutdown_event.set()
            self._maintenance_thread.join(timeout=5.0)
        
        # Clear cache
        self.clear()
        
        logger.info("Cache manager shutdown complete")


# Global cache manager instance
_global_cache_manager: Optional[ProfessionalCacheManager] = None
_cache_lock = threading.Lock()


def get_cache_manager() -> ProfessionalCacheManager:
    """Get global cache manager instance (singleton)."""
    global _global_cache_manager
    
    if _global_cache_manager is None:
        with _cache_lock:
            if _global_cache_manager is None:
                _global_cache_manager = ProfessionalCacheManager()
    
    return _global_cache_manager


def create_professional_cache(max_size_gb: float = 2.0,
                            max_entries: int = 10000,
                            cache_dir: Optional[Path] = None) -> ProfessionalCacheManager:
    """
    Factory function for creating professional cache manager.
    
    Args:
        max_size_gb: Maximum cache size in gigabytes
        max_entries: Maximum number of cache entries
        cache_dir: Directory for persistent cache files
        
    Returns:
        Configured ProfessionalCacheManager
    """
    max_size_bytes = int(max_size_gb * 1024**3)
    
    return ProfessionalCacheManager(
        max_size_bytes=max_size_bytes,
        max_entries=max_entries,
        enable_persistence=True,
        cache_dir=cache_dir
    )


# Export main cache components
__all__ = [
    "ProfessionalCacheManager",
    "CacheEntry",
    "CacheStatistics", 
    "CacheMetadata",
    "get_cache_manager",
    "create_professional_cache"
]

