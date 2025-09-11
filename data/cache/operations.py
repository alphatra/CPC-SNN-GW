"""
Cache operations and utilities.

This module contains cache operation utilities and decorators extracted from
cache management files for better modularity.

Split from cache management files for better maintainability.
"""

import logging
import functools
import hashlib
import time
from typing import Any, Callable, Dict, Optional

from .manager import get_cache_manager, CacheMetadata

logger = logging.getLogger(__name__)


def cache_decorator(expiry_hours: float = 24.0,
                   enable_cache: bool = True,
                   cache_key_prefix: str = "",
                   include_args_in_key: bool = True):
    """
    Decorator for caching function results.
    
    Args:
        expiry_hours: Cache expiry time in hours
        enable_cache: Whether caching is enabled
        cache_key_prefix: Prefix for cache keys
        include_args_in_key: Whether to include function arguments in cache key
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not enable_cache:
                return func(*args, **kwargs)
            
            # Generate cache key
            if include_args_in_key:
                # Include function arguments in key
                key_data = f"{func.__name__}_{args}_{kwargs}"
            else:
                # Use only function name
                key_data = func.__name__
            
            cache_key = f"{cache_key_prefix}{hashlib.md5(key_data.encode()).hexdigest()}"
            
            # Try to get from cache
            cache_manager = get_cache_manager()
            cached_result = cache_manager.get(cache_key)
            
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Compute result
            logger.debug(f"Computing result for {func.__name__}")
            start_time = time.time()
            result = func(*args, **kwargs)
            computation_time = time.time() - start_time
            
            # Store in cache
            metadata = CacheMetadata(
                created_at=time.time(),
                data_type=type(result).__name__,
                source_info={
                    'function': func.__name__,
                    'computation_time': computation_time,
                    'args_included': include_args_in_key
                }
            )
            
            success = cache_manager.put(cache_key, result, metadata)
            
            if success:
                logger.debug(f"Cached result for {func.__name__} (computed in {computation_time:.3f}s)")
            else:
                logger.warning(f"Failed to cache result for {func.__name__}")
            
            return result
        
        # Add cache control methods to function
        wrapper.cache_clear = lambda: clear_function_cache(func.__name__, cache_key_prefix)
        wrapper.cache_info = lambda: get_function_cache_info(func.__name__, cache_key_prefix)
        
        return wrapper
    
    return decorator


def clear_cache_operations(pattern: Optional[str] = None):
    """
    Clear cache entries matching pattern.
    
    Args:
        pattern: Optional pattern to match keys (None = clear all)
    """
    cache_manager = get_cache_manager()
    
    if pattern is None:
        # Clear all
        cache_manager.clear()
        logger.info("All cache entries cleared")
    else:
        # Clear matching entries
        cleared_count = 0
        keys_to_remove = []
        
        with cache_manager._lock:
            for key in cache_manager._cache.keys():
                if pattern in key:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if cache_manager.evict(key):
                cleared_count += 1
        
        logger.info(f"Cleared {cleared_count} cache entries matching '{pattern}'")


def clear_function_cache(function_name: str, prefix: str = ""):
    """Clear cache entries for specific function."""
    pattern = f"{prefix}{function_name}"
    clear_cache_operations(pattern)


def get_function_cache_info(function_name: str, prefix: str = "") -> Dict[str, Any]:
    """Get cache information for specific function."""
    cache_manager = get_cache_manager()
    pattern = f"{prefix}{function_name}"
    
    matching_entries = 0
    total_size = 0
    
    with cache_manager._lock:
        for key, entry in cache_manager._cache.items():
            if pattern in key:
                matching_entries += 1
                total_size += entry.size_bytes
    
    return {
        'function_name': function_name,
        'cached_entries': matching_entries,
        'total_size_bytes': total_size,
        'total_size_mb': total_size / 1024 / 1024
    }


def cache_warming_decorator(cache_keys: list):
    """
    Decorator for warming up cache with specific keys.
    
    Args:
        cache_keys: List of cache keys to pre-warm
        
    Returns:
        Decorated function with cache warming
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Pre-warm cache if specified keys are missing
            cache_manager = get_cache_manager()
            
            missing_keys = []
            for key in cache_keys:
                if cache_manager.get(key) is None:
                    missing_keys.append(key)
            
            if missing_keys:
                logger.info(f"Cache warming needed for {len(missing_keys)} keys")
                # In practice, would trigger background warming
                
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def monitor_cache_performance() -> Dict[str, Any]:
    """Monitor and report cache performance metrics."""
    cache_manager = get_cache_manager()
    stats = cache_manager.get_statistics()
    
    # Calculate performance metrics
    performance = {
        'hit_rate': stats.hit_rate,
        'memory_usage_mb': stats.memory_usage_mb,
        'memory_usage_percent': (stats.total_size_bytes / stats.max_size_bytes) * 100,
        'entry_count': stats.entry_count,
        'memory_pressure': stats.memory_pressure,
        'total_requests': stats.hits + stats.misses,
        'eviction_rate': stats.evictions / max(stats.hits + stats.misses, 1)
    }
    
    # Performance assessment
    if performance['hit_rate'] > 0.8:
        performance['performance_rating'] = 'excellent'
    elif performance['hit_rate'] > 0.6:
        performance['performance_rating'] = 'good'
    elif performance['hit_rate'] > 0.4:
        performance['performance_rating'] = 'fair'
    else:
        performance['performance_rating'] = 'poor'
    
    return performance


# Re-export from manager for convenience
from .manager import CacheMetadata, CacheStatistics

# Export cache operations
__all__ = [
    "cache_decorator",
    "clear_cache_operations",
    "clear_function_cache",
    "get_function_cache_info",
    "cache_warming_decorator",
    "monitor_cache_performance",
    "CacheMetadata",
    "CacheStatistics"
]

