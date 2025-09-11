"""
Cache Module: Professional Caching System

Modular implementation of caching components consolidated from
cache_manager.py, cache_storage.py, cache_metadata.py for better maintainability.

Components:
- manager: Main ProfessionalCacheManager class
- operations: Cache operations and utilities
"""

from .manager import (
    ProfessionalCacheManager,
    get_cache_manager,
    create_professional_cache
)
from .operations import (
    cache_decorator,
    clear_cache_operations,
    CacheMetadata,
    CacheStatistics
)

__all__ = [
    # Main manager
    "ProfessionalCacheManager",
    "get_cache_manager", 
    "create_professional_cache",
    
    # Operations
    "cache_decorator",
    "clear_cache_operations",
    "CacheMetadata",
    "CacheStatistics"
]

