"""
Cache Manager (MODULAR)

This file delegates to modular cache components for better maintainability.
The actual implementation has been split into:
- cache/manager.py: ProfessionalCacheManager
- cache/operations.py: Cache operations and decorators

This file maintains backward compatibility through delegation.
"""

import logging
import warnings

# Import from new modular components
from .cache.manager import (
    ProfessionalCacheManager,
    CacheEntry,
    CacheStatistics,
    CacheMetadata,
    get_cache_manager,
    create_professional_cache
)
from .cache.operations import (
    cache_decorator,
    clear_cache_operations,
    monitor_cache_performance
)

logger = logging.getLogger(__name__)

# ===== BACKWARD COMPATIBILITY EXPORTS =====
__all__ = [
    # Main manager (now modular)
    "ProfessionalCacheManager",
    "CacheEntry",
    "CacheStatistics", 
    "CacheMetadata",
    "get_cache_manager",
    "create_professional_cache",
    
    # Operations (now modular)
    "cache_decorator",
    "clear_cache_operations",
    "monitor_cache_performance"
]

# ===== DEPRECATION NOTICE =====
warnings.warn(
    "Direct imports from cache_manager.py are deprecated. "
    "Use modular imports: from data.cache import ProfessionalCacheManager, cache_decorator",
    DeprecationWarning,
    stacklevel=2
)

logger.info("📦 Using modular cache components (cache_manager.py → cache/)")

