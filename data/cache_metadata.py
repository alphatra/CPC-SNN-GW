"""
Cache Metadata: Basic Structures and Metadata Management
Extracted from cache_manager.py for modular architecture.
"""

import time
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheMetadata:
    """Metadata for cached entries."""
    filename: str
    sha256: str
    created_at: float  # Unix timestamp
    last_accessed: float  # Unix timestamp
    file_size: int  # Bytes
    cache_key: str
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheMetadata':
        """Create from dictionary."""
        return cls(**data)
    
    def is_expired(self, max_age_days: int = 30) -> bool:
        """Check if cache entry is expired."""
        age_seconds = time.time() - self.created_at
        return age_seconds > (max_age_days * 24 * 3600)
    
    def update_access_time(self) -> None:
        """Update last accessed time."""
        self.last_accessed = time.time()


@dataclass
class CacheStatistics:
    """Cache usage statistics."""
    total_entries: int = 0
    total_size_bytes: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0
    
    @property
    def size_mb(self) -> float:
        """Get total size in megabytes."""
        return self.total_size_bytes / (1024 * 1024)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_entries': self.total_entries,
            'total_size_bytes': self.total_size_bytes,
            'total_size_mb': self.size_mb,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'eviction_count': self.eviction_count,
            'hit_rate': self.hit_rate
        }


class LRUCacheWithTTL:
    """In-memory LRU cache with TTL (Time To Live) support."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """
        Initialize LRU cache with TTL.
        
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time to live for entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: List[str] = []
        
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        return time.time() - entry['timestamp'] > self.ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self._cache:
            return None
            
        entry = self._cache[key]
        
        # Check if expired
        if self._is_expired(entry):
            self._remove_key(key)
            return None
        
        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        return entry['value']
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        current_time = time.time()
        
        # Remove if already exists
        if key in self._cache:
            self._remove_key(key)
        
        # Check if we need to evict
        while len(self._cache) >= self.max_size:
            oldest_key = self._access_order[0]
            self._remove_key(oldest_key)
        
        # Add new entry
        self._cache[key] = {
            'value': value,
            'timestamp': current_time
        }
        self._access_order.append(key)
    
    def _remove_key(self, key: str) -> None:
        """Remove key from cache."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()
    
    def size(self) -> int:
        """Get number of entries in cache."""
        return len(self._cache)


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        SHA256 hash string
    """
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Failed to compute hash for {file_path}: {e}")
        return ""


def verify_file_integrity(file_path: Path, expected_hash: str) -> bool:
    """
    Verify file integrity using SHA256 hash.
    
    Args:
        file_path: Path to file
        expected_hash: Expected SHA256 hash
        
    Returns:
        True if file integrity is verified
    """
    if not file_path.exists():
        return False
    
    actual_hash = compute_file_hash(file_path)
    return actual_hash == expected_hash


def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate a cache key from arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        SHA256 hash as cache key
    """
    # Create a deterministic string representation
    key_data = {
        'args': [str(arg) for arg in args],
        'kwargs': {k: str(v) for k, v in sorted(kwargs.items())}
    }
    
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_string.encode()).hexdigest()[:16]


def load_metadata_file(metadata_path: Path) -> Dict[str, CacheMetadata]:
    """
    Load metadata from JSON file.
    
    Args:
        metadata_path: Path to metadata file
        
    Returns:
        Dictionary of cache metadata
    """
    if not metadata_path.exists():
        return {}
    
    try:
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        metadata = {}
        for key, meta_dict in data.items():
            metadata[key] = CacheMetadata.from_dict(meta_dict)
        
        return metadata
    
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_path}: {e}")
        return {}


def save_metadata_file(metadata: Dict[str, CacheMetadata], 
                      metadata_path: Path) -> bool:
    """
    Save metadata to JSON file.
    
    Args:
        metadata: Dictionary of cache metadata
        metadata_path: Path to metadata file
        
    Returns:
        True if successful
    """
    try:
        # Convert to serializable format
        data = {}
        for key, meta in metadata.items():
            data[key] = meta.to_dict()
        
        # Create directory if needed
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write atomically
        temp_path = metadata_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Atomic rename
        temp_path.rename(metadata_path)
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to save metadata to {metadata_path}: {e}")
        return False


def get_cache_dir(app_name: str = "ligo-cpc-snn") -> Path:
    """
    Get platform-appropriate cache directory.
    
    Args:
        app_name: Application name for subdirectory
        
    Returns:
        Path to cache directory
    """
    import os
    import platform
    
    system = platform.system()
    
    if system == "Darwin":  # macOS
        cache_dir = Path.home() / "Library" / "Caches" / app_name
    elif system == "Windows":
        cache_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / app_name
    else:  # Linux and others
        cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / app_name
    
    return cache_dir


def get_cache_size(cache_dir: Path) -> int:
    """
    Get total size of cache directory in bytes.
    
    Args:
        cache_dir: Cache directory path
        
    Returns:
        Total size in bytes
    """
    if not cache_dir.exists():
        return 0
    
    total_size = 0
    try:
        for file_path in cache_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    except Exception as e:
        logger.error(f"Error calculating cache size: {e}")
    
    return total_size


def cleanup_expired_entries(metadata: Dict[str, CacheMetadata],
                          cache_dir: Path,
                          max_age_days: int = 30) -> int:
    """
    Clean up expired cache entries.
    
    Args:
        metadata: Cache metadata dictionary
        cache_dir: Cache directory
        max_age_days: Maximum age in days
        
    Returns:
        Number of entries cleaned up
    """
    cleanup_count = 0
    expired_keys = []
    
    for key, meta in metadata.items():
        if meta.is_expired(max_age_days):
            # Remove file
            file_path = cache_dir / meta.filename
            try:
                if file_path.exists():
                    file_path.unlink()
                expired_keys.append(key)
                cleanup_count += 1
            except Exception as e:
                logger.error(f"Failed to remove expired file {file_path}: {e}")
    
    # Remove from metadata
    for key in expired_keys:
        del metadata[key]
    
    return cleanup_count 