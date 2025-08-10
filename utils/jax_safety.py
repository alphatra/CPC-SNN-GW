"""
JAX safety utilities for stable array creation across backends (CPU/CUDA/Metal).

- Avoids creating JAX arrays directly from Python lists on problematic backends
  by first creating NumPy arrays and then transferring with device_put.
"""

from typing import Any, Iterable, Optional

import numpy as np
import jax


def safe_array_to_device(data: Any, dtype: Optional[np.dtype] = None):
    """Create a device array safely via NumPy then device_put.

    Parameters
    ----------
    data: Any
        Input data convertible to a NumPy array
    dtype: Optional[np.dtype]
        Desired dtype of the resulting array
    """
    host_array = np.array(data, dtype=dtype) if dtype is not None else np.array(data)
    return jax.device_put(host_array)


def safe_stack_to_device(items: Iterable[Any], dtype: Optional[np.dtype] = None):
    """Stack a sequence of arrays/items safely and transfer to device.

    Each item is converted to a NumPy array before stacking.
    """
    host_stack = np.stack([np.array(x) for x in items])
    if dtype is not None:
        host_stack = host_stack.astype(dtype)
    return jax.device_put(host_stack)

