"""Top-level package for the DADA + Hybrid Augmentation TensorFlow implementation.

This package reorganizes the original single-file training script into a clearer
module structure suitable for thesis review without changing runtime behavior.
"""

from .config import cfg  # re-export for convenience

__all__ = [
    "cfg",
]
